import os
import sys
import json
import time
import argparse
import logging
import cv2
import numpy as np
import torch
import psutil

# Add parent directory to path so adframe can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adframe.vision.vision_model import VisionModel
from adframe.config import config, VisionBackend
from adframe.world_model.world_model import WorldModel
from adframe.world_model.memory_fusion import SceneMemoryFusion
from adframe.planner.placement_planner import PlacementPlanner

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("run_world_model")

def get_gpu_metrics():
    try:
        import subprocess
        output = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits"
        ]).decode().strip()
        parts = output.split(",")
        mem_used = float(parts[0].strip())
        util = float(parts[1].strip())
        return mem_used, util
    except Exception:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(0) / (1024 ** 2), 0.0
        return 0.0, 0.0

def get_peak_ram():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2) # RSS RAM in MiB

def extract_frame_from_video(video_path: str, frame_idx: int, output_path: str):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Target video not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx >= total_frames or frame_idx < 0:
        cap.release()
        raise ValueError(f"Requested frame index {frame_idx} is out of bounds (total frames: {total_frames})")
        
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Failed to read frame at index {frame_idx}")
        
    cv2.imwrite(output_path, frame)
    height, width, _ = frame.shape
    cap.release()
    return width, height

def save_world_model_debug(frame_path: str, wm: WorldModel, output_path: str):
    image = cv2.imread(frame_path)
    if image is None:
        logger.error(f"Failed to load frame image for debugging annotation from {frame_path}")
        return
        
    height, width, _ = image.shape
    overlay = image.copy()
    
    # 1. Draw surfaces (light blue, BGR (255, 120, 0))
    for surf in wm.surfaces:
        color = (255, 120, 0)
        label = f"Surface: {surf['surface_id']} ({surf.get('material', 'wood')})"
        bbox = surf["bbox"]
        ymin, xmin, ymax, xmax = bbox
        
        cv2.rectangle(overlay, (int(xmin * width), int(ymin * height)), (int(xmax * width), int(ymax * height)), color, -1)
        cv2.rectangle(image, (int(xmin * width), int(ymin * height)), (int(xmax * width), int(ymax * height)), color, 2)
        cv2.putText(image, label, (int(xmin * width), int(ymin * height) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # 2. Draw objects (light red, BGR (50, 50, 255))
    for obj in wm.objects:
        color = (50, 50, 255)
        # Use last bbox in bbox_history
        if obj.get("bbox_history"):
            bbox = obj["bbox_history"][-1]
            ymin, xmin, ymax, xmax = bbox
            label = f"Object: {obj['object_id']}"
            
            cv2.rectangle(overlay, (int(xmin * width), int(ymin * height)), (int(xmax * width), int(ymax * height)), color, -1)
            cv2.rectangle(image, (int(xmin * width), int(ymin * height)), (int(xmax * width), int(ymax * height)), color, 2)
            cv2.putText(image, label, (int(xmin * width), int(ymin * height) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            
            # Draw trajectory path
            traj = obj.get("trajectory", [])
            for pt in traj:
                cx, cy = pt
                cv2.circle(image, (int(cx * width), int(cy * height)), 3, (0, 255, 255), -1)

    # 3. Draw placement candidates (neon green, BGR (0, 255, 100))
    for reg in wm.placement_regions:
        color = (0, 255, 100)
        label = f"Candidate: {reg['region_id']} (Stability: {reg['stability_score']:.2f})"
        bbox = reg["bbox"]
        ymin, xmin, ymax, xmax = bbox
        
        cv2.rectangle(overlay, (int(xmin * width), int(ymin * height)), (int(xmax * width), int(ymax * height)), color, -1)
        cv2.rectangle(image, (int(xmin * width), int(ymin * height)), (int(xmax * width), int(ymax * height)), color, 3)
        cv2.putText(image, label, (int(xmin * width), int(ymin * height) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    # Blend overlay with original image
    cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)
    
    # Draw timeline & statistics summaries in corner
    cv2.rectangle(image, (10, 10), (450, 120), (0, 0, 0), -1)
    
    cam_info = f"Camera movement: {wm.statistics.get('camera_movement', 'unknown')}"
    light_info = f"Lighting temp: {wm.statistics.get('average_lighting_temperature', 4000.0):.1f}K"
    surface_info = f"Tracked Surfaces: {len(wm.surfaces)}, Objects: {len(wm.objects)}"
    timeline_info = f"Timeline length: {len(wm.camera_timeline)} keyframes"
    
    cv2.putText(image, "ADFRAME WORLD MODEL TRACKER", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, cam_info, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, light_info, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, surface_info, (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, timeline_info, (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Save to disk
    cv2.imwrite(output_path, image)
    logger.info(f"Saved visual debug frame to {output_path}")

def run_world_model_analysis(args):
    # Ensure folders exist
    os.makedirs(args.output, exist_ok=True)
    scene_graphs_dir = os.path.join(args.output, "scene_graphs")
    os.makedirs(scene_graphs_dir, exist_ok=True)
    
    # Load JSON schema
    current_dir = os.path.dirname(os.path.abspath(__file__))
    schema_path = os.path.join(current_dir, "..", "adframe", "schema", "scene_graph_schema.json")
    with open(schema_path, "r") as f:
        scene_graph_schema = json.load(f)
        
    gpu_before_load, _ = get_gpu_metrics()
    
    # 1. Load Vision Model
    logger.info(f"Loading Vision Model using backend: {args.backend}...")
    t_start_load = time.time()
    vlm = VisionModel(backend=args.backend)
    t_end_load = time.time()
    load_time = t_end_load - t_start_load
    logger.info(f"Model loaded successfully in {load_time:.2f} seconds.")
    
    gpu_after_load, _ = get_gpu_metrics()
    
    # Initialize World Model & Fusion Engine
    wm = WorldModel(scene_id="podcast_room_model")
    fusion = SceneMemoryFusion(wm)
    
    inference_times = []
    fused_graphs_count = 0
    t_start_fusion_cpu = time.time()
    
    # 2. Extract and Process each specified keyframe
    last_frame_path = None
    width, height = 960, 540
    
    for frame_idx in args.frames:
        frame_filename = f"frame_{frame_idx:03d}.png"
        frame_path = os.path.join(args.output, frame_filename)
        last_frame_path = frame_path
        
        # Extract frame
        width, height = extract_frame_from_video(args.video, frame_idx, frame_path)
        
        # Inference Prompt
        prompt = (
            "You are an expert VFX scene understanding engine for virtual product placement. "
            "Analyze this video frame to build a comprehensive Scene Graph. "
            "1. Identify the scene details, camera position, pitch, yaw, roll, and fov_estimate. "
            "2. Analyze lighting condition (type, direction, temperature, intensity). "
            "3. Locate and segment up to 3 major horizontal and vertical surfaces (materials, Orientation). "
            "4. Segment all visible objects, depth order, and occlusion properties. "
            "5. Detect all empty placement regions on horizontal surfaces. "
            "6. Calculate and rank placement candidates with scores (0.0 to 1.0) and recommended product size. "
            "Format all coordinate bounding boxes [ymin, xmin, ymax, xmax] as normalized float coordinates on a 0.0 to 1.0 scale. "
            "Polygons must be lists of [x, y] coordinates in the range 0.0 to 1.0. "
            f"Output must be a single JSON object strictly matching this schema:\n{json.dumps(scene_graph_schema, indent=2)}"
        )
        
        # Query Model
        logger.info(f"Executing scene graph query for frame {frame_idx}...")
        t_start_inf = time.time()
        raw_response = vlm.query(prompt, image_paths=[frame_path], expected_schema=scene_graph_schema)
        t_end_inf = time.time()
        
        inf_duration = t_end_inf - t_start_inf
        inference_times.append(inf_duration)
        logger.info(f"Frame {frame_idx} inference completed in {inf_duration:.2f} seconds.")
        
        # Parse & Validate
        try:
            sg_data = vlm.query_json(prompt, image_paths=[frame_path], expected_schema=scene_graph_schema)
            # Normalize Qwen coords
            for obj in sg_data.get("objects", []):
                if "bbox" in obj:
                    obj["bbox"] = [float(v) / 1000.0 if v > 1.0 else float(v) for v in obj["bbox"]]
            for surf in sg_data.get("surfaces", []):
                if "bbox" in surf:
                    surf["bbox"] = [float(v) / 1000.0 if v > 1.0 else float(v) for v in surf["bbox"]]
            for reg in sg_data.get("empty_regions", []):
                if "bbox" in reg:
                    reg["bbox"] = [float(v) / 1000.0 if v > 1.0 else float(v) for v in reg["bbox"]]
            for cand in sg_data.get("placement_candidates", []):
                if "bbox" in cand:
                    cand["bbox"] = [float(v) / 1000.0 if v > 1.0 else float(v) for v in cand["bbox"]]
            
            # Save frame scene graph
            sg_path = os.path.join(scene_graphs_dir, f"scene_graph_{frame_idx:03d}.json")
            with open(sg_path, "w") as sf:
                json.dump(sg_data, sf, indent=2)
                
            # Fuse into World Model
            fusion.fuse_scene_graph(sg_data, frame_idx)
            fused_graphs_count += 1
            
        except Exception as e:
            logger.error(f"Fusing frame {frame_idx} failed: {e}")
            if args.backend == "qwen":
                raise e

    t_end_fusion_cpu = time.time()
    fusion_cpu_time = t_end_fusion_cpu - t_start_fusion_cpu
    
    # 3. Save World Model Outputs
    world_model_path = os.path.join(args.output, "world_model.json")
    wm.save_to_json(world_model_path)
    
    # Save World Statistics
    stats_path = os.path.join(args.output, "world_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(wm.statistics, f, indent=2)
    logger.info(f"Saved world_statistics.json to {stats_path}")
    
    # Save Placement History
    history_path = os.path.join(args.output, "placement_history.json")
    with open(history_path, "w") as f:
        json.dump(wm.placement_regions, f, indent=2)
    logger.info(f"Saved placement_history.json to {history_path}")
    
    # 4. Run Placement Planner (consuming world_model.json)
    planner = PlacementPlanner()
    planner_data = planner.plan_placement(
        scene_graph=wm.to_dict(),
        product_metadata={"product_id": "luxury_perfume", "name": "luxury perfume bottle"}
    )
    
    planner_path = os.path.join(args.output, "planner.json")
    with open(planner_path, "w") as f:
        json.dump(planner_data, f, indent=2)
    logger.info(f"Saved planner.json using WorldModel at {planner_path}")
    
    # 5. Visual Debugging Image
    debug_path = os.path.join(args.output, "world_model_debug.png")
    if last_frame_path:
        save_world_model_debug(last_frame_path, wm, debug_path)
        
    # Get peak resource usages
    gpu_after_inf, gpu_util = get_gpu_metrics()
    peak_ram = get_peak_ram()
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
    else:
        peak_vram = 0.0
        
    # 6. Save Benchmark Metrics
    metrics_path = os.path.join(args.output, "metrics.json")
    metrics = {
        "number_of_processed_frames": len(args.frames),
        "number_of_fused_scene_graphs": fused_graphs_count,
        "world_model_build_time_sec": load_time + sum(inference_times) + fusion_cpu_time,
        "gpu_inference_total_time_sec": sum(inference_times),
        "cpu_fusion_time_sec": fusion_cpu_time,
        "peak_ram_mib": peak_ram,
        "peak_vram_mib": peak_vram if peak_vram > 0 else (gpu_after_inf - gpu_after_load),
        "gpu_utilization_percent": gpu_util,
        "frame_resolution": f"{width}x{height}"
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved final world metrics to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdFrame v2 Scene Memory Fusion & World Model Compiler")
    parser.add_argument("--video", type=str, default="/workspace/demo.mp4", help="Path to input video file")
    parser.add_argument("--frames", type=int, nargs="+", default=[120, 135, 150], help="List of frames to extract and fuse")
    parser.add_argument("--output", type=str, default="./outputs", help="Directory to save output files")
    parser.add_argument("--backend", type=str, default="qwen", choices=["qwen", "mock"], help="Model backend: qwen or mock")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Hugging Face model ID")
    
    args = parser.parse_args()
    
    config.vlm_model_id = args.model_id
    config.vision_backend = args.backend
    
    run_world_model_analysis(args)
