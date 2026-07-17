import os
import sys
import json
import time
import argparse
import logging
import re
import cv2
import numpy as np
import torch

# Add parent directory to path so adframe can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adframe.vision.vision_model import VisionModel
from adframe.config import config, VisionBackend
from adframe.planner.placement_planner import PlacementPlanner

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("run_scene_analysis")

def get_gpu_metrics():
    """
    Queries nvidia-smi for precise physical VRAM usage (in MiB) and GPU utilization (%).
    """
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
    except Exception as e:
        logger.warning(f"Failed to query nvidia-smi: {e}. Defaulting to PyTorch memory stats.")
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(0) / (1024 ** 2), 0.0
        return 0.0, 0.0

def extract_frame_from_video(video_path: str, frame_idx: int, output_path: str):
    """
    Extracts a specific frame index from the video and saves it.
    """
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
    logger.info(f"Successfully extracted frame {frame_idx} (resolution: {width}x{height}) to {output_path}")
    return width, height

def normalize_bbox(bbox):
    if not bbox or len(bbox) != 4:
        return bbox
    if any(val > 1.0 for val in bbox):
        return [float(val) / 1000.0 for val in bbox]
    return [float(val) for val in bbox]

def normalize_polygon(poly):
    if not poly:
        return poly
    needs_norm = False
    for pt in poly:
        if len(pt) == 2 and (pt[0] > 1.0 or pt[1] > 1.0):
            needs_norm = True
            break
    if needs_norm:
        return [[float(pt[0]) / 1000.0, float(pt[1]) / 1000.0] for pt in poly]
    return [[float(pt[0]), float(pt[1])] for pt in poly]

def normalize_scene_graph(sg: dict) -> dict:
    for obj in sg.get("objects", []):
        if "bbox" in obj:
            obj["bbox"] = normalize_bbox(obj["bbox"])
        if "polygon" in obj:
            obj["polygon"] = normalize_polygon(obj["polygon"])
            
    for surf in sg.get("surfaces", []):
        if "bbox" in surf:
            surf["bbox"] = normalize_bbox(surf["bbox"])
        if "polygon" in surf:
            surf["polygon"] = normalize_polygon(surf["polygon"])
            
    for region in sg.get("empty_regions", []):
        if "bbox" in region:
            region["bbox"] = normalize_bbox(region["bbox"])
        if "polygon" in region:
            region["polygon"] = normalize_polygon(region["polygon"])
            
    for cand in sg.get("placement_candidates", []):
        if "bbox" in cand:
            cand["bbox"] = normalize_bbox(cand["bbox"])
        if "polygon" in cand:
            cand["polygon"] = normalize_polygon(cand["polygon"])
            
    for occ in sg.get("occlusions", []):
        if "occlusion_bbox" in occ:
            occ["occlusion_bbox"] = normalize_bbox(occ["occlusion_bbox"])
            
    return sg

def save_annotated_frame(frame_path: str, scene_graph: dict, output_path: str):
    image = cv2.imread(frame_path)
    if image is None:
        logger.error(f"Failed to load frame image for annotation from {frame_path}")
        return
        
    height, width, _ = image.shape
    overlay = image.copy()
    
    # 1. Draw surfaces (light blue, BGR (255, 120, 0))
    for surf in scene_graph.get("surfaces", []):
        color = (255, 120, 0)
        label = f"{surf.get('label', 'Surface')} ({surf.get('material', 'unknown')})"
        poly = surf.get("polygon", [])
        bbox = surf.get("bbox", [])
        
        if poly:
            pts = np.array([[int(pt[0] * width), int(pt[1] * height)] for pt in poly], np.int32)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
        elif bbox:
            ymin, xmin, ymax, xmax = bbox
            cv2.rectangle(overlay, (int(xmin * width), int(ymin * height)), (int(xmax * width), int(ymax * height)), color, -1)
            cv2.rectangle(image, (int(xmin * width), int(ymin * height)), (int(xmax * width), int(ymax * height)), color, 2)
            
        box = bbox if bbox else ([poly[0][1], poly[0][0], poly[0][1], poly[0][0]] if poly else [0.1, 0.1, 0.1, 0.1])
        ymin, xmin, _, _ = box
        cv2.putText(image, label, (int(xmin * width), int(ymin * height) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # 2. Draw objects (light red, BGR (50, 50, 255))
    for obj in scene_graph.get("objects", []):
        color = (50, 50, 255)
        label = f"{obj.get('label', 'Object')}"
        poly = obj.get("polygon", [])
        bbox = obj.get("bbox", [])
        
        if poly:
            pts = np.array([[int(pt[0] * width), int(pt[1] * height)] for pt in poly], np.int32)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
        elif bbox:
            ymin, xmin, ymax, xmax = bbox
            cv2.rectangle(overlay, (int(xmin * width), int(ymin * height)), (int(xmax * width), int(ymax * height)), color, -1)
            cv2.rectangle(image, (int(xmin * width), int(ymin * height)), (int(xmax * width), int(ymax * height)), color, 2)
            
        box = bbox if bbox else ([poly[0][1], poly[0][0], poly[0][1], poly[0][0]] if poly else [0.1, 0.1, 0.1, 0.1])
        ymin, xmin, _, _ = box
        cv2.putText(image, label, (int(xmin * width), int(ymin * height) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # 3. Draw empty regions (yellow, BGR (0, 230, 230))
    for reg in scene_graph.get("empty_regions", []):
        color = (0, 230, 230)
        label = f"Empty: {reg.get('region_id', 'Region')}"
        poly = reg.get("polygon", [])
        bbox = reg.get("bbox", [])
        
        if poly:
            pts = np.array([[int(pt[0] * width), int(pt[1] * height)] for pt in poly], np.int32)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
        elif bbox:
            ymin, xmin, ymax, xmax = bbox
            cv2.rectangle(overlay, (int(xmin * width), int(ymin * height)), (int(xmax * width), int(ymax * height)), color, -1)
            cv2.rectangle(image, (int(xmin * width), int(ymin * height)), (int(xmax * width), int(ymax * height)), color, 2)
            
        box = bbox if bbox else ([poly[0][1], poly[0][0], poly[0][1], poly[0][0]] if poly else [0.1, 0.1, 0.1, 0.1])
        ymin, xmin, _, _ = box
        cv2.putText(image, label, (int(xmin * width), int(ymin * height) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # 4. Draw placement candidates (neon green, BGR (0, 255, 100))
    for cand in scene_graph.get("placement_candidates", []):
        color = (0, 255, 100)
        label = f"Candidate: {cand.get('candidate_id', 'Cand')} (Score: {cand.get('score', 0.0):.2f})"
        poly = cand.get("polygon", [])
        bbox = cand.get("bbox", [])
        
        if poly:
            pts = np.array([[int(pt[0] * width), int(pt[1] * height)] for pt in poly], np.int32)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(image, [pts], isClosed=True, color=color, thickness=3)
        elif bbox:
            ymin, xmin, ymax, xmax = bbox
            cv2.rectangle(overlay, (int(xmin * width), int(ymin * height)), (int(xmax * width), int(ymax * height)), color, -1)
            cv2.rectangle(image, (int(xmin * width), int(ymin * height)), (int(xmax * width), int(ymax * height)), color, 3)
            
        box = bbox if bbox else ([poly[0][1], poly[0][0], poly[0][1], poly[0][0]] if poly else [0.1, 0.1, 0.1, 0.1])
        ymin, xmin, _, _ = box
        cv2.putText(image, label, (int(xmin * width), int(ymin * height) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # Blend overlay with original image
    cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)
    
    # 5. Draw camera info
    cam = scene_graph.get("camera", {})
    cam_info = f"Camera: {cam.get('position', 'unknown')}, Pitch: {cam.get('pitch', 0.0):.1f}, Yaw: {cam.get('yaw', 0.0):.1f}"
    cv2.putText(image, cam_info, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, cam_info, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    # Save to disk
    cv2.imwrite(output_path, image)
    logger.info(f"Successfully saved annotated frame to {output_path}")

def run_analysis(args):
    os.makedirs(args.output, exist_ok=True)
    
    # Define filenames
    frame_path = os.path.join(args.output, "frame.png")
    prompt_path = os.path.join(args.output, "prompt.txt")
    raw_response_path = os.path.join(args.output, "raw_response.txt")
    scene_graph_path = os.path.join(args.output, "scene_graph.json")
    planner_path = os.path.join(args.output, "planner.json")
    metrics_path = os.path.join(args.output, "metrics.json")
    annotated_path = os.path.join(args.output, "annotated_frame.png")
    
    # Benchmark initial states
    gpu_before_load, _ = get_gpu_metrics()
    
    # 1. Extract Frame
    width, height = extract_frame_from_video(args.video, args.frame, frame_path)
    
    # 2. Load Vision Model
    logger.info(f"Loading Vision Model using backend: {args.backend}...")
    t_start_load = time.time()
    
    vlm = VisionModel(backend=args.backend)
    
    t_end_load = time.time()
    load_time = t_end_load - t_start_load
    logger.info(f"Model loaded successfully in {load_time:.2f} seconds.")
    
    gpu_after_load, _ = get_gpu_metrics()
    
    # Load JSON schema from disk
    current_dir = os.path.dirname(os.path.abspath(__file__))
    schema_file_path = os.path.join(current_dir, "..", "adframe", "schema", "scene_graph_schema.json")
    with open(schema_file_path, "r") as f:
        schema = json.load(f)
        
    # 3. Create prompt
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
        f"Output must be a single JSON object strictly matching this schema:\n{json.dumps(schema, indent=2)}"
    )
    
    with open(prompt_path, "w") as f:
        f.write(prompt)
    logger.info(f"Saved inference prompt template to {prompt_path}")
    
    # 4. Perform Inference
    logger.info("Executing scene graph query...")
    t_start_inf = time.time()
    
    # Count mock or estimate tokens
    prompt_tokens = len(prompt.split()) * 2  # raw estimate
    
    raw_response = vlm.query(prompt, image_paths=[frame_path], expected_schema=schema)
    
    t_end_inf = time.time()
    inference_time = t_end_inf - t_start_inf
    logger.info(f"Inference completed in {inference_time:.2f} seconds.")
    
    output_tokens = len(raw_response.split()) * 2  # raw estimate
    
    # Save raw output
    with open(raw_response_path, "w") as f:
        f.write(raw_response)
    logger.info(f"Saved raw response string to {raw_response_path}")
    
    # Parse and validate JSON
    json_validation_status = True
    t_start_val = time.time()
    scene_graph_data = {}
    try:
        scene_graph_data = vlm.query_json(prompt, image_paths=[frame_path], expected_schema=schema)
        scene_graph_data = normalize_scene_graph(scene_graph_data)
        
        # Verify using jsonschema if available
        try:
            import jsonschema
            jsonschema.validate(instance=scene_graph_data, schema=schema)
            logger.info("Schema validation with jsonschema PASSED.")
        except ImportError:
            logger.warning("jsonschema library not installed. Skipping strict verification.")
        except Exception as ve:
            logger.error(f"Strict schema validation FAILED: {ve}")
            raise ve
            
        with open(scene_graph_path, "w") as f:
            json.dump(scene_graph_data, f, indent=2)
        logger.info(f"Successfully generated and validated scene_graph.json at {scene_graph_path}")
    except Exception as e:
        logger.error(f"Structured JSON output validation failed: {e}")
        json_validation_status = False
        if args.backend == VisionBackend.QWEN:
            raise e
            
    json_validation_time = time.time() - t_start_val
            
    # 5. Build planner.json using PlacementPlanner
    planner_data = {}
    if json_validation_status:
        try:
            planner = PlacementPlanner()
            planner_data = planner.plan_placement(
                scene_graph=scene_graph_data,
                product_metadata={"product_id": "luxury_perfume", "name": "luxury perfume bottle"}
            )
            
            # Strict validation for planner output
            planner_schema_path = os.path.join(current_dir, "..", "adframe", "schema", "planner_schema.json")
            with open(planner_schema_path, "r") as pf:
                planner_schema = json.load(pf)
            try:
                import jsonschema
                # Validate only the keys present in planner_schema (target_surface, placement_candidate, etc)
                planner_subset = {
                    "target_surface": planner_data.get("target_surface"),
                    "placement_candidate": planner_data.get("placement_candidate"),
                    "rendering_constraints": planner_data.get("rendering_constraints"),
                    "negative_constraints": planner_data.get("negative_constraints")
                }
                jsonschema.validate(instance=planner_subset, schema=planner_schema)
                logger.info("Planner Schema validation PASSED.")
            except Exception as ve:
                logger.error(f"Planner validation error: {ve}")
                
            with open(planner_path, "w") as f:
                json.dump(planner_data, f, indent=2)
            logger.info(f"Generated planner.json at {planner_path}")
            
            # Generate annotated visualization
            save_annotated_frame(frame_path, scene_graph_data, annotated_path)
            
        except Exception as e:
            logger.error(f"Failed to generate planner config or annotation: {e}")
            
    # Get peak memory and utilization during execution
    gpu_after_inf, gpu_util = get_gpu_metrics()
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
    else:
        peak_vram = 0.0
        
    # 6. Save Benchmark Metrics
    metrics = {
        "model_id": args.model_id,
        "backend": args.backend,
        "frame_index": args.frame,
        "frame_resolution": f"{width}x{height}",
        "model_load_time_sec": load_time,
        "inference_time_sec": inference_time,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "gpu_memory_before_load_mib": gpu_before_load,
        "gpu_memory_after_load_mib": gpu_after_load,
        "gpu_memory_during_inference_mib": gpu_after_inf,
        "peak_vram_mib": peak_vram if peak_vram > 0 else (gpu_after_inf - gpu_after_load),
        "gpu_utilization_percent": gpu_util,
        "json_validation_status": json_validation_status,
        "json_validation_time_sec": json_validation_time
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Benchmarks saved to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdFrame v2 Scene Analysis & Benchmark Suite")
    parser.add_argument("--video", type=str, default="/workspace/demo.mp4", help="Path to input video file")
    parser.add_argument("--frame", type=int, default=120, help="Frame index to analyze")
    parser.add_argument("--output", type=str, default="./outputs", help="Directory to save artifacts")
    parser.add_argument("--backend", type=str, default="qwen", choices=["qwen", "mock"], help="Model backend: qwen or mock")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Hugging Face model ID")
    
    args = parser.parse_args()
    
    config.vlm_model_id = args.model_id
    config.vision_backend = args.backend
    
    run_analysis(args)
