import os
import sys
import json
import time
import argparse
import logging
import re
import cv2
import torch

# Add parent directory to path so adframe can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adframe.vision.vision_model import VisionModel
from adframe.config import config, VisionBackend

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
    # If any value is > 1.0, it is likely on Qwen's 1000x1000 grid
    if any(val > 1.0 for val in bbox):
        return [float(val) / 1000.0 for val in bbox]
    return [float(val) for val in bbox]

def normalize_scene_memory(scene_mem: dict) -> dict:
    for obj in scene_mem.get("objects", []):
        if "bbox_2d" in obj:
            obj["bbox_2d"] = normalize_bbox(obj["bbox_2d"])
    for surf in scene_mem.get("surfaces", []):
        if "bbox_2d" in surf:
            surf["bbox_2d"] = normalize_bbox(surf["bbox_2d"])
    for region in scene_mem.get("empty_regions", []):
        if "bbox_2d" in region:
            region["bbox_2d"] = normalize_bbox(region["bbox_2d"])
    return scene_mem

def generate_planner_config(scene_mem: dict) -> dict:
    """
    Translates structured SceneMemory JSON into a valid planner.json configuration for FLUX.
    """
    # Defaults
    target_surface_id = "surface_default_1"
    bbox_2d = [0.55, 0.35, 0.75, 0.65] # center-mid default
    
    # 1. Search for surfaces/empty regions to place the object
    empty_regions = scene_mem.get("empty_regions", [])
    if empty_regions:
        best_region = empty_regions[0]
        bbox_2d = best_region.get("bbox_2d", bbox_2d)
        target_surface_id = best_region.get("surface_id", target_surface_id)
    else:
        surfaces = scene_mem.get("surfaces", [])
        # Find horizontal surfaces first
        horizontal_surfaces = [s for s in surfaces if s.get("orientation", "").lower() == "horizontal"]
        if horizontal_surfaces:
            target_surface_id = horizontal_surfaces[0].get("surface_id", target_surface_id)
            # Use lower-half of surface bbox as candidate placement area
            s_bbox = horizontal_surfaces[0].get("bbox_2d", bbox_2d)
            ymin, xmin, ymax, xmax = s_bbox
            bbox_2d = [ymin + (ymax - ymin)*0.2, xmin + (xmax - xmin)*0.2, ymax - (ymax - ymin)*0.2, xmax - (xmax - xmin)*0.2]

    # 2. Extract lighting details for rendering constraints
    lighting = scene_mem.get("lighting", {})
    lighting_dir = lighting.get("direction", "top-right")
    
    planner_json = {
        "placement": {
            "bbox_2d": bbox_2d,
            "target_surface_id": target_surface_id
        },
        "rotation": {
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0
        },
        "scale": 0.85,
        "visibility": {
            "occluded_by": [],
            "visible_percentage": 100.0
        },
        "prompt": "a luxury wrist watch standing upright, soft shadows, photorealistic, cinematic lighting, 8k",
        "negative_prompt": "floating, bad lighting, cropped, blurry, low quality",
        "rendering_constraints": {
            "lighting_direction": lighting_dir,
            "shadow_softness": "soft"
        }
    }
    return planner_json

def run_analysis(args):
    os.makedirs(args.output, exist_ok=True)
    
    # Define filenames
    frame_path = os.path.join(args.output, "frame.png")
    prompt_path = os.path.join(args.output, "prompt.txt")
    raw_response_path = os.path.join(args.output, "raw_response.txt")
    scene_memory_path = os.path.join(args.output, "scene_memory.json")
    planner_path = os.path.join(args.output, "planner.json")
    metrics_path = os.path.join(args.output, "metrics.json")
    
    # Benchmark initial states
    gpu_before_load, _ = get_gpu_metrics()
    
    # 1. Extract Frame
    width, height = extract_frame_from_video(args.video, args.frame, frame_path)
    
    # 2. Load Vision Model
    logger.info(f"Loading Vision Model using backend: {args.backend}...")
    t_start_load = time.time()
    
    # Explicitly instantiate VisionModel with correct backend
    vlm = VisionModel(backend=args.backend)
    
    t_end_load = time.time()
    load_time = t_end_load - t_start_load
    logger.info(f"Model loaded successfully in {load_time:.2f} seconds.")
    
    gpu_after_load, _ = get_gpu_metrics()
    
    # Define expected Schema
    schema = {
        "title": "SceneMemory",
        "type": "object",
        "properties": {
            "scene": {
                "type": "object",
                "properties": {
                    "scene_id": {"type": "string"},
                    "room_type": {"type": "string"}
                },
                "required": ["scene_id", "room_type"]
            },
            "camera": {
                "type": "object",
                "properties": {
                    "motion_type": {"type": "string"},
                    "direction": {"type": "string"},
                    "speed": {"type": "string"}
                },
                "required": ["motion_type", "direction", "speed"]
            },
            "lighting": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string"},
                    "color_temperature_k": {"type": "integer"}
                },
                "required": ["direction", "color_temperature_k"]
            },
            "objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "object_id": {"type": "string"},
                        "label": {"type": "string"},
                        "bbox_2d": {"type": "array", "minItems": 4, "maxItems": 4},
                        "depth_order": {"type": "integer"}
                    },
                    "required": ["object_id", "label", "bbox_2d"]
                }
            },
            "surfaces": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "surface_id": {"type": "string"},
                        "label": {"type": "string"},
                        "bbox_2d": {"type": "array", "minItems": 4, "maxItems": 4},
                        "material": {"type": "string"},
                        "orientation": {"type": "string"}
                    },
                    "required": ["surface_id", "label", "bbox_2d", "material", "orientation"]
                }
            },
            "empty_regions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "region_id": {"type": "string"},
                        "bbox_2d": {"type": "array", "minItems": 4, "maxItems": 4},
                        "surface_id": {"type": "string"},
                        "dimensions_px": {"type": "array", "minItems": 2, "maxItems": 2}
                    },
                    "required": ["region_id", "bbox_2d", "surface_id"]
                }
            }
        },
        "required": ["scene", "camera", "lighting", "objects", "surfaces", "empty_regions"]
    }
    
    # 3. Create prompt
    prompt = (
        "Identify the architectural structure, lighting condition, camera path motion type, "
        "and segment all horizontal and vertical surfaces (e.g. tables, floors, desks, walls) in this frame. "
        "Locate all empty region boxes on horizontal surfaces where a product container could be cleanly placed. "
        "Format coordinates as normalized float coordinates [ymin, xmin, ymax, xmax] in the range 0.0 to 1.0. "
        f"Output must be a single JSON object strictly matching the following schema:\n{json.dumps(schema, indent=2)}"
    )
    
    with open(prompt_path, "w") as f:
        f.write(prompt)
    logger.info(f"Saved inference prompt template to {prompt_path}")
    
    # 4. Perform Inference
    logger.info("Executing scene memory query...")
    t_start_inf = time.time()
    
    # Run the query. In QWEN mode this will hit Qwen2.5-VL-7B-Instruct, in MOCK it returns mock data
    raw_response = vlm.query(prompt, image_paths=[frame_path], expected_schema=schema)
    
    t_end_inf = time.time()
    inference_time = t_end_inf - t_start_inf
    logger.info(f"Inference completed in {inference_time:.2f} seconds.")
    
    # Save raw output
    with open(raw_response_path, "w") as f:
        f.write(raw_response)
    logger.info(f"Saved raw response string to {raw_response_path}")
    
    # Parse and validate JSON
    json_validation_status = True
    scene_memory_data = {}
    try:
        scene_memory_data = vlm.query_json(prompt, image_paths=[frame_path], expected_schema=schema)
        scene_memory_data = normalize_scene_memory(scene_memory_data)
        # Ensure we write valid structured JSON output
        with open(scene_memory_path, "w") as f:
            json.dump(scene_memory_data, f, indent=2)
        logger.info(f"Successfully generated and validated scene_memory.json at {scene_memory_path}")
    except Exception as e:
        logger.error(f"Structured JSON output validation failed: {e}")
        json_validation_status = False
        if args.backend == VisionBackend.QWEN:
            # Re-raise failure for true execution
            raise e
            
    # 5. Build planner.json
    planner_data = {}
    if json_validation_status:
        try:
            planner_data = generate_planner_config(scene_memory_data)
            with open(planner_path, "w") as f:
                json.dump(planner_data, f, indent=2)
            logger.info(f"Generated planner.json at {planner_path}")
        except Exception as e:
            logger.error(f"Failed to generate planner config: {e}")
            
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
        "gpu_memory_before_load_mib": gpu_before_load,
        "gpu_memory_after_load_mib": gpu_after_load,
        "gpu_memory_during_inference_mib": gpu_after_inf,
        "peak_vram_mib": peak_vram if peak_vram > 0 else (gpu_after_inf - gpu_after_load),
        "gpu_utilization_percent": gpu_util,
        "json_validation_status": json_validation_status
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
    
    # Inject values to global config context so VisionModel retrieves them
    config.vlm_model_id = args.model_id
    config.vision_backend = args.backend
    
    run_analysis(args)
