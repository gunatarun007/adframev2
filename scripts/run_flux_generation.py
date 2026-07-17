import os
import sys
import json
import time
import argparse
import logging
import torch
import psutil

# Add parent directory to path so adframe can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adframe.config import config
from adframe.generation.flux_generator import FluxGenerator

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("run_flux_generation")

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

def run_generation(args):
    # Ensure folder path exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # 1. Read planner data
    if not os.path.exists(args.planner):
        raise FileNotFoundError(f"Planner file not found: {args.planner}")
        
    with open(args.planner, "r") as f:
        planner_data = json.load(f)
        
    # Query current GPU stats before model load
    gpu_before_load, _ = get_gpu_metrics()
    
    # 2. Initialize generator
    logger.info(f"Initializing FluxGenerator (backend: {args.backend}, provider: {args.provider})...")
    generator = FluxGenerator(backend=args.backend, provider=args.provider)
    
    # 3. Trigger model load explicitly
    logger.info("Loading FLUX pipeline...")
    t_start_load = time.time()
    generator.load_model()
    t_end_load = time.time()
    load_time = t_end_load - t_start_load
    logger.info(f"Pipeline ready in {load_time:.2f} seconds.")
    
    # Query GPU stats after load
    gpu_after_load, _ = get_gpu_metrics()
    
    # 4. Generate rendered output
    logger.info(f"Starting inpainting on {args.frame}...")
    metrics_run = generator.generate(
        image_path=args.frame,
        planner_data=planner_data,
        output_path=args.output
    )
    
    # Query GPU stats after inference
    gpu_after_inf, gpu_util = get_gpu_metrics()
    
    # Compute peak VRAM
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
    else:
        peak_vram = 0.0
        
    # 5. Output metrics
    metrics_path = args.output.replace(".png", "_metrics.json")
    metrics_data = {
        "model_id": config.flux_modelscope_id if args.provider == "modelscope" else config.flux_model_id,
        "backend": args.backend,
        "provider": args.provider,
        "model_load_time_sec": load_time,
        "inference_time_sec": metrics_run["inference_time_sec"],
        "gpu_memory_before_load_mib": gpu_before_load,
        "gpu_memory_after_load_mib": gpu_after_load,
        "gpu_memory_after_inference_mib": gpu_after_inf,
        "peak_vram_mib": peak_vram if peak_vram > 0 else (gpu_after_inf - gpu_before_load),
        "gpu_utilization_percent": gpu_util
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
        
    logger.info(f"Rendered frame successfully saved to {args.output}")
    logger.info(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdFrame v2 Phase 2: FLUX Inpainting Generator")
    parser.add_argument("--frame", type=str, required=True, help="Path to original input frame image")
    parser.add_argument("--planner", type=str, required=True, help="Path to planner.json config file")
    parser.add_argument("--output", type=str, default="./outputs/rendered_frame.png", help="Path to save output rendered image")
    parser.add_argument("--backend", type=str, default="flux", choices=["flux", "mock"], help="Pipeline backend: flux or mock")
    parser.add_argument("--provider", type=str, default="modelscope", choices=["modelscope", "huggingface", "local"], help="Model registry provider")
    
    args = parser.parse_args()
    
    run_generation(args)
