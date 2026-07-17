import os
import sys
import json
import time
import argparse
import logging
import torch
import psutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# Add parent directory to path so adframe can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adframe.config import config
from adframe.planner.placement_planner import PlacementPlanner
from adframe.generation.flux_generator import FluxGenerator

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("run_product_placement")

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

def load_product_metadata(product_arg):
    if os.path.exists(product_arg):
        logger.info(f"Loading product metadata from JSON file: {product_arg}")
        with open(product_arg, 'r') as f:
            return json.load(f)
    
    # Dynamic mapping if it's a type string
    logger.info(f"Resolving product metadata dynamically for key: '{product_arg}'")
    normalized = product_arg.lower().replace(" ", "_").replace("-", "_")
    if "coca_cola" in normalized or "coke" in normalized:
        return {
            "product_id": "coca_cola_bottle",
            "name": "Coca-Cola Bottle",
            "category": "beverage",
            "dimensions": {"height_cm": 21, "width_cm": 6.5},
            "preferred_surfaces": ["desk", "table", "counter", "tray"],
            "avoid_surfaces": ["wall", "monitor", "keyboard", "human face", "hand"],
            "preferred_orientation": "upright",
            "minimum_visibility": 0.70
        }
    elif "perfume" in normalized:
        return {
            "product_id": "luxury_perfume",
            "name": "luxury perfume bottle",
            "category": "cosmetics",
            "dimensions": {"height_cm": 12, "width_cm": 7},
            "preferred_surfaces": ["desk", "table", "vanity", "counter"],
            "avoid_surfaces": ["wall", "monitor", "floor", "keyboard", "human face"],
            "preferred_orientation": "upright",
            "minimum_visibility": 0.60
        }
    elif "mug" in normalized or "coffee" in normalized:
        return {
            "product_id": "coffee_mug",
            "name": "ceramic coffee mug",
            "category": "beverage",
            "dimensions": {"height_cm": 10, "width_cm": 12},
            "preferred_surfaces": ["desk", "table", "counter", "nightstand"],
            "avoid_surfaces": ["wall", "monitor", "keyboard"],
            "preferred_orientation": "upright",
            "minimum_visibility": 0.75
        }
    elif "laptop" in normalized:
        return {
            "product_id": "ultrabook_laptop",
            "name": "modern open laptop",
            "category": "electronics",
            "dimensions": {"height_cm": 2, "width_cm": 35, "depth_cm": 25},
            "preferred_surfaces": ["desk", "table", "bench"],
            "avoid_surfaces": ["wall", "monitor", "floor", "sofa", "keyboard"],
            "preferred_orientation": "flat",
            "minimum_visibility": 0.80
        }
    elif "smartphone" in normalized or "phone" in normalized:
        return {
            "product_id": "premium_smartphone",
            "name": "smartphone screen facing up",
            "category": "electronics",
            "dimensions": {"height_cm": 1, "width_cm": 15, "depth_cm": 7.5},
            "preferred_surfaces": ["desk", "table", "sofa", "counter"],
            "avoid_surfaces": ["wall", "floor"],
            "preferred_orientation": "flat",
            "minimum_visibility": 0.80
        }
    elif "energy_drink" in normalized or "drink" in normalized or "can" in normalized:
        return {
            "product_id": "energy_drink_can",
            "name": "energy drink can",
            "category": "beverage",
            "dimensions": {"height_cm": 15, "width_cm": 6},
            "preferred_surfaces": ["desk", "table", "counter"],
            "avoid_surfaces": ["wall", "monitor", "keyboard", "human face"],
            "preferred_orientation": "upright",
            "minimum_visibility": 0.70
        }
    else:
        return {
            "product_id": normalized,
            "name": product_arg,
            "category": "generic",
            "dimensions": {"height_cm": 15, "width_cm": 8},
            "preferred_surfaces": ["desk", "table", "counter"],
            "avoid_surfaces": ["wall", "monitor"],
            "preferred_orientation": "upright",
            "minimum_visibility": 0.60
        }

def draw_polygon_overlay(img, polygon_norm, color, thickness=2, fill_alpha=0.3):
    h, w, _ = img.shape
    pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in polygon_norm], np.int32)
    if pts.size == 0:
        return img
    pts = pts.reshape((-1, 1, 2))
    
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.polylines(overlay, [pts], True, color, thickness, lineType=cv2.LINE_AA)
    
    return cv2.addWeighted(overlay, fill_alpha, img, 1.0 - fill_alpha, 0)

def generate_visualization(frame_path, candidates, product_name, camera_dir, output_path):
    img = cv2.imread(frame_path)
    if img is None:
        raise FileNotFoundError(f"Frame image not found for drawing: {frame_path}")
        
    h, w, _ = img.shape
    
    # Palette for up to 5 candidates
    colors = [
        (255, 0, 128),  # neon pink (Rank 1)
        (0, 255, 128),  # bright mint (Rank 2)
        (0, 128, 255),  # electric blue (Rank 3)
        (255, 128, 0),  # vibrant orange (Rank 4)
        (128, 0, 255)   # royal purple (Rank 5)
    ]
    
    for i, cand in enumerate(candidates):
        color = colors[i % len(colors)]
        polygon = cand.get("polygon", [])
        
        # Draw overlay
        img = draw_polygon_overlay(img, polygon, color, thickness=3, fill_alpha=0.25)
        
        # Compute label center
        if polygon:
            xs = [p[0] * w for p in polygon]
            ys = [p[1] * h for p in polygon]
            cx, cy = int(sum(xs) / len(xs)), int(sum(ys) / len(ys))
        else:
            bbox = cand.get("bbox", [0.4, 0.4, 0.6, 0.6])
            cx = int(((bbox[1] + bbox[3]) / 2) * w)
            cy = int(((bbox[0] + bbox[2]) / 2) * h)
            
        # Draw small tag
        label = f"#{i+1}: {product_name} (Score: {cand['overall_score']:.2f})"
        cv2.circle(img, (cx, cy), 6, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(img, label, (cx + 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
    # Draw camera indicator at top-left
    cv2.rectangle(img, (10, 10), (220, 45), (30, 30, 30), -1)
    cv2.putText(img, f"Camera Direction: {camera_dir}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    
    cv2.imwrite(output_path, img)
    logger.info(f"Saved candidate overlays to {output_path}")

def main(args):
    t_start_all = time.time()
    
    # 1. Load configuration and input paths
    os.makedirs(args.output, exist_ok=True)
    
    if not os.path.exists(args.frame):
        raise FileNotFoundError(f"Frame image not found: {args.frame}")
    if not os.path.exists(args.world_model):
        raise FileNotFoundError(f"World model file not found: {args.world_model}")
        
    with open(args.world_model, "r") as f:
        world_model_data = json.load(f)
        
    product_metadata = load_product_metadata(args.product)
    
    # Query system resources before run
    gpu_before_load, _ = get_gpu_metrics()
    ram_before = psutil.virtual_memory().used / (1024 ** 2)
    
    # 2. Run PlacementPlanner
    t_start_plan = time.time()
    planner = PlacementPlanner()
    
    # Get config candidate limit or default to 5
    max_cands = 5
    planner_result = planner.plan_placement(
        scene_graph=world_model_data,
        product_metadata=product_metadata,
        max_candidates=max_cands
    )
    t_end_plan = time.time()
    plan_time = t_end_plan - t_start_plan
    
    # 3. Save planner.json
    planner_path = os.path.join(args.output, "planner.json")
    with open(planner_path, "w") as f:
        json.dump(planner_result, f, indent=2)
    logger.info(f"Saved planner.json to {planner_path}")
    
    candidates = planner_result.get("placement_candidates", [])
    
    # 4. Save placement_analysis.json (debugging reasoning details)
    analysis_data = {}
    for i, cand in enumerate(candidates):
        cid = cand["candidate_id"]
        analysis_data[cid] = {
            "rank": i + 1,
            "overall_score": cand["overall_score"],
            "confidence": cand["confidence"],
            "reason_selected": cand["reason"],
            "score_breakdown": {
                "surface_score": cand["surface_score"],
                "occlusion_score": cand["occlusion_score"],
                "camera_score": cand["camera_score"],
                "lighting_score": cand["lighting_score"],
                "realism_score": cand["realism_score"]
            },
            "brand_safety": "high" if cand["occlusion_score"] > 0.8 else ("medium" if cand["occlusion_score"] > 0.5 else "low"),
            "surface_reasoning": f"Surface {cand['surface_id']} matches product preference logic.",
            "camera_reasoning": f"Estimated distance {cand['estimated_distance_from_camera']:.2f}m fits visibility bounds."
        }
        
    analysis_path = os.path.join(args.output, "placement_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis_data, f, indent=2)
    logger.info(f"Saved placement_analysis.json to {analysis_path}")
    
    # 5. Save brand_report.json for the top candidate
    if candidates:
        top_cand = candidates[0]
        brand_report = {
            "product_id": product_metadata.get("product_id", "generic"),
            "product_name": product_metadata["name"],
            "expected_visibility": top_cand["visibility_score"],
            "estimated_logo_visibility": top_cand["estimated_logo_visibility"],
            "estimated_screen_occupancy": float((top_cand["bbox"][2]-top_cand["bbox"][0]) * (top_cand["bbox"][3]-top_cand["bbox"][1])),
            "distance_from_camera_meters": top_cand["estimated_distance_from_camera"],
            "occlusion_probability": top_cand["occlusion_score"],
            "brand_safety": "passed" if top_cand["occlusion_score"] > 0.75 else "review_required",
            "placement_confidence": top_cand["confidence"],
            "overall_placement_quality": "excellent" if top_cand["overall_score"] > 0.4 else "acceptable"
        }
    else:
        brand_report = {}
        
    brand_path = os.path.join(args.output, "brand_report.json")
    with open(brand_path, "w") as f:
        json.dump(brand_report, f, indent=2)
    logger.info(f"Saved brand_report.json to {brand_path}")
    
    # 6. Save placement_candidates.png overlays
    camera_dir = "facing center"
    cam_info = world_model_data.get("camera", {})
    if cam_info:
        camera_dir = f"pitch: {cam_info.get('pitch', 0.0):.1f}, yaw: {cam_info.get('yaw', 0.0):.1f}"
        
    vis_path = os.path.join(args.output, "placement_candidates.png")
    generate_visualization(
        frame_path=args.frame,
        candidates=candidates,
        product_name=product_metadata["name"],
        camera_dir=camera_dir,
        output_path=vis_path
    )
    
    # 7. Render ONLY Top 2 candidates using FLUX Fill
    t_start_render = time.time()
    
    generator = FluxGenerator(backend=args.backend, provider=args.provider)
    if args.backend == "flux":
        logger.info("Initializing and loading FLUX Fill pipeline...")
        generator.load_model()
        
    rendered_files = []
    
    for i in range(min(2, len(candidates))):
        cand = candidates[i]
        logger.info(f"Rendering Candidate {i+1} (ID: {cand['candidate_id']}) via FLUX Fill...")
        
        # Override plan bbox and prompt for this candidate
        cand_plan = planner_result.copy()
        cand_plan["placement"] = {
            "bbox_2d": cand["bbox"],
            "target_surface_id": cand["surface_id"]
        }
        cand_plan["scale"] = cand["recommended_scale"]
        
        cand_output_path = os.path.join(args.output, f"candidate_{i+1}.png")
        
        generator.generate(
            image_path=args.frame,
            planner_data=cand_plan,
            output_path=cand_output_path
        )
        rendered_files.append(cand_output_path)
        
    t_end_render = time.time()
    render_time = t_end_render - t_start_render
    
    # 8. Create comparison.png grid
    if len(rendered_files) >= 2:
        logger.info("Assembling comparison.png grid...")
        orig_pil = Image.open(args.frame).convert("RGB")
        w, h = orig_pil.size
        
        # Side-by-side grid: Original, Candidate 1, Candidate 2
        comp_img = Image.new("RGB", (w * 3, h))
        comp_img.paste(orig_pil, (0, 0))
        comp_img.paste(Image.open(rendered_files[0]), (w, 0))
        comp_img.paste(Image.open(rendered_files[1]), (w * 2, 0))
        
        # Draw labels on top of grid sections
        draw = ImageDraw.Draw(comp_img)
        # Draw semi-transparent tag bars
        for idx, text in enumerate(["ORIGINAL FRAME", "CANDIDATE 1 (Rank #1)", "CANDIDATE 2 (Rank #2)"]):
            draw.rectangle([idx * w + 10, 10, idx * w + 260, 40], fill=(30, 30, 30))
            draw.text((idx * w + 20, 15), text, fill=(255, 255, 255))
            
        comp_path = os.path.join(args.output, "comparison.png")
        comp_img.save(comp_path)
        logger.info(f"Saved comparison grid to {comp_path}")
        
    # Query system resources after run
    gpu_after_inf, _ = get_gpu_metrics()
    ram_after = psutil.virtual_memory().used / (1024 ** 2)
    
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
    else:
        peak_vram = 0.0
        
    # 9. Save placement_metrics.json
    metrics_path = os.path.join(args.output, "placement_metrics.json")
    metrics_data = {
        "planner_inference_time_sec": plan_time,
        "render_time_sec": render_time,
        "gpu_vram_mib": peak_vram if peak_vram > 0 else (gpu_after_inf - gpu_before_load),
        "cpu_ram_used_mib": ram_after - ram_before,
        "candidate_scores": [float(c["overall_score"]) for c in candidates],
        "product_type": product_metadata["name"],
        "number_of_candidates_generated": len(candidates)
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    logger.info(f"Saved placement_metrics.json to {metrics_path}")
    
    t_end_all = time.time()
    logger.info(f"Evaluation pipeline completed in {t_end_all - t_start_all:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", required=True, help="Path to input frame image")
    parser.add_argument("--world-model", required=True, help="Path to world_model.json")
    parser.add_argument("--product", required=True, help="Product type string or path to product.json")
    parser.add_argument("--output", required=True, help="Path to output folder directory")
    parser.add_argument("--provider", default="modelscope", help="Checkpoints registry provider")
    parser.add_argument("--backend", default="flux", choices=["flux", "mock"], help="Generation backend")
    
    args = parser.parse_args()
    main(args)
