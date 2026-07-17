import json
import logging
from typing import Dict, Any, Optional
from adframe.vision.vision_model import VisionModel

logger = logging.getLogger("adframe.planner")

import os
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
schema_path = os.path.join(current_dir, "..", "schema", "planner_schema.json")
try:
    with open(schema_path, "r") as f:
        PLANNER_SCHEMA = json.load(f)
except Exception as e:
    logger = logging.getLogger("adframe.planner")
    logger.error(f"Failed to load planner schema from {schema_path}: {e}")
    PLANNER_SCHEMA = {}

# Backward compatibility
PLACEMENT_PLAN_SCHEMA = {"title": "PlacementPlan", "type": "OBJECT"}

class PlacementPlanner:
    """
    Placement Planner computes the optimal bounding box and rendering instructions
    for placing a target product into the scene. It uses Scene Memory as its input
    and prioritizes physical realism (lighting alignment, perspective, stability)
    over product visibility.
    """
    def __init__(self, vision_model: Optional[VisionModel] = None):
        self.vlm = vision_model

    def plan_placement(self, scene_graph: Optional[Dict[str, Any]] = None, product_metadata: Optional[Dict[str, Any]] = None, scene_memory_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Processes the Scene Graph or WorldModel to choose and rank candidates, producing a planner.json layout.
        Also retains backward-compatibility keys for legacy callers/tests.
        """
        if scene_graph is None:
            scene_graph = scene_memory_state or {}
        if product_metadata is None:
            product_metadata = {}
            
        logger.info(f"Planning product placement for product: {product_metadata.get('product_id', 'unknown')}")
        
        # Extract metadata
        prod_name = product_metadata.get("name", "product")
        
        # 1. Normalize/Ensure placement candidates exist (checking "placement_regions" first for WorldModel)
        regions = scene_graph.get("placement_regions", [])
        candidates = []
        
        if regions:
            for reg in regions:
                candidates.append({
                    "candidate_id": reg.get("region_id"),
                    "surface": reg.get("surface_id"),
                    "polygon": reg.get("polygon", []),
                    "bbox": reg.get("bbox", reg.get("bbox_2d", [0.4, 0.4, 0.6, 0.6])),
                    "score": reg.get("stability_score", 1.0) * reg.get("available_area", 0.8),
                    "stability_score": reg.get("stability_score", 1.0),
                    "reason": f"Tracked region {reg.get('region_id')} (stability: {reg.get('stability_score', 1.0):.2f})",
                    "recommended_product_size": reg.get("recommended_product_size", "medium"),
                    "camera_visibility": 0.95,
                    "risk": reg.get("occlusion_probability", 0.05),
                    "confidence": reg.get("confidence", 0.90)
                })
        else:
            candidates = scene_graph.get("placement_candidates", [])
            
        # Fallback if no placement candidates are defined in the scene_graph or empty
        if not candidates:
            # Try to build candidates from empty_regions or surfaces
            empty_regions = scene_graph.get("empty_regions", [])
            surfaces = scene_graph.get("surfaces", [])
            
            for i, region in enumerate(empty_regions):
                surf_id = region.get("surface_id", "default_surface")
                surf = next((s for s in surfaces if s.get("surface_id") == surf_id), None)
                surf_label = surf.get("label", "surface") if surf else "surface"
                
                # Check orientation
                orientation = surf.get("orientation", "horizontal") if surf else "horizontal"
                # Give horizontal regions higher base score
                base_score = 0.9 if orientation == "horizontal" else 0.5
                
                candidates.append({
                    "candidate_id": region.get("region_id", f"candidate_{i}"),
                    "surface": surf_id,
                    "polygon": region.get("polygon", []),
                    "bbox": region.get("bbox", region.get("bbox_2d", [0.4, 0.4, 0.6, 0.6])),
                    "score": base_score,
                    "stability_score": 1.0,
                    "reason": f"Derived from empty region {region.get('region_id')}",
                    "recommended_product_size": "medium container",
                    "camera_visibility": 0.95,
                    "risk": 0.05,
                    "confidence": 0.90
                })
            
            # If still no candidates, check surfaces
            if not candidates and surfaces:
                for i, surf in enumerate(surfaces):
                    orientation = surf.get("orientation", "horizontal")
                    base_score = 0.8 if orientation == "horizontal" else 0.4
                    candidates.append({
                        "candidate_id": f"candidate_surface_{i}",
                        "surface": surf.get("surface_id", f"surf_{i}"),
                        "polygon": surf.get("polygon", []),
                        "bbox": surf.get("bbox", surf.get("bbox_2d", [0.4, 0.4, 0.6, 0.6])),
                        "score": base_score,
                        "stability_score": 1.0,
                        "reason": f"Derived from surface {surf.get('surface_id')}",
                        "recommended_product_size": "medium container",
                        "camera_visibility": 0.90,
                        "risk": 0.1,
                        "confidence": 0.85
                    })
        
        # Default candidate if absolutely nothing exists
        if not candidates:
            candidates.append({
                "candidate_id": "default_center",
                "surface": "default_surface",
                "polygon": [],
                "bbox": [0.4, 0.4, 0.6, 0.6],
                "score": 0.5,
                "stability_score": 1.0,
                "reason": "Fallback default",
                "recommended_product_size": "medium",
                "camera_visibility": 1.0,
                "risk": 0.0,
                "confidence": 1.0
            })
            
        # 2. Automatically rank candidates by temporal stability score primary, and visual score secondary
        ranked_candidates = sorted(candidates, key=lambda x: (x.get("stability_score", 1.0), x.get("score", 0.0)), reverse=True)
        top_candidate = ranked_candidates[0]
        
        # Get target surface details
        target_surf_id = top_candidate["surface"]
        surfaces = scene_graph.get("surfaces", [])
        target_surface = next((s for s in surfaces if s.get("surface_id") == target_surf_id), None)
        material = target_surface.get("material", "wood") if target_surface else "wood"
        reflection = target_surface.get("reflection_strength", 0.15) if target_surface else 0.15
        
        # Get lighting & camera info
        lighting = scene_graph.get("lighting", {})
        lighting_dir = lighting.get("direction", "top-right")
        lighting_type = lighting.get("type", "ambient")
        
        camera = scene_graph.get("camera", {})
        camera_pitch = camera.get("pitch", -10.0)
        
        # Construct rendering and negative constraints
        rendering_constraints = {
            "shadow": "soft" if lighting.get("ambient", 0.3) > 0.2 else "hard",
            "reflection": reflection,
            "lighting": f"{lighting_type} ({lighting_dir})",
            "camera_pitch": camera_pitch
        }
        
        negative_constraints = [
            "avoid face",
            "avoid keyboard",
            "avoid monitor"
        ]
        
        # Generate prompt for FLUX in legacy key (backward compatibility)
        prompt_text = f"a luxury {prod_name} sitting upright on a {material} surface, blending with the surroundings, cinematic lighting from the {lighting_dir}, realistic shadows"
        
        # Build the unified plan dictionary containing both new and old fields
        plan = {
            # New schema fields (complying with planner_schema.json)
            "target_surface": target_surf_id,
            "placement_candidate": top_candidate["candidate_id"],
            "rendering_constraints": rendering_constraints,
            "negative_constraints": negative_constraints,
            
            # Legacy fields for backward compatibility
            "placement": {
                "bbox_2d": top_candidate["bbox"],
                "target_surface_id": target_surf_id
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
            "prompt": prompt_text,
            "negative_prompt": "floating, bad lighting, cropped, blurry, low quality"
        }
        
        return plan
