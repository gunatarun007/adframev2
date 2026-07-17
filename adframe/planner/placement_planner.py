import json
import logging
from typing import Dict, Any, Optional, List
import os
import math

logger = logging.getLogger("adframe.planner")

current_dir = os.path.dirname(os.path.abspath(__file__))
schema_path = os.path.join(current_dir, "..", "schema", "planner_schema.json")
try:
    with open(schema_path, "r") as f:
        PLANNER_SCHEMA = json.load(f)
except Exception as e:
    logger.error(f"Failed to load planner schema from {schema_path}: {e}")
    PLANNER_SCHEMA = {}

# Backward compatibility
PLACEMENT_PLAN_SCHEMA = {"title": "PlacementPlan", "type": "OBJECT"}

class PlacementPlanner:
    """
    Placement Planner computes the optimal polygon/bounding box and rendering instructions
    for placing a target product into the scene. It reasons dynamically using product metadata.
    """
    def __init__(self, vision_model: Optional[Any] = None):
        self.vlm = vision_model

    def plan_placement(
        self,
        scene_graph: Optional[Dict[str, Any]] = None,
        product_metadata: Optional[Dict[str, Any]] = None,
        scene_memory_state: Optional[Dict[str, Any]] = None,
        max_candidates: int = 5
    ) -> Dict[str, Any]:
        """
        Processes the Scene Graph or WorldModel to choose and rank candidates, producing a planner.json layout.
        Reasons dynamically based on product metadata guidelines.
        """
        if scene_graph is None:
            scene_graph = scene_memory_state or {}
        if product_metadata is None:
            product_metadata = {}

        logger.info(f"Planning product placement for product: {product_metadata.get('product_id', product_metadata.get('name', 'unknown'))}")

        # Fallback default product metadata if not specified
        prod_name = product_metadata.get("name", "luxury perfume bottle")
        prod_category = product_metadata.get("category", "cosmetics")
        prod_dims = product_metadata.get("dimensions", {"height_cm": 15, "width_cm": 8})
        pref_surfaces = product_metadata.get("preferred_surfaces", ["desk", "table", "counter", "surface"])
        avoid_surfaces = product_metadata.get("avoid_surfaces", ["wall", "monitor", "keyboard"])
        pref_orient = product_metadata.get("preferred_orientation", "upright")
        min_visibility = product_metadata.get("minimum_visibility", 0.60)

        # 1. Gather regions/surfaces from Scene Graph / WorldModel
        regions = scene_graph.get("placement_regions", [])
        surfaces = scene_graph.get("surfaces", [])
        empty_regions = scene_graph.get("empty_regions", [])
        objects = scene_graph.get("objects", [])

        # Derive candidates from surfaces to ensure we always have diverse options across the scene
        derived_candidates = []
        for i, surf in enumerate(surfaces):
            surf_id = surf.get("surface_id", f"surf_{i}")
            surf_bbox = surf.get("bbox", surf.get("bbox_2d", [0.2, 0.2, 0.8, 0.8]))
            ymin, xmin, ymax, xmax = surf_bbox
            
            w_box = xmax - xmin
            h_box = ymax - ymin
            
            # We want each candidate region to take about 15% width / 20% height of the surface
            cw = 0.15
            ch = 0.20
            
            # Define 5 distinct anchor centers:
            anchors = [
                ("center", xmin + w_box * 0.5, ymin + h_box * 0.5),
                ("left", xmin + w_box * 0.25, ymin + h_box * 0.5),
                ("right", xmin + w_box * 0.75, ymin + h_box * 0.5),
                ("front", xmin + w_box * 0.5, ymin + h_box * 0.75),
                ("back", xmin + w_box * 0.5, ymin + h_box * 0.25)
            ]
            
            for name, cx, cy in anchors:
                # clamp within bounds
                c_ymin = max(ymin, cy - ch / 2)
                c_xmin = max(xmin, cx - cw / 2)
                c_ymax = min(ymax, cy + ch / 2)
                c_xmax = min(xmax, cx + cw / 2)
                
                derived_candidates.append({
                    "region_id": f"region_{surf_id}_{name}",
                    "surface_id": surf_id,
                    "polygon": [
                        [c_xmin, c_ymin],
                        [c_xmax, c_ymin],
                        [c_xmax, c_ymax],
                        [c_xmin, c_ymax]
                    ],
                    "bbox": [c_ymin, c_xmin, c_ymax, c_xmax],
                    "stability_score": float(surf.get("confidence", 0.9) * (0.8 if name != "center" else 1.0)),
                    "confidence": float(surf.get("confidence", 0.9)),
                    "occlusion_probability": 0.05 if name != "front" else 0.15,
                    "recommended_product_size": "medium",
                    "available_area": 0.8
                })

        raw_candidates = []
        # Prepend original regions if they exist
        for reg in regions:
            raw_candidates.append({
                "region_id": reg.get("region_id"),
                "surface_id": reg.get("surface_id"),
                "polygon": reg.get("polygon", []),
                "bbox": reg.get("bbox", reg.get("bbox_2d", [0.4, 0.4, 0.6, 0.6])),
                "stability_score": reg.get("stability_score", 1.0),
                "confidence": reg.get("confidence", 0.90),
                "occlusion_probability": reg.get("occlusion_probability", 0.05),
                "recommended_product_size": reg.get("recommended_product_size", "medium"),
                "available_area": reg.get("available_area", 0.8)
            })

        raw_candidates.extend(derived_candidates)

        # Fallback to empty_regions if absolutely empty
        if not raw_candidates:
            for i, reg in enumerate(empty_regions):
                raw_candidates.append({
                    "region_id": reg.get("region_id", f"region_{i}"),
                    "surface_id": reg.get("surface_id", "default_surface"),
                    "polygon": reg.get("polygon", []),
                    "bbox": reg.get("bbox", reg.get("bbox_2d", [0.4, 0.4, 0.6, 0.6])),
                    "stability_score": 1.0,
                    "confidence": 0.85,
                    "occlusion_probability": 0.05,
                    "recommended_product_size": "medium",
                    "available_area": 0.7
                })

        if not raw_candidates:
            # Absolute fallback
            raw_candidates.append({
                "region_id": "default_center",
                "surface_id": "default_surface",
                "polygon": [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]],
                "bbox": [0.4, 0.4, 0.6, 0.6],
                "stability_score": 0.5,
                "confidence": 0.70,
                "occlusion_probability": 0.0,
                "recommended_product_size": "medium",
                "available_area": 0.5
            })

        # 2. Score Candidates using Product Metadata
        scored_candidates = []
        for reg in raw_candidates:
            candidate_id = reg["region_id"]
            surf_id = reg["surface_id"]
            bbox = reg["bbox"]
            polygon = reg["polygon"]

            # Locate matching surface details
            surface_obj = next((s for s in surfaces if s.get("surface_id") == surf_id), None)
            surf_label = (surface_obj.get("label", "surface") if surface_obj else "surface").lower()
            surf_material = surface_obj.get("material", "wood") if surface_obj else "wood"
            surf_orientation = (surface_obj.get("orientation", "horizontal") if surface_obj else "horizontal").lower()
            surf_reflection = surface_obj.get("reflection_strength", 0.15) if surface_obj else 0.15

            # Calculate individual sub-scores
            # A. Surface Score: matching preferred and avoided surfaces
            surf_score = 0.5
            # Preference match
            if any(pref in surf_label for pref in pref_surfaces):
                surf_score = 0.9
            # Avoidance match
            if any(av in surf_label for av in avoid_surfaces):
                surf_score = 0.1

            # Orientation check
            if pref_orient == "upright" and surf_orientation == "vertical":
                surf_score *= 0.3 # Penalize placing upright items on walls

            # B. Occlusion Score: based on probability and brand safety (avoid overlap with humans/important objects)
            base_occlusion = 1.0 - reg["occlusion_probability"]
            # Brand safety check: does it overlap with faces or hands?
            brand_safety_factor = 1.0
            for obj in objects:
                obj_label = obj.get("label", "").lower()
                if any(k in obj_label for k in ["face", "hand", "person"]):
                    # simple overlap or distance penalty based on bbox center distance
                    obj_bbox = obj.get("bbox_2d", [0,0,0,0])
                    dist = math.sqrt((bbox[0]-obj_bbox[0])**2 + (bbox[1]-obj_bbox[1])**2)
                    if dist < 0.2:
                        brand_safety_factor = min(brand_safety_factor, 0.2)

            occlusion_score = base_occlusion * brand_safety_factor

            # C. Camera Score: based on distance and camera pitch
            camera = scene_graph.get("camera", {})
            camera_pitch = camera.get("pitch", -10.0)
            camera_dist_factor = 1.0
            
            # Estimate distance from camera based on scale (bounding box area)
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area > 0:
                estimated_distance = 1.0 / math.sqrt(area)
            else:
                estimated_distance = 2.0
                
            # Ideal distance is between 1.0 and 3.0 meters
            if estimated_distance < 0.8:
                camera_dist_factor = 0.6 # Too close / cropped
            elif estimated_distance > 4.0:
                camera_dist_factor = 0.5 # Too far / low resolution

            camera_score = camera_dist_factor * (1.0 - abs(camera_pitch) / 90.0)

            # D. Lighting Score: lighting matches requirements
            lighting = scene_graph.get("lighting", {})
            ambient_light = lighting.get("ambient", 0.5)
            lighting_score = ambient_light

            # E. Realism Score: combination of stability and orientation logic
            realism_score = reg["stability_score"] * (0.9 if surf_orientation == "horizontal" else 0.5)

            # F. Overall Score: weighted product of sub-scores
            overall_score = float(surf_score * occlusion_score * camera_score * lighting_score * realism_score)

            # G. Recommended Scale
            recommended_scale = float(max(0.4, min(1.5, reg["available_area"] * 1.2)))

            # H. Logo / Screen visibility estimation
            logo_vis = float(min(1.0, max(0.0, occlusion_score * 0.9)))
            screen_vis = float(min(1.0, max(0.0, occlusion_score * 0.95)))

            scored_candidates.append({
                "candidate_id": candidate_id,
                "surface_id": surf_id,
                "polygon": polygon if polygon else [[bbox[1], bbox[0]], [bbox[3], bbox[0]], [bbox[3], bbox[2]], [bbox[1], bbox[2]]],
                "bbox": bbox,
                "visibility_score": float(occlusion_score),
                "realism_score": float(realism_score),
                "camera_score": float(camera_score),
                "occlusion_score": float(occlusion_score),
                "lighting_score": float(lighting_score),
                "surface_score": float(surf_score),
                "overall_score": float(overall_score),
                "confidence": float(reg["confidence"]),
                "reason": f"Placed on {surf_label} surface ({surf_material}) with stability {reg['stability_score']:.2f}",
                "recommended_scale": recommended_scale,
                "estimated_logo_visibility": logo_vis,
                "estimated_screen_visibility": screen_vis,
                "estimated_distance_from_camera": float(estimated_distance)
            })

        # 3. Sort and Rank candidates
        ranked_candidates = sorted(scored_candidates, key=lambda x: x["overall_score"], reverse=True)[:max_candidates]

        # Ensure we have at least one top candidate to build the legacy dict
        top_candidate = ranked_candidates[0] if ranked_candidates else {
            "candidate_id": "default",
            "surface_id": "default",
            "polygon": [],
            "bbox": [0.4, 0.4, 0.6, 0.6],
            "overall_score": 0.5,
            "surface_score": 0.5,
            "occlusion_score": 0.5,
            "camera_score": 0.5,
            "lighting_score": 0.5,
            "realism_score": 0.5,
            "confidence": 0.8,
            "reason": "Fallback default",
            "recommended_scale": 0.85,
            "estimated_logo_visibility": 0.8,
            "estimated_screen_visibility": 0.8,
            "estimated_distance_from_camera": 2.0
        }

        # Build legacy/top-level planner fields
        lighting = scene_graph.get("lighting", {})
        lighting_dir = lighting.get("direction", "top-right")
        lighting_type = lighting.get("type", "ambient")
        
        target_surf_id = top_candidate["surface_id"]
        target_surface = next((s for s in surfaces if s.get("surface_id") == target_surf_id), None)
        material = target_surface.get("material", "wood") if target_surface else "wood"
        reflection = target_surface.get("reflection_strength", 0.15) if target_surface else 0.15
        camera_pitch = scene_graph.get("camera", {}).get("pitch", -10.0)

        rendering_constraints = {
            "shadow": "soft" if lighting.get("ambient", 0.3) > 0.2 else "hard",
            "reflection": float(reflection),
            "lighting": f"{lighting_type} ({lighting_dir})",
            "camera_pitch": float(camera_pitch)
        }

        # Generate prompt for FLUX Fill using product constraints
        prompt_text = f"a realistic {prod_name} standing {pref_orient} on a {material} surface, blending with surroundings, casting shadows, lighting from {lighting_dir}"

        plan = {
            # Required fields for planner_schema.json
            "target_surface": target_surf_id,
            "placement_candidate": top_candidate["candidate_id"],
            "rendering_constraints": rendering_constraints,
            "negative_constraints": ["floating", "blurry", "cropped", "low quality", "human hands", "avoid face"],

            # Backward compatibility fields
            "placement": {
                "bbox_2d": top_candidate["bbox"],
                "target_surface_id": target_surf_id
            },
            "rotation": {
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0
            },
            "scale": float(top_candidate["recommended_scale"]),
            "visibility": {
                "occluded_by": [],
                "visible_percentage": float(top_candidate["visibility_score"] * 100.0)
            },
            "prompt": prompt_text,
            "negative_prompt": "floating, bad lighting, cropped, blurry, low quality",
            
            # Top Candidates List
            "placement_candidates": ranked_candidates
        }

        return plan

