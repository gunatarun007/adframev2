import json
import logging
from typing import Dict, Any, Optional
from adframe.vision.vision_model import VisionModel

logger = logging.getLogger("adframe.planner")

PLACEMENT_PLAN_SCHEMA = {
    "title": "PlacementPlan",
    "type": "OBJECT",
    "properties": {
        "placement": {
            "type": "OBJECT",
            "properties": {
                "bbox_2d": { "type": "ARRAY", "items": { "type": "NUMBER" }, "minItems": 4, "maxItems": 4 },
                "target_surface_id": { "type": "STRING" }
            },
            "required": ["bbox_2d", "target_surface_id"]
        },
        "rotation": {
            "type": "OBJECT",
            "properties": {
                "yaw": { "type": "NUMBER" },
                "pitch": { "type": "NUMBER" },
                "roll": { "type": "NUMBER" }
            },
            "required": ["yaw", "pitch", "roll"]
        },
        "scale": { "type": "NUMBER" },
        "visibility": {
            "type": "OBJECT",
            "properties": {
                "occluded_by": { "type": "ARRAY", "items": { "type": "STRING" } },
                "visible_percentage": { "type": "NUMBER" }
            },
            "required": ["visible_percentage"]
        },
        "prompt": { "type": "STRING" },
        "negative_prompt": { "type": "STRING" },
        "rendering_constraints": {
            "type": "OBJECT",
            "properties": {
                "lighting_direction": { "type": "STRING" },
                "shadow_softness": { "type": "STRING" }
            },
            "required": ["lighting_direction", "shadow_softness"]
        }
    },
    "required": ["placement", "rotation", "scale", "visibility", "prompt", "rendering_constraints"]
}

class PlacementPlanner:
    """
    Placement Planner computes the optimal bounding box and rendering instructions
    for placing a target product into the scene. It uses Scene Memory as its input
    and prioritizes physical realism (lighting alignment, perspective, stability)
    over product visibility.
    """
    def __init__(self, vision_model: Optional[VisionModel] = None):
        self.vlm = vision_model

    def plan_placement(self, scene_memory_state: Dict[str, Any], product_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determines the optimal placement strategy using the VLM (or deterministic rule fallback).
        """
        logger.info(f"Planning product placement for product: {product_metadata.get('product_id', 'unknown')}")
        
        if self.vlm:
            try:
                # Compile a high-fidelity semantic prompt for the VLM to choose the best empty region
                prompt = (
                    f"You are the Placement Planner in an AI product placement pipeline. Given the current Scene Memory:\n"
                    f"{self._format_scene_memory_for_prompt(scene_memory_state)}\n\n"
                    f"And the target product metadata:\n"
                    f"{self._format_product_metadata_for_prompt(product_metadata)}\n\n"
                    f"Determine the most realistic empty placement region and output a detailed PlacementPlan JSON. "
                    f"Prioritize realism: place horizontal products (bottles, boxes) on horizontal surfaces, align lighting directions, "
                    f"ensure perspective scales, and respect occlusion. Never float products. "
                    f"Generate a rendering prompt for FLUX Fill describing the product in its new environment, blending with the target surface material."
                )
                
                plan = self.vlm.query_json(
                    prompt=prompt,
                    image_paths=None,  # VLM reasons semantically over scene memory json
                    expected_schema=PLACEMENT_PLAN_SCHEMA
                )
                return plan
            except Exception as e:
                logger.error(f"VLM placement planning failed: {e}. Falling back to deterministic planner.")
        
        return self._deterministic_fallback_plan(scene_memory_state, product_metadata)

    def _format_scene_memory_for_prompt(self, state: Dict[str, Any]) -> str:
        # Simplify scene memory for prompt reading
        simplified = {
            "room_type": state.get("room_type", "unknown"),
            "lighting": state.get("lighting", {}),
            "surfaces": [
                {
                    "id": s["surface_id"],
                    "label": s["label"],
                    "material": s["material"],
                    "orientation": s["orientation"]
                }
                for s in state.get("surfaces", [])
            ],
            "empty_regions": state.get("empty_regions", [])
        }
        return json.dumps(simplified, indent=2)

    def _format_product_metadata_for_prompt(self, meta: Dict[str, Any]) -> str:
        return json.dumps(meta, indent=2)

    def _deterministic_fallback_plan(self, state: Dict[str, Any], product: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determines placement deterministically if the VLM is unavailable.
        """
        logger.warning("Executing deterministic fallback placement plan.")
        
        # Pick the first horizontal surface with an empty region
        target_region = None
        target_surface = None
        
        surfaces = state.get("surfaces", [])
        empty_regions = state.get("empty_regions", [])
        
        for region in empty_regions:
            surface_id = region.get("surface_id")
            surface = next((s for s in surfaces if s["surface_id"] == surface_id), None)
            if surface and surface.get("orientation") == "horizontal":
                target_region = region
                target_surface = surface
                break
                
        # Default region if none found
        if not target_region and empty_regions:
            target_region = empty_regions[0]
            surface_id = target_region.get("surface_id")
            target_surface = next((s for s in surfaces if s["surface_id"] == surface_id), None)
            
        bbox = target_region.get("bbox_2d", [0.4, 0.4, 0.6, 0.6]) if target_region else [0.4, 0.4, 0.6, 0.6]
        surface_id = target_region.get("surface_id", "default_surface") if target_region else "default_surface"
        material = target_surface.get("material", "surface") if target_surface else "surface"
        
        prod_name = product.get("name", "product")
        lighting_dir = state.get("lighting", {}).get("direction", "top")
        
        # Simple generated prompt
        prompt = f"a realistic {prod_name} sitting on the {material}, matching shadows, realistic {lighting_dir} light, highly detailed, 4k"
        
        return {
            "placement": {
                "bbox_2d": bbox,
                "target_surface_id": surface_id
            },
            "rotation": {
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0
            },
            "scale": 1.0,
            "visibility": {
                "occluded_by": [],
                "visible_percentage": 100.0
            },
            "prompt": prompt,
            "negative_prompt": "floating, deformed, wrong perspective, poorly composited",
            "rendering_constraints": {
                "lighting_direction": lighting_dir,
                "shadow_softness": "soft"
            }
        }
