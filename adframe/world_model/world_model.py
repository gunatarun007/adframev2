import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger("adframe.world_model.world_model")

class WorldModel:
    """
    WorldModel represents the single source of truth for scene parameters compiled
    across a video timeline. It contains fused representations of objects, surfaces,
    lighting, camera movement statistics, and product placement regions.
    """
    def __init__(self, scene_id: str = "default_scene"):
        self.scene_id = scene_id
        self.duration_frames = 0
        
        # Current camera & lighting state estimates (latest averaged)
        self.camera_state = {
            "position": "center",
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "fov_estimate": 60.0,
            "camera_height": 1.5,
            "confidence": 0.90
        }
        
        self.lighting_state = {
            "type": "ambient",
            "direction": "top-right",
            "temperature": 4000.0,
            "intensity": 0.8,
            "ambient": 0.3,
            "confidence": 0.90
        }
        
        # Fused components
        self.surfaces: List[Dict[str, Any]] = []
        self.objects: List[Dict[str, Any]] = []
        self.placement_regions: List[Dict[str, Any]] = []
        
        # Timelines
        self.camera_timeline: List[Dict[str, Any]] = []
        self.lighting_timeline: List[Dict[str, Any]] = []
        self.object_timeline: List[Dict[str, Any]] = []
        
        # Statistics & Confidence metrics
        self.statistics = {
            "average_lighting_temperature": 4000.0,
            "lighting_stability": 0.90,
            "lighting_changes_detected": 0,
            "temperature_trend": "stable",
            "shadow_direction_trend": "stable",
            "reflection_trend": "stable",
            "camera_movement": "static",
            "camera_velocity": 0.0,
            "camera_stability": 0.95
        }
        
        self.confidence = {
            "overall_scene_confidence": 0.90,
            "surfaces_confidence": 0.90,
            "objects_confidence": 0.90
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the WorldModel state into a schema-compliant dictionary structure.
        """
        return {
            "scene": {
                "scene_id": self.scene_id,
                "duration_frames": self.duration_frames
            },
            "camera": self.camera_state,
            "lighting": self.lighting_state,
            "surfaces": self.surfaces,
            "objects": self.objects,
            "placement_regions": self.placement_regions,
            "timelines": {
                "camera": self.camera_timeline,
                "lighting": self.lighting_timeline,
                "objects": self.object_timeline
            },
            "statistics": self.statistics,
            "confidence": self.confidence
        }

    def save_to_json(self, path: str):
        """
        Dumps the world model to a JSON file and runs schema validation.
        """
        data = self.to_dict()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Saved world_model.json to {path}")
        
        # Validate schema
        current_dir = os.path.dirname(os.path.abspath(__file__))
        schema_path = os.path.join(current_dir, "..", "schema", "world_model_schema.json")
        if os.path.exists(schema_path):
            try:
                import jsonschema
                with open(schema_path, "r") as sf:
                    schema = json.load(sf)
                jsonschema.validate(instance=data, schema=schema)
                logger.info("world_model.json schema validation PASSED.")
            except ImportError:
                logger.warning("jsonschema library not found. Skipping strict schema validation.")
            except Exception as e:
                logger.error(f"world_model.json schema validation FAILED: {e}")
                raise e
