import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("adframe.scene_memory")

class SceneMemory:
    """
    Scene Memory serves as the central source of truth for scene attributes accumulated
    across video keyframes. Rather than treating every frame independently, it aggregates
    geometry, objects, lighting, and surfaces.
    """
    def __init__(self, scene_id: str = "default_scene"):
        self.scene_id = scene_id
        self.room_type: str = "unknown"
        self.camera_path: Dict[str, Any] = {"motion_type": "static", "direction": "none", "speed": "none"}
        self.camera_pose: Dict[str, Any] = {"pitch": "unknown", "yaw": "unknown", "roll": "unknown"}
        self.lighting: Dict[str, Any] = {"direction": "unknown", "color_temperature_k": 5000, "intensity": "soft"}
        self.surfaces: List[Dict[str, Any]] = []
        self.objects: List[Dict[str, Any]] = []
        self.empty_regions: List[Dict[str, Any]] = []
        self.placement_history: List[Dict[str, Any]] = []
        
    def update_state(self, frame_analysis: Dict[str, Any]) -> None:
        """
        Merges new VLM keyframe analysis metadata into the existing Scene Memory.
        """
        logger.info(f"Updating SceneMemory '{self.scene_id}' with new keyframe analysis.")
        
        # General Scene Properties
        if "room_type" in frame_analysis:
            self.room_type = frame_analysis["room_type"]
            
        if "camera_path" in frame_analysis:
            self.camera_path.update(frame_analysis["camera_path"])
            
        if "camera_pose" in frame_analysis:
            self.camera_pose.update(frame_analysis["camera_pose"])
            
        if "lighting" in frame_analysis:
            self.lighting.update(frame_analysis["lighting"])
            
        # Surfaces
        new_surfaces = frame_analysis.get("surfaces", [])
        for ns in new_surfaces:
            # Avoid duplicate surfaces by comparing surface_id
            existing = next((s for s in self.surfaces if s["surface_id"] == ns["surface_id"]), None)
            if existing:
                existing.update(ns)
            else:
                self.surfaces.append(ns)
                
        # Objects
        new_objects = frame_analysis.get("objects", [])
        for no in new_objects:
            existing = next((o for o in self.objects if o["object_id"] == no["object_id"]), None)
            if existing:
                existing.update(no)
            else:
                self.objects.append(no)
                
        # Empty Regions
        new_regions = frame_analysis.get("empty_regions", [])
        for nr in new_regions:
            existing = next((r for r in self.empty_regions if r["region_id"] == nr["region_id"]), None)
            if existing:
                existing.update(nr)
            else:
                self.empty_regions.append(nr)
                
        # Placement History
        if "placement_history" in frame_analysis:
            self.placement_history.extend(frame_analysis["placement_history"])

    def record_placement(self, frame_idx: int, product_id: str, bbox_2d: List[float]) -> None:
        """
        Logs a product placement execution into placement history.
        """
        self.placement_history.append({
            "frame_idx": frame_idx,
            "product_id": product_id,
            "bbox_2d": bbox_2d
        })

    def get_state(self) -> Dict[str, Any]:
        """
        Returns the entire state as a JSON-serializable dictionary.
        """
        return {
            "scene_id": self.scene_id,
            "room_type": self.room_type,
            "camera_path": self.camera_path,
            "camera_pose": self.camera_pose,
            "lighting": self.lighting,
            "surfaces": self.surfaces,
            "objects": self.objects,
            "empty_regions": self.empty_regions,
            "placement_history": self.placement_history
        }

    def save_to_file(self, file_path: str) -> None:
        """
        Saves the memory state to disk.
        """
        with open(file_path, "w") as f:
            json.dump(self.get_state(), f, indent=2)
        logger.info(f"Saved SceneMemory to {file_path}")

    def load_from_file(self, file_path: str) -> None:
        """
        Loads the memory state from disk.
        """
        with open(file_path, "r") as f:
            state = json.load(f)
        self.scene_id = state.get("scene_id", self.scene_id)
        self.room_type = state.get("room_type", "unknown")
        self.camera_path = state.get("camera_path", {})
        self.camera_pose = state.get("camera_pose", {})
        self.lighting = state.get("lighting", {})
        self.surfaces = state.get("surfaces", [])
        self.objects = state.get("objects", [])
        self.empty_regions = state.get("empty_regions", [])
        self.placement_history = state.get("placement_history", [])
        logger.info(f"Loaded SceneMemory from {file_path}")
