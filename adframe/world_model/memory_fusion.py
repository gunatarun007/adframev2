import logging
from typing import Dict, Any, List
from adframe.world_model.world_model import WorldModel
from adframe.world_model.track_manager import TrackManager

logger = logging.getLogger("adframe.world_model.memory_fusion")

class SceneMemoryFusion:
    """
    Temporal fusion engine that processes incoming scene graphs frame-by-frame,
    maintains object tracks with stable IDs, tracks camera and lighting timelines,
    and updates the persistent WorldModel.
    """
    def __init__(self, world_model: WorldModel):
        self.world_model = world_model
        self.tracker = TrackManager()
        self.processed_frames: List[int] = []

    def fuse_scene_graph(self, sg: Dict[str, Any], frame_idx: int):
        """
        Fuses a single frame's Scene Graph into the persistent WorldModel.
        """
        logger.info(f"Fusing Scene Graph for frame {frame_idx}")
        self.processed_frames.append(frame_idx)
        self.world_model.duration_frames = max(self.world_model.duration_frames, frame_idx + 1)
        
        # 1. Update Camera timeline
        cam_info = sg.get("camera", {})
        self.world_model.camera_timeline.append({
            "frame": frame_idx,
            "position": cam_info.get("position", "center"),
            "pitch": float(cam_info.get("pitch", 0.0)),
            "yaw": float(cam_info.get("yaw", 0.0)),
            "roll": float(cam_info.get("roll", 0.0))
        })
        
        # 2. Update Lighting timeline
        light_info = sg.get("lighting", {})
        self.world_model.lighting_timeline.append({
            "frame": frame_idx,
            "type": light_info.get("type", "ambient"),
            "direction": light_info.get("direction", "top-right"),
            "temperature": float(light_info.get("temperature", 4000.0)),
            "intensity": float(light_info.get("intensity", 0.8))
        })
        
        # 3. Track surfaces and objects using TrackManager
        raw_surfaces = sg.get("surfaces", [])
        tracked_surfaces = self.tracker.track_surfaces(raw_surfaces, frame_idx)
        
        raw_objects = sg.get("objects", [])
        tracked_objects = self.tracker.track_objects(raw_objects, frame_idx)
        
        raw_regions = sg.get("empty_regions", sg.get("placement_candidates", []))
        tracked_regions = self.tracker.track_regions(raw_regions, frame_idx)
        
        # Update object timeline
        for obj in tracked_objects:
            self.world_model.object_timeline.append({
                "frame": frame_idx,
                "object_id": obj["object_id"],
                "bbox": obj["bbox"]
            })

        # 4. Map tracked lists back to WorldModel persistent states
        # Convert tracks to list format compliant with schema requirements
        self.world_model.surfaces = []
        for track in self.tracker.surface_tracks.values():
            self.world_model.surfaces.append({
                "surface_id": track["surface_id"],
                "label": track["label"],
                "polygon": track["polygon"],
                "bbox": track["bbox"],
                "material": track["material"],
                "orientation": track["orientation"],
                "depth_estimate": track["depth_estimate"],
                "surface_normal": track["surface_normal"],
                "reflection_strength": track["reflection_strength"],
                "shadow_strength": track["shadow_strength"],
                "usable_area": track["usable_area"],
                "first_seen_frame": track["first_seen_frame"],
                "last_seen_frame": track["last_seen_frame"],
                "confidence": track["confidence"]
            })
            
        self.world_model.objects = []
        for track in self.tracker.object_tracks.values():
            # Trajectory is bbox history centers
            trajectory = []
            for bbox in track["bbox_history"]:
                ymin, xmin, ymax, xmax = bbox
                cy = (ymin + ymax) / 2.0
                cx = (xmin + xmax) / 2.0
                trajectory.append([cx, cy])
                
            self.world_model.objects.append({
                "object_id": track["object_id"],
                "class_label": track["class_label"],
                "trajectory": trajectory,
                "bbox_history": track["bbox_history"],
                "polygon_history": track["polygon_history"],
                "visibility_history": track["visibility_history"],
                "depth_history": track["depth_history"],
                "occlusion_history": track["occlusion_history"],
                "motion_state": track["motion_state"],
                "confidence": track["confidence"]
            })
            
        self.world_model.placement_regions = []
        for track in self.tracker.region_tracks.values():
            self.world_model.placement_regions.append({
                "region_id": track["region_id"],
                "surface_id": track["surface_id"],
                "polygon": track["polygon"],
                "bbox": track["bbox"],
                "available_area": track["available_area"],
                "visibility_history": track["visibility_history"],
                "occlusion_probability": track["occlusion_probability"],
                "distance_to_camera": track["distance_to_camera"],
                "camera_angle_history": track["camera_angle_history"],
                "stability_score": track["stability_score"],
                "average_ranking": track["average_ranking"],
                "recommended_product_size": track["recommended_product_size"],
                "confidence": track["confidence"]
            })

        # 5. Compute Statistics averages and stability trends
        self._recompute_statistics()

    def _recompute_statistics(self):
        """
        Recomputes average lighting and camera movements over the compiled timelines.
        """
        w_stats = self.world_model.statistics
        w_conf = self.world_model.confidence
        
        # average lighting temperature
        temps = [item["temperature"] for item in self.world_model.lighting_timeline]
        avg_temp = sum(temps) / len(temps) if temps else 4000.0
        w_stats["average_lighting_temperature"] = float(avg_temp)
        
        # lighting stability (based on temperature variance)
        if len(temps) > 1:
            mean = sum(temps) / len(temps)
            variance = sum((t - mean) ** 2 for t in temps) / len(temps)
            std_dev = variance ** 0.5
            w_stats["lighting_stability"] = max(0.0, min(1.0, 1.0 - (std_dev / 1000.0)))
        else:
            w_stats["lighting_stability"] = 0.95
            
        # camera stability & movement
        pitches = [item["pitch"] for item in self.world_model.camera_timeline]
        yaws = [item["yaw"] for item in self.world_model.camera_timeline]
        
        if len(pitches) > 1:
            delta_pitch = abs(pitches[-1] - pitches[0])
            delta_yaw = abs(yaws[-1] - yaws[0])
            total_motion = delta_pitch + delta_yaw
            
            w_stats["camera_movement"] = "pan/tilt" if total_motion > 5.0 else "static"
            w_stats["camera_velocity"] = float(total_motion / len(self.world_model.camera_timeline))
            w_stats["camera_stability"] = max(0.0, min(1.0, 1.0 - (total_motion / 50.0)))
        else:
            w_stats["camera_movement"] = "static"
            w_stats["camera_velocity"] = 0.0
            w_stats["camera_stability"] = 0.98

        # 6. Update overall scene confidence as mean track confidences
        conf_scores = [t["confidence"] for t in self.tracker.object_tracks.values()] + \
                      [t["confidence"] for t in self.tracker.surface_tracks.values()]
        avg_conf = sum(conf_scores) / len(conf_scores) if conf_scores else 0.90
        w_conf["overall_scene_confidence"] = float(avg_conf)
        
        # Update latest camera state to reflect timeline averages
        if self.world_model.camera_timeline:
            last_cam = self.world_model.camera_timeline[-1]
            self.world_model.camera_state["pitch"] = last_cam["pitch"]
            self.world_model.camera_state["yaw"] = last_cam["yaw"]
            self.world_model.camera_state["roll"] = last_cam["roll"]
            self.world_model.camera_state["position"] = last_cam["position"]
            
        if self.world_model.lighting_timeline:
            last_light = self.world_model.lighting_timeline[-1]
            self.world_model.lighting_state["type"] = last_light["type"]
            self.world_model.lighting_state["direction"] = last_light["direction"]
            self.world_model.lighting_state["temperature"] = last_light["temperature"]
            self.world_model.lighting_state["intensity"] = last_light["intensity"]
