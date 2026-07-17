import logging
from typing import List, Dict, Any

logger = logging.getLogger("adframe.world_model.track_manager")

def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """
    Computes Intersection over Union (IoU) of two normalized boxes [ymin, xmin, ymax, xmax].
    """
    if not boxA or not boxB or len(boxA) != 4 or len(boxB) != 4:
        return 0.0
    yminA, xminA, ymaxA, xmaxA = boxA
    yminB, xminB, ymaxB, xmaxB = boxB

    # Determine intersection coordinates
    yminI = max(yminA, yminB)
    xminI = max(xminA, xminB)
    ymaxI = min(ymaxA, ymaxB)
    xmaxI = min(xmaxA, xmaxB)

    # Compute intersection area
    inter_w = max(0.0, xmaxI - xminI)
    inter_h = max(0.0, ymaxI - yminI)
    inter_area = inter_w * inter_h

    # Compute union area
    areaA = (xmaxA - xminA) * (ymaxA - yminA)
    areaB = (xmaxB - xminB) * (ymaxB - yminB)
    union_area = areaA + areaB - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area

class TrackManager:
    """
    Manages temporal correspondence and maintains persistent IDs for objects and surfaces
    across multiple keyframe observations.
    """
    def __init__(self):
        self.object_tracks: Dict[str, Dict[str, Any]] = {}
        self.surface_tracks: Dict[str, Dict[str, Any]] = {}
        self.region_tracks: Dict[str, Dict[str, Any]] = {}
        
        self.object_count = 0
        self.surface_count = 0
        self.region_count = 0

    def track_objects(self, detections: List[Dict[str, Any]], frame_idx: int) -> List[Dict[str, Any]]:
        """
        Matches frame-level object detections to persistent tracks.
        """
        tracked_results = []
        matched_track_ids = set()

        for det in detections:
            det_label = det.get("label", "unknown").lower()
            det_bbox = det.get("bbox", det.get("bbox_2d", [0.4, 0.4, 0.6, 0.6]))
            
            # Find best matching track based on class alignment and bounding box IoU
            best_track_id = None
            best_iou = -1.0

            for track_id, track in self.object_tracks.items():
                if track_id in matched_track_ids:
                    continue
                # Labels must match or overlap
                if track["class_label"] == det_label:
                    # Compute IoU with last seen bbox
                    last_bbox = track["bbox_history"][-1]
                    iou = compute_iou(det_bbox, last_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_track_id = track_id

            # IoU match threshold: 0.15 for relaxed tracking under camera motion
            if best_track_id is not None and best_iou >= 0.15:
                # Update existing track
                track = self.object_tracks[best_track_id]
                track["bbox_history"].append(det_bbox)
                track["polygon_history"].append(det.get("polygon", []))
                track["visibility_history"].append(1.0)
                track["depth_history"].append(float(det.get("depth", 0.5)))
                track["occlusion_history"].append(float(det.get("occluded_by", 0) if isinstance(det.get("occluded_by"), (int, float)) else 0.0))
                track["last_seen_frame"] = frame_idx
                matched_track_ids.add(best_track_id)
            else:
                # Create a new track
                self.object_count += 1
                best_track_id = f"object_{det_label}_{self.object_count}"
                self.object_tracks[best_track_id] = {
                    "object_id": best_track_id,
                    "class_label": det_label,
                    "bbox_history": [det_bbox],
                    "polygon_history": [det.get("polygon", [])],
                    "visibility_history": [1.0],
                    "depth_history": [float(det.get("depth", 0.5))],
                    "occlusion_history": [0.0],
                    "motion_state": "static" if not det.get("movable", True) else "dynamic",
                    "first_seen_frame": frame_idx,
                    "last_seen_frame": frame_idx,
                    "confidence": float(det.get("confidence", 0.90))
                }
                matched_track_ids.add(best_track_id)

            # Build result element for this frame
            res_obj = self.object_tracks[best_track_id].copy()
            # Include frame local bbox
            res_obj["bbox"] = det_bbox
            res_obj["polygon"] = det.get("polygon", [])
            tracked_results.append(res_obj)

        # Backfill non-detected active tracks with visibility=0.0
        for track_id, track in self.object_tracks.items():
            if track_id not in matched_track_ids and track["last_seen_frame"] >= frame_idx - 5: # keep alive buffer
                track["bbox_history"].append(track["bbox_history"][-1])
                track["visibility_history"].append(0.0)

        return tracked_results

    def track_surfaces(self, detections: List[Dict[str, Any]], frame_idx: int) -> List[Dict[str, Any]]:
        """
        Matches frame-level surface detections to persistent tracks.
        """
        tracked_results = []
        matched_track_ids = set()

        for det in detections:
            det_label = det.get("label", det.get("surface_id", "surface")).lower()
            det_bbox = det.get("bbox", det.get("bbox_2d", [0.0, 0.0, 1.0, 1.0]))
            
            best_track_id = None
            best_iou = -1.0

            for track_id, track in self.surface_tracks.items():
                if track_id in matched_track_ids:
                    continue
                if track["label"] == det_label or track["surface_id"] == det.get("surface_id", "").lower():
                    last_bbox = track["bbox"]
                    iou = compute_iou(det_bbox, last_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_track_id = track_id

            if best_track_id is not None and best_iou >= 0.15:
                # Update existing track
                track = self.surface_tracks[best_track_id]
                track["bbox"] = det_bbox
                track["polygon"] = det.get("polygon", [])
                track["polygon_history"].append(det.get("polygon", []))
                track["last_seen_frame"] = frame_idx
                matched_track_ids.add(best_track_id)
            else:
                self.surface_count += 1
                best_track_id = f"surface_{det_label}_{self.surface_count}"
                self.surface_tracks[best_track_id] = {
                    "surface_id": best_track_id,
                    "label": det_label,
                    "polygon": det.get("polygon", []),
                    "polygon_history": [det.get("polygon", [])],
                    "bbox": det_bbox,
                    "material": det.get("material", "wood").lower(),
                    "orientation": det.get("orientation", "horizontal").lower(),
                    "depth_estimate": float(det.get("depth_estimate", 2.0)),
                    "surface_normal": det.get("surface_normal", [0.0, 1.0, 0.0]),
                    "reflection_strength": float(det.get("reflection_strength", 0.15)),
                    "shadow_strength": float(det.get("shadow_strength", 0.2)),
                    "usable_area": float(det.get("usable_area", 0.8)),
                    "first_seen_frame": frame_idx,
                    "last_seen_frame": frame_idx,
                    "confidence": float(det.get("confidence", 0.90))
                }
                matched_track_ids.add(best_track_id)

            tracked_results.append(self.surface_tracks[best_track_id])

        return tracked_results

    def track_regions(self, detections: List[Dict[str, Any]], frame_idx: int) -> List[Dict[str, Any]]:
        """
        Matches frame-level empty placement regions to persistent tracks.
        """
        tracked_results = []
        matched_track_ids = set()

        for det in detections:
            det_bbox = det.get("bbox", det.get("bbox_2d", [0.4, 0.4, 0.6, 0.6]))
            det_surf = det.get("surface_id", "unknown")
            
            best_track_id = None
            best_iou = -1.0

            for track_id, track in self.region_tracks.items():
                if track_id in matched_track_ids:
                    continue
                # Map to same target surface
                if track["surface_id"] == det_surf:
                    last_bbox = track["bbox"]
                    iou = compute_iou(det_bbox, last_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_track_id = track_id

            if best_track_id is not None and best_iou >= 0.15:
                track = self.region_tracks[best_track_id]
                track["bbox"] = det_bbox
                track["polygon"] = det.get("polygon", [])
                track["visibility_history"].append(1.0)
                track["camera_angle_history"].append(float(det.get("camera_angle", 0.0)))
                track["last_seen_frame"] = frame_idx
                matched_track_ids.add(best_track_id)
            else:
                self.region_count += 1
                best_track_id = f"region_{self.region_count}"
                self.region_tracks[best_track_id] = {
                    "region_id": best_track_id,
                    "surface_id": det_surf,
                    "polygon": det.get("polygon", []),
                    "bbox": det_bbox,
                    "available_area": float(det.get("available_area", 0.8)),
                    "visibility_history": [1.0],
                    "occlusion_probability": float(det.get("occlusion_probability", 0.1)),
                    "distance_to_camera": float(det.get("distance_to_camera", 2.0)),
                    "camera_angle_history": [0.0],
                    "stability_score": 1.0,
                    "average_ranking": 1.0,
                    "recommended_product_size": det.get("recommended_product_size", "medium"),
                    "first_seen_frame": frame_idx,
                    "last_seen_frame": frame_idx,
                    "confidence": float(det.get("confidence", 0.90))
                }
                matched_track_ids.add(best_track_id)

            tracked_results.append(self.region_tracks[best_track_id])

        # Update stability metrics for active region tracks
        for track_id, track in self.region_tracks.items():
            if track_id not in matched_track_ids and track["last_seen_frame"] >= frame_idx - 5:
                track["visibility_history"].append(0.0)
            
            # stability_score = ratio of frames seen to total track duration
            total_duration = max(1, frame_idx - track["first_seen_frame"] + 1)
            frames_seen = sum(1 for v in track["visibility_history"] if v > 0.5)
            track["stability_score"] = float(frames_seen) / float(total_duration)

        return tracked_results
