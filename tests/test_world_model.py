import os
import json
import pytest
from adframe.world_model.world_model import WorldModel
from adframe.world_model.memory_fusion import SceneMemoryFusion
from adframe.world_model.track_manager import TrackManager
from adframe.planner.placement_planner import PlacementPlanner

def test_track_manager_deduplication():
    tracker = TrackManager()
    
    # Frame 1: Sofa detected
    det_f1 = [
        {"label": "Sofa", "bbox": [0.2, 0.2, 0.4, 0.4], "depth": 2.5, "movable": False, "confidence": 0.95}
    ]
    tracked_f1 = tracker.track_objects(det_f1, frame_idx=0)
    assert len(tracked_f1) == 1
    sofa_id = tracked_f1[0]["object_id"]
    assert "sofa" in sofa_id
    
    # Frame 2: Sofa detected again with slight bbox overlap
    det_f2 = [
        {"label": "Sofa", "bbox": [0.22, 0.22, 0.42, 0.42], "depth": 2.6, "movable": False, "confidence": 0.96}
    ]
    tracked_f2 = tracker.track_objects(det_f2, frame_idx=1)
    assert len(tracked_f2) == 1
    assert tracked_f2[0]["object_id"] == sofa_id  # Stable ID check

def test_scene_memory_fusion():
    wm = WorldModel(scene_id="test_fusion")
    fusion = SceneMemoryFusion(wm)
    
    # Process Frame 0 Scene Graph
    sg1 = {
        "scene": {"id": "test", "type": "living_room", "confidence": 0.98},
        "camera": {"position": "center", "pitch": -10.0, "yaw": 0.0, "roll": 0.0, "fov_estimate": 60.0, "camera_height": 1.5, "confidence": 0.99},
        "lighting": {"type": "ambient", "direction": "top-right", "temperature": 4000.0, "intensity": 0.8, "ambient": 0.3, "confidence": 0.95},
        "surfaces": [
            {"surface_id": "desk", "label": "Wooden Desk", "bbox": [0.4, 0.1, 0.8, 0.9], "material": "wood", "orientation": "horizontal", "depth_estimate": 1.8, "reflection_strength": 0.1, "shadow_strength": 0.2, "usable_area": 0.9, "confidence": 0.95}
        ],
        "objects": [
            {"id": "laptop", "label": "Laptop", "bbox": [0.5, 0.3, 0.7, 0.6], "depth": 1.7, "occluder": False, "movable": True, "brand_safe": True, "confidence": 0.98}
        ],
        "empty_regions": [
            {"region_id": "r1", "surface_id": "desk", "bbox": [0.5, 0.1, 0.6, 0.3], "available_area": 0.8, "visibility_score": 0.9, "occlusion_probability": 0.05, "distance_to_camera": 1.8, "confidence": 0.95}
        ],
        "occlusions": [],
        "placement_candidates": [
            {"candidate_id": "r1", "surface": "desk", "bbox": [0.5, 0.1, 0.6, 0.3], "score": 0.9, "reason": "visible", "recommended_product_size": "medium", "camera_visibility": 0.9, "risk": 0.05, "confidence": 0.95}
        ]
    }
    
    fusion.fuse_scene_graph(sg1, frame_idx=0)
    assert wm.duration_frames == 1
    assert len(wm.objects) == 1
    assert len(wm.surfaces) == 1
    assert len(wm.placement_regions) == 1
    
    # Process Frame 1 Scene Graph (stable surfaces/objects)
    sg2 = sg1.copy()
    fusion.fuse_scene_graph(sg2, frame_idx=1)
    
    assert wm.duration_frames == 2
    assert len(wm.objects) == 1  # Deduplicated
    assert len(wm.surfaces) == 1  # Deduplicated
    assert wm.placement_regions[0]["stability_score"] == 1.0  # Temporally stable region!

def test_planner_consumes_world_model():
    wm = WorldModel(scene_id="test_plan")
    
    # Populate a compliant fused WorldModel mock structure
    wm.surfaces = [
        {
            "surface_id": "surface_desk",
            "label": "wooden desk",
            "polygon": [],
            "bbox": [0.4, 0.1, 0.8, 0.9],
            "material": "wood",
            "orientation": "horizontal",
            "depth_estimate": 1.8,
            "surface_normal": [0.0, 1.0, 0.0],
            "reflection_strength": 0.18,
            "shadow_strength": 0.25,
            "usable_area": 0.9,
            "first_seen_frame": 0,
            "last_seen_frame": 10,
            "confidence": 0.95
        }
    ]
    wm.placement_regions = [
        {
            "region_id": "region_desk_center",
            "surface_id": "surface_desk",
            "polygon": [],
            "bbox": [0.5, 0.2, 0.7, 0.5],
            "available_area": 0.85,
            "visibility_history": [1.0, 1.0],
            "occlusion_probability": 0.05,
            "distance_to_camera": 1.8,
            "camera_angle_history": [0.0],
            "stability_score": 1.0,
            "average_ranking": 1.0,
            "recommended_product_size": "medium",
            "confidence": 0.95
        }
    ]
    
    planner = PlacementPlanner()
    plan = planner.plan_placement(scene_graph=wm.to_dict(), product_metadata={"name": "perfume"})
    
    assert plan["target_surface"] == "surface_desk"
    assert plan["placement_candidate"] == "region_desk_center"
    assert plan["rendering_constraints"]["reflection"] == 0.18
