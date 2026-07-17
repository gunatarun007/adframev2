import os
import shutil
import pytest
from PIL import Image

from adframe.config import config
from adframe.vision.vision_model import VisionModel
from adframe.scene_memory.memory_store import SceneMemory
from adframe.planner.placement_planner import PlacementPlanner
from adframe.generation.flux_generator import FluxGenerator
from adframe.judge.vision_judge import VisionJudge
from adframe.correction.correction_manager import CorrectionManager
from adframe.renderer.blend_renderer import BlendRenderer
from adframe.pipeline.orchestration_pipeline import OrchestrationPipeline

@pytest.fixture(scope="module", autouse=True)
def setup_test_directories():
    """Ensure clean test environment directories."""
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.cache_dir, exist_ok=True)
    yield
    # Clean up test directories after run if desired
    if os.path.exists(config.cache_dir):
        shutil.rmtree(config.cache_dir)

def test_vision_model_mock_json():
    vlm = VisionModel(use_mock=True)
    schema = {"title": "SceneMemory", "type": "OBJECT"}
    result = vlm.query_json("Analyze scene", expected_schema=schema)
    assert isinstance(result, dict)
    assert result["room_type"] == "living_room"
    assert "surfaces" in result
    assert len(result["surfaces"]) > 0

def test_scene_memory_accumulation():
    memory = SceneMemory(scene_id="test_accumulation")
    
    frame_1 = {
        "room_type": "kitchen",
        "lighting": {"direction": "left", "color_temperature_k": 3000},
        "surfaces": [
            {
                "surface_id": "surf_counter",
                "label": "marble counter",
                "bbox_2d": [0.1, 0.5, 0.9, 0.9],
                "material": "marble",
                "orientation": "horizontal"
            }
        ],
        "objects": [
            {"object_id": "obj_microwave", "label": "microwave oven", "bbox_2d": [0.2, 0.2, 0.4, 0.4], "depth_order": 1}
        ],
        "empty_regions": [
            {"region_id": "reg_counter_left", "bbox_2d": [0.2, 0.5, 0.4, 0.7], "surface_id": "surf_counter", "dimensions_px": [100, 100]}
        ]
    }
    
    memory.update_state(frame_1)
    
    # Assert fields parsed
    assert memory.room_type == "kitchen"
    assert memory.lighting["color_temperature_k"] == 3000
    assert len(memory.surfaces) == 1
    assert len(memory.objects) == 1
    
    # Add frame 2 with updated coordinates / new objects
    frame_2 = {
        "surfaces": [
            {
                "surface_id": "surf_counter",
                "label": "marble counter",
                "bbox_2d": [0.1, 0.5, 0.95, 0.9], # extended
                "material": "marble",
                "orientation": "horizontal"
            }
        ],
        "objects": [
            {"object_id": "obj_toaster", "label": "electric toaster", "bbox_2d": [0.5, 0.5, 0.6, 0.6], "depth_order": 1}
        ]
    }
    
    memory.update_state(frame_2)
    
    # Assert surface merged/updated, new object added
    assert len(memory.surfaces) == 1
    assert memory.surfaces[0]["bbox_2d"][2] == 0.95  # updated coordinate
    assert len(memory.objects) == 2  # obj_microwave + obj_toaster

def test_placement_planner_fallback():
    planner = PlacementPlanner(vision_model=None) # Forces fallback
    scene_state = {
        "surfaces": [
            {"surface_id": "s1", "label": "desk", "material": "wood", "orientation": "horizontal"}
        ],
        "empty_regions": [
            {"region_id": "r1", "bbox_2d": [0.1, 0.1, 0.3, 0.3], "surface_id": "s1"}
        ]
    }
    product = {"product_id": "p_soda", "name": "soda can"}
    
    plan = planner.plan_placement(scene_state, product)
    assert plan["placement"]["target_surface_id"] == "s1"
    assert "soda can" in plan["prompt"]
    assert "wood" in plan["prompt"]

def test_flux_generator_mock():
    gen = FluxGenerator(use_mock=True)
    
    # Create temp images
    img_path = os.path.join(config.cache_dir, "temp_img.png")
    mask_path = os.path.join(config.cache_dir, "temp_mask.png")
    
    Image.new("RGB", (200, 200), (255, 0, 0)).save(img_path)
    Image.new("L", (200, 200), 0).save(mask_path)
    
    output = gen.generate_fill(
        image_path=img_path,
        mask_path=mask_path,
        prompt="a test product"
    )
    
    assert output.size == (200, 200)

def test_blend_renderer_mock():
    renderer = BlendRenderer(use_mock=True)
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    bbox = [0.1, 0.1, 0.5, 0.5]
    
    mask = renderer.generate_binary_mask(img, bbox)
    assert mask.size == (100, 100)
    
    # Verify pixels are active inside bbox
    mask_data = mask.load()
    assert mask_data[30, 30] == 255
    assert mask_data[80, 80] == 0

def test_correction_manager():
    manager = CorrectionManager(max_retries=2)
    frame_id = "f_01"
    
    # Under threshold, retry budget left
    assert manager.should_retry(frame_id, score=0.6, threshold=0.8) is True
    
    original_plan = {
        "placement": {"bbox_2d": [0.2, 0.2, 0.4, 0.4], "target_surface_id": "desk"},
        "prompt": "product on desk"
    }
    
    feedback = {
        "corrections": {
            "adjust_prompt": "product on desk, shadow bottom",
            "adjust_mask": {
                "shift_px": [100, 200],  # shift [x, y]
                "scale_factor": 1.5
            }
        }
    }
    
    retry_plan = manager.formulate_retry(original_plan, feedback, frame_id)
    
    assert retry_plan["prompt"] == "product on desk, shadow bottom"
    # Verify scale and coordinate shift logic was applied
    new_bbox = retry_plan["placement"]["bbox_2d"]
    assert new_bbox != [0.2, 0.2, 0.4, 0.4]
    
    # Attempt 2
    assert manager.should_retry(frame_id, score=0.7, threshold=0.8) is True
    retry_plan_2 = manager.formulate_retry(retry_plan, feedback, frame_id)
    
    # Attempt 3 - budget exhausted
    assert manager.should_retry(frame_id, score=0.7, threshold=0.8) is False

def test_orchestration_pipeline_end_to_end_mock():
    # End-to-end orchestration with mock dependencies
    pipeline = OrchestrationPipeline(use_mock=True)
    
    product_metadata = {
        "product_id": "test_perfume_bottle",
        "name": "luxury perfume bottle",
        "dimensions": "15x5x5 cm",
        "brand_constraints": "do not place near food"
    }
    
    # Run pipeline on a mock video path
    # Since OpenCV is mocked, it will create visual room templates and output the blended results
    output_video = pipeline.run_pipeline("mock_video_input.mp4", product_metadata)
    
    assert os.path.exists(output_video + ".png") or os.path.exists(output_video)
    
    # Verify intermediate files were generated in output folder
    final_mem_path = os.path.join(config.output_dir, "final_scene_memory.json")
    assert os.path.exists(final_mem_path)
    mem = json_load(final_mem_path)
    assert len(mem["placement_history"]) == 3 # 3 mock keyframes processed

def test_orchestration_pipeline_retry_loop_trigger():
    # Force a mock retry trigger using prompt string mapping
    pipeline = OrchestrationPipeline(use_mock=True)
    
    product_metadata = {
        "product_id": "test_perfume_bottle",
        "name": "retry_trigger perfume bottle",  # triggers mock judge rejection score=0.65
        "dimensions": "15x5x5 cm"
    }
    
    # Execute pipeline; frame_000 will retry up to config max_retries (3 attempts)
    output_video = pipeline.run_pipeline("mock_video_input.mp4", product_metadata)
    
    # Read status logs for frame_000 to verify 3 attempts were recorded
    status_path = os.path.join(config.output_dir, "frame_000_status.json")
    assert os.path.exists(status_path)
    status = json_load(status_path)
    assert status["attempt"] == config.max_retries + 1  # Initial attempt + max_retries attempts

def json_load(path: str):
    import json
    with open(path, "r") as f:
        return json.load(f)
