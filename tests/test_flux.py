import os
import json
import pytest
from PIL import Image
from adframe.generation.flux_generator import FluxGenerator

def test_mock_flux_generation(tmp_path):
    # Create a simple white background image for testing
    img_path = os.path.join(tmp_path, "test_frame.png")
    img = Image.new("RGB", (200, 200), color="white")
    img.save(img_path)
    
    # Simple planner config
    planner_data = {
        "prompt": "a green bottle",
        "placement": {
            "bbox_2d": [0.2, 0.2, 0.8, 0.8]
        }
    }
    
    output_path = os.path.join(tmp_path, "rendered_frame.png")
    
    gen = FluxGenerator(backend="mock")
    metrics = gen.generate(img_path, planner_data, output_path)
    
    assert os.path.exists(output_path)
    assert os.path.exists(output_path.replace(".png", "_mask.png"))
    assert os.path.exists(metrics["metadata_path"])
    
    # Load metadata and check keys
    with open(metrics["metadata_path"], "r") as f:
        meta = json.load(f)
    assert meta["prompt"] == "a green bottle"
    assert meta["backend"] == "mock"
