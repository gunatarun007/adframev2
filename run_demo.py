import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from adframe.pipeline.orchestration_pipeline import OrchestrationPipeline
from adframe.config import config

product_metadata = {
    "product_id": "luxury_watch_model_x",
    "name": "Luxury Chronograph Gold Watch",
    "dimensions": "40mm dial",
    "brand_constraints": "do not place near alcohol or rubbish"
}

print("Starting end-to-end AdFrame v2 demo on RunPod...")
print(f"Input video: /workspace/demo.mp4")
print(f"Global configuration: VLM={config.vlm_model_id}, Device={config.vlm_device}, OutputDir={config.output_dir}")

pipeline = OrchestrationPipeline(use_mock=True)
output_path = pipeline.run_pipeline("/workspace/demo.mp4", product_metadata)

print("--------------------------------------------------")
print("DEMO EXECUTION COMPLETE!")
print(f"Output video compiled to: {output_path}")
print("Final scene memory state saved to outputs/final_scene_memory.json")
print("--------------------------------------------------")
