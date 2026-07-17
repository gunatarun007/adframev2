import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any

class VisionBackend(str, Enum):
    MOCK = "mock"
    QWEN = "qwen"

@dataclass
class PipelineConfig:
    # Model configuration
    vlm_model_id: str = os.getenv("ADFRAME_VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    flux_model_id: str = os.getenv("ADFRAME_FLUX_MODEL", "black-forest-labs/FLUX.1-Fill-dev")
    sam_model_id: str = os.getenv("ADFRAME_SAM_MODEL", "facebook/sam2-hiera-large")
    
    # Selected backend for vision model
    vision_backend: str = os.getenv("ADFRAME_VISION_BACKEND", "mock")

    
    # Device configuration
    vlm_device: str = os.getenv("ADFRAME_VLM_DEVICE", "cuda")
    flux_device: str = os.getenv("ADFRAME_FLUX_DEVICE", "cuda")
    sam_device: str = os.getenv("ADFRAME_SAM_DEVICE", "cuda")
    
    # Thresholds
    judge_threshold: float = float(os.getenv("ADFRAME_JUDGE_THRESHOLD", "0.80"))
    max_retries: int = int(os.getenv("ADFRAME_MAX_RETRIES", "3"))
    
    # Directories
    output_dir: str = os.getenv("ADFRAME_OUTPUT_DIR", "./outputs")
    cache_dir: str = os.getenv("ADFRAME_CACHE_DIR", "./.cache")
    
    # Keyframe Selection
    optical_flow_threshold: float = 0.5  # novelty score to trigger new keyframe
    max_frames_to_skip: int = 15          # max spacing between keyframes
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

# Default global config
config = PipelineConfig()
