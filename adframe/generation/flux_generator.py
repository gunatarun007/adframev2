import os
import json
import time
import logging
from typing import Dict, Any, Optional
import torch
import numpy as np
import cv2
from PIL import Image

from adframe.config import config

logger = logging.getLogger("adframe.generation.flux_generator")

class FluxGenerator:
    """
    Manages loading and running the FLUX.1 Fill image generator.
    Supports ModelScope, Hugging Face, and Local Directory registries.
    """
    def __init__(self, backend: Optional[str] = None, provider: Optional[str] = None, model_id: Optional[str] = None, device: Optional[str] = None, use_mock: bool = False):
        self.backend = backend or ("mock" if use_mock else "flux")
        self.provider = provider or config.flux_provider
        self.pipeline = None
        self.model_load_time_sec = 0.0

    def load_model(self):
        """
        Loads the FLUX.1 Fill pipeline into memory based on the provider.
        """
        if self.backend == "mock":
            logger.info("Initializing mock FLUX generator pipeline.")
            return
            
        if self.pipeline is not None:
            return
            
        t_start = time.time()
        
        # 1. Resolve path based on provider abstraction
        if self.provider == "modelscope":
            logger.info(f"Using ModelScope provider to fetch: {config.flux_modelscope_id}")
            try:
                from modelscope import snapshot_download
                # Resolve local folder
                model_dir = snapshot_download(config.flux_modelscope_id, cache_dir=config.cache_dir)
                logger.info(f"ModelScope model snapshot resolved at: {model_dir}")
                model_load_path = model_dir
            except ImportError:
                logger.warning("modelscope Python SDK is not installed. Trying fallback to Hugging Face model ID.")
                model_load_path = config.flux_model_id
        elif self.provider == "local":
            logger.info(f"Using local path: {config.flux_model_id}")
            model_load_path = config.flux_model_id
        else:
            logger.info(f"Using Hugging Face provider to fetch: {config.flux_model_id}")
            model_load_path = config.flux_model_id

        # 2. Initialize diffusers pipeline
        from diffusers import FluxFillPipeline
        torch_dtype = torch.bfloat16 if config.flux_dtype == "bfloat16" else torch.float32
        
        logger.info(f"Initializing FluxFillPipeline from: {model_load_path}")
        self.pipeline = FluxFillPipeline.from_pretrained(
            model_load_path,
            torch_dtype=torch_dtype,
            cache_dir=config.cache_dir
        )
        
        # 3. Memory offloading configuration
        if config.use_cpu_offload:
            logger.info("Enabling CPU offload for FluxFillPipeline VRAM efficiency.")
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline.to(config.flux_device)
            
        t_end = time.time()
        self.model_load_time_sec = t_end - t_start
        logger.info(f"FLUX Fill model loaded successfully in {self.model_load_time_sec:.2f} seconds.")

    def generate(self, image_path: str, planner_data: Dict[str, Any], output_path: str) -> Dict[str, Any]:
        """
        Runs inpainting generation using FLUX Fill inside the masked bbox region.
        """
        t_start_inf = time.time()
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input frame image not found: {image_path}")
            
        # Load image
        pil_image = Image.open(image_path).convert("RGB")
        width, height = pil_image.size
        
        # Extract bbox [ymin, xmin, ymax, xmax]
        placement = planner_data.get("placement", {})
        bbox = placement.get("bbox_2d", [0.4, 0.4, 0.6, 0.6])
        
        ymin, xmin, ymax, xmax = bbox
        left = int(xmin * width)
        top = int(ymin * height)
        right = int(xmax * width)
        bottom = int(ymax * height)
        
        left = max(0, min(width - 1, left))
        top = max(0, min(height - 1, top))
        right = max(left + 1, min(width, right))
        bottom = max(top + 1, min(height, bottom))
        
        # Build binary mask
        mask_np = np.zeros((height, width), dtype=np.uint8)
        mask_np[top:bottom, left:right] = 255
        pil_mask = Image.fromarray(mask_np)
        
        mask_output_path = output_path.replace(".png", "_mask.png")
        pil_mask.save(mask_output_path)
        logger.info(f"Saved binary mask to {mask_output_path}")
        
        prompt = planner_data.get("prompt", "a product")
        
        if self.backend == "mock":
            logger.info("Executing mock image generation...")
            img_np = np.array(pil_image)
            cv2.rectangle(img_np, (left, top), (right, bottom), (150, 0, 200), -1)
            cv2.putText(img_np, "MOCK PRODUCT", (left + 5, top + int((bottom - top) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            result_img = Image.fromarray(img_np)
            result_img.save(output_path)
            time.sleep(1.0)
        else:
            self.load_model()
            logger.info("Invoking real FLUX Fill inference...")
            
            inf_w = ((width + 7) // 8) * 8
            inf_h = ((height + 7) // 8) * 8
            
            resized_image = pil_image.resize((inf_w, inf_h), Image.Resampling.LANCZOS)
            resized_mask = pil_mask.resize((inf_w, inf_h), Image.Resampling.NEAREST)
            
            generation_result = self.pipeline(
                prompt=prompt,
                image=resized_image,
                mask_image=resized_mask,
                height=inf_h,
                width=inf_w,
                guidance_scale=30.0,
                num_inference_steps=28
            ).images[0]
            
            result_img = generation_result.resize((width, height), Image.Resampling.LANCZOS)
            result_img.save(output_path)
            
        t_end_inf = time.time()
        inf_duration = t_end_inf - t_start_inf
        logger.info(f"FLUX Fill generation completed in {inf_duration:.2f} seconds.")
        
        metadata = {
            "image_size": f"{width}x{height}",
            "prompt": prompt,
            "bbox_coords": [ymin, xmin, ymax, xmax],
            "backend": self.backend,
            "provider": self.provider,
            "inference_steps": 28
        }
        
        metadata_path = output_path.replace(".png", "_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
        return {
            "inference_time_sec": inf_duration,
            "model_load_time_sec": self.model_load_time_sec,
            "metadata_path": metadata_path
        }

    def generate_fill(self, image_path: str, mask_path: str, prompt: str, negative_prompt: Optional[str] = None) -> Image.Image:
        """
        Legacy wrapper for generate_fill used by OrchestrationPipeline.
        """
        planner_data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "placement": {
                "bbox_2d": self._extract_bbox_from_mask(mask_path)
            }
        }
        
        temp_out = os.path.join(config.cache_dir, "temp_gen_out.png")
        self.generate(image_path, planner_data, temp_out)
        
        return Image.open(temp_out).convert("RGB")

    def _extract_bbox_from_mask(self, mask_path: str) -> list:
        try:
            mask = Image.open(mask_path).convert("L")
            mask_np = np.array(mask)
            non_zero = np.argwhere(mask_np > 128)
            if non_zero.size == 0:
                return [0.4, 0.4, 0.6, 0.6]
            ymin, xmin = non_zero.min(axis=0)
            ymax, xmax = non_zero.max(axis=0)
            h, w = mask_np.shape
            return [float(ymin) / h, float(xmin) / w, float(ymax) / h, float(xmax) / w]
        except Exception:
            return [0.4, 0.4, 0.6, 0.6]
