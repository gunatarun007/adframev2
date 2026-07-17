import os
import logging
from typing import Optional
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("adframe.generation")

class FluxGenerator:
    """
    Wrapper for FLUX.1 Fill (Inpainting model) or standard diffusion pipeline.
    Synthesizes realistic pixels within the masked region of the original frame.
    Supports a mock renderer fallback to enable complete integration testing on non-GPU systems.
    """
    def __init__(self, model_id: str = "black-forest-labs/FLUX.1-Fill-dev", device: str = "cuda", use_mock: bool = False):
        self.model_id = model_id
        self.device = device
        self.use_mock = use_mock or os.getenv("ADFRAME_MOCK_GENERATION", "true").lower() == "true"
        
        self.pipeline = None
        
        if not self.use_mock:
            try:
                import torch
                from diffusers import FluxInpaintPipeline
                logger.info(f"Loading FLUX.1 Fill model: {self.model_id} on {self.device}")
                # Load pipeline in bfloat16 or float8 for L40S efficiency
                self.pipeline = FluxInpaintPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16
                )
                self.pipeline.to(self.device)
                
                # Optional: Enable memory optimizations if VRAM is tight
                # self.pipeline.enable_model_cpu_offload()
                # self.pipeline.enable_sequential_cpu_offload()
            except Exception as e:
                logger.warning(f"Failed to load real FLUX Fill model ({e}). Falling back to MOCK generator.")
                self.use_mock = True

    def generate_fill(
        self,
        image_path: str,
        mask_path: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_steps: int = 25,
        guidance_scale: float = 30.0
    ) -> Image.Image:
        """
        Executes inpainting on the original frame within the masked region.
        """
        logger.info(f"Generating image fill with prompt: '{prompt}'")
        
        original_image = Image.open(image_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("L")
        
        if self.use_mock:
            return self._generate_mock_fill(original_image, mask_image, prompt)
            
        try:
            import torch
            
            # Execute standard FLUX Fill generation
            output = self.pipeline(
                prompt=prompt,
                image=original_image,
                mask_image=mask_image,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                width=original_image.width,
                height=original_image.height
            )
            return output.images[0]
        except Exception as e:
            logger.error(f"Error during FLUX Fill generation: {e}")
            logger.warning("Generation failed. Returning mock filled frame.")
            return self._generate_mock_fill(original_image, mask_image, prompt)

    def _generate_mock_fill(self, original_image: Image.Image, mask_image: Image.Image, prompt: str) -> Image.Image:
        """
        Draws a realistic product mockup in the masked region for end-to-end local testing.
        """
        logger.debug("Executing mock generator drawing operations.")
        # Find the bounding box of the mask
        mask_data = mask_image.load()
        width, height = mask_image.size
        
        x_indices = []
        y_indices = []
        for y in range(height):
            for x in range(width):
                if mask_data[x, y] > 127:  # Mask is active
                    x_indices.append(x)
                    y_indices.append(y)
                    
        # If mask is empty, pick a center rectangle
        if not x_indices:
            x1, y1, x2, y2 = int(width * 0.4), int(height * 0.4), int(width * 0.6), int(height * 0.6)
        else:
            x1, y1, x2, y2 = min(x_indices), min(y_indices), max(x_indices), max(y_indices)
            
        # Draw mock product inside the masked region
        output_image = original_image.copy()
        draw = ImageDraw.Draw(output_image)
        
        # Draw a solid mock bottle/box shape
        fill_color = (70, 130, 180)  # Steel Blue product body
        shadow_color = (30, 30, 30)  # Soft contact shadow
        
        # 1. Contact shadow at bottom of product
        draw.ellipse([x1, y2 - 10, x2, y2 + 10], fill=shadow_color)
        
        # 2. Product container body
        draw.rounded_rectangle([x1 + 10, y1 + 20, x2 - 10, y2 - 5], radius=15, fill=fill_color, outline=(255, 255, 255), width=2)
        
        # 3. Product cap
        cap_width = (x2 - x1) // 3
        cx1 = x1 + cap_width
        cx2 = x2 - cap_width
        draw.rectangle([cx1, y1, cx2, y1 + 20], fill=(220, 220, 220), outline=(255, 255, 255), width=1)
        
        # 4. Draw mock product label / text
        label_text = "Brand Product"
        # Extract name from prompt if possible
        for word in prompt.split():
            if len(word) > 4 and word.lower() not in ["standing", "sitting", "realistic", "table", "shadows", "sharp", "focus", "photo"]:
                label_text = word.capitalize()
                break
                
        # Draw label rectangle
        draw.rectangle([x1 + 20, (y1+y2)//2 - 15, x2 - 20, (y1+y2)//2 + 15], fill=(255, 255, 255))
        draw.text((x1 + 25, (y1+y2)//2 - 10), label_text, fill=(0, 0, 0))
        
        return output_image
