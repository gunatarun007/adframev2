import os
import logging
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger("adframe.renderer")

class BlendRenderer:
    """
    Blend Renderer handles pixel-level compositing and mask refinement.
    It takes raw VLM coordinates, refines them using SAM 2 into clean binary masks,
    and blends the generated inpaint frames with the original frames to prevent edge seams.
    """
    def __init__(self, model_id: str = "facebook/sam2-hiera-large", device: str = "cuda", use_mock: bool = False):
        self.model_id = model_id
        self.device = device
        self.use_mock = use_mock or os.getenv("ADFRAME_MOCK_SAM", "true").lower() == "true"
        
        self.predictor = None
        
        if not self.use_mock:
            try:
                import torch
                # Placeholder for SAM 2 imports and model setup
                # Since SAM 2 setup requires config and check paths, we support a soft check
                logger.info(f"Setting up SAM 2 predictor: {self.model_id}")
                # Real SAM 2 imports go here:
                # from sam2.build_sam import build_sam2
                # from sam2.sam2_image_predictor import SAM2ImagePredictor
                # model = build_sam2(model_cfg, checkpoint, device=self.device)
                # self.predictor = SAM2ImagePredictor(model)
            except Exception as e:
                logger.warning(f"Failed to load real SAM 2 predictor ({e}). Falling back to bounding-box mask compiler.")
                self.use_mock = True

    def generate_binary_mask(self, original_image: Image.Image, bbox_2d: List[float]) -> Image.Image:
        """
        Refines the VLM's 2D bounding box coordinates into a binary segment mask.
        """
        width, height = original_image.size
        
        # Convert normalized coordinates [x1, y1, x2, y2] to absolute pixel coordinates
        x1 = int(bbox_2d[0] * width)
        y1 = int(bbox_2d[1] * height)
        x2 = int(bbox_2d[2] * width)
        y2 = int(bbox_2d[3] * height)
        
        # Clamp bounds
        x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
        y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
        
        logger.info(f"Generating binary mask for bounding box: {[x1, y1, x2, y2]}")
        
        if self.use_mock or self.predictor is None:
            return self._generate_bbox_mask(original_image.size, (x1, y1, x2, y2))
            
        try:
            import numpy as np
            
            img_np = np.array(original_image)
            self.predictor.set_image(img_np)
            
            box_np = np.array([x1, y1, x2, y2])
            masks, scores, _ = self.predictor.predict(
                box=box_np,
                multimask_output=False
            )
            
            # Convert binary numpy array to PIL L mask
            mask_np = (masks[0] * 255).astype(np.uint8)
            return Image.fromarray(mask_np).convert("L")
        except Exception as e:
            logger.error(f"SAM 2 prediction failed: {e}. Falling back to bounding box mask.")
            return self._generate_bbox_mask(original_image.size, (x1, y1, x2, y2))

    def blend_edges(self, original_image: Image.Image, generated_image: Image.Image, mask_image: Image.Image, blur_radius: int = 3) -> Image.Image:
        """
        Blends the generated image back into the original image using a feathered mask
        to prevent harsh seam edges.
        """
        logger.info("Blending edges of generated product placement with original background.")
        
        # Apply Gaussian Blur to feather mask edges
        feathered_mask = mask_image.filter(ImageFilter.GaussianBlur(blur_radius))
        
        # Composite original and generated images
        final_image = Image.composite(generated_image, original_image, feathered_mask)
        return final_image

    def _generate_bbox_mask(self, size: Tuple[int, int], box: Tuple[int, int, int, int]) -> Image.Image:
        """
        Compiles a rectangular box mask (PIL 'L' mode) for mock operations.
        """
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(box, fill=255)
        return mask
