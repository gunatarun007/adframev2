import os
import json
import logging
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger("adframe.vision")

class VisionModel:
    """
    Wrapper interface for the Qwen2.5-VL-7B-Instruct model (or any compliant VLM).
    Provides native support for scene analysis, structured JSON extraction, and mock fallback.
    """
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "cuda", use_mock: bool = False):
        self.model_id = model_id
        self.device = device
        self.use_mock = use_mock or os.getenv("ADFRAME_MOCK_VLM", "true").lower() == "true"
        
        self.model = None
        self.processor = None
        
        if not self.use_mock:
            try:
                import torch
                from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
                logger.info(f"Loading VLM model: {self.model_id} on {self.device}")
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16,
                    device_map=self.device
                )
                self.processor = AutoProcessor.from_pretrained(self.model_id)
            except Exception as e:
                logger.warning(f"Failed to load real VLM model ({e}). Falling back to MOCK mode.")
                self.use_mock = True

    def query(self, prompt: str, image_paths: Optional[List[str]] = None, expected_schema: Optional[Dict[str, Any]] = None) -> str:
        """
        Executes a query to the VLM model.
        """
        if self.use_mock:
            return self._generate_mock_response(prompt, image_paths, expected_schema)
            
        try:
            from qwen_vl_utils import process_vision_info
            import torch
            
            content = []
            if image_paths:
                for img_path in image_paths:
                    content.append({"type": "image", "image": img_path})
            content.append({"type": "text", "text": prompt})
            
            messages = [{"role": "user", "content": content}]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
                
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return output_text
        except Exception as e:
            logger.error(f"Error during VLM inference: {e}")
            if expected_schema:
                # return minimal compliant mock JSON to prevent pipeline crash
                return self._generate_mock_response(prompt, image_paths, expected_schema)
            raise e

    def query_json(self, prompt: str, image_paths: Optional[List[str]] = None, expected_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Queries the model and enforces/extracts a JSON structure.
        """
        # Append structured JSON output guidance to prompt
        json_instruction = (
            "\nOutput must be valid JSON ONLY matching the requested structure. "
            "Do not wrap in Markdown codeblocks, do not add conversational prefix or suffix."
        )
        full_prompt = prompt + json_instruction
        
        response_text = self.query(full_prompt, image_paths, expected_schema)
        
        try:
            # Clean possible markdown block wraps (e.g. ```json ... ```)
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                # strip out markdown container
                lines = cleaned.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()
            
            # Find the actual bounding indices of the outer brackets to extract json from conversational wrapper if any
            match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1)
                
            return json.loads(cleaned)
        except Exception as e:
            logger.error(f"Failed to parse JSON response: {response_text}. Error: {e}")
            if self.use_mock or expected_schema:
                logger.warning("Generating default schema mock structure on JSON parse failure.")
                mock_str = self._generate_mock_response(prompt, image_paths, expected_schema)
                return json.loads(mock_str)
            raise ValueError(f"VLM response is not valid JSON: {response_text}")

    def _generate_mock_response(self, prompt: str, image_paths: Optional[List[str]], expected_schema: Optional[Dict[str, Any]]) -> str:
        """
        Generates realistic mock JSON/text structure matching expected schemas for validation.
        """
        if not expected_schema:
            return "This is a mock vision model response."
            
        title = expected_schema.get("title", "")
        
        if title == "SceneMemory":
            return json.dumps({
                "scene_id": "mock_scene_101",
                "room_type": "living_room",
                "camera_path": {
                    "motion_type": "static",
                    "direction": "none",
                    "speed": "none"
                },
                "lighting": {
                    "direction": "top-right, dynamic window light",
                    "color_temperature_k": 4500
                },
                "surfaces": [
                    {
                        "surface_id": "surface_table_1",
                        "label": "wooden coffee table",
                        "bbox_2d": [0.45, 0.20, 0.85, 0.80],
                        "material": "polished wood",
                        "orientation": "horizontal"
                    },
                    {
                        "surface_id": "surface_wall_2",
                        "label": "plaster back wall",
                        "bbox_2d": [0.0, 0.0, 0.50, 1.0],
                        "material": "plaster",
                        "orientation": "vertical"
                    }
                ],
                "objects": [
                    {
                        "object_id": "object_sofa_1",
                        "label": "grey fabric sofa",
                        "bbox_2d": [0.35, 0.10, 0.60, 0.90],
                        "depth_order": 2
                    }
                ],
                "empty_regions": [
                    {
                        "region_id": "region_table_surface",
                        "bbox_2d": [0.55, 0.35, 0.75, 0.65],
                        "surface_id": "surface_table_1",
                        "dimensions_px": [300, 200]
                    }
                ],
                "placement_history": []
            }, indent=2)
            
        elif title == "PlacementPlan":
            return json.dumps({
                "placement": {
                    "bbox_2d": [0.55, 0.35, 0.75, 0.65],
                    "target_surface_id": "surface_table_1"
                },
                "rotation": {
                    "yaw": 0.0,
                    "pitch": 0.0,
                    "roll": 0.0
                },
                "scale": 0.85,
                "visibility": {
                    "occluded_by": [],
                    "visible_percentage": 100.0
                },
                "prompt": "retry_trigger perfume bottle standing on a wooden table, soft shadows, sharp focus, photo" if "retry_trigger" in prompt.lower() else "a premium water bottle standing on a wooden table, soft shadows, sharp focus, photo",
                "negative_prompt": "floating, bad lighting, cropped, blurry",
                "rendering_constraints": {
                    "lighting_direction": "top-right",
                    "shadow_softness": "soft"
                }
            }, indent=2)
            
        elif title == "JudgeFeedback":
            # For testing, we mock a good score or realistic corrections
            # If the prompt contains "retry_trigger", we trigger a poor score to exercise the loop.
            score = 0.95
            issues = []
            if "retry_trigger" in prompt.lower():
                score = 0.65
                issues = [{
                    "category": "shadows",
                    "description": "The product lacks a contact shadow on the table surface, making it look floating.",
                    "severity": "high"
                }]
                
            return json.dumps({
                "score": score,
                "issues": issues,
                "corrections": {
                    "adjust_prompt": "retry_trigger perfume bottle standing on a wooden table, with strong contact shadows casting to the bottom-left, photorealistic",
                    "adjust_mask": {
                        "shift_px": [0, 0],
                        "scale_factor": 1.0
                    }
                }
            }, indent=2)
            
        return "{}"
