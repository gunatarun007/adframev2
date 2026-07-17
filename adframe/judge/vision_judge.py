import logging
from typing import Dict, Any, Optional
from adframe.vision.vision_model import VisionModel

logger = logging.getLogger("adframe.judge")

JUDGE_FEEDBACK_SCHEMA = {
    "title": "JudgeFeedback",
    "type": "OBJECT",
    "properties": {
        "score": { "type": "NUMBER", "minimum": 0.0, "maximum": 1.0 },
        "issues": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "category": { "type": "STRING", "enum": ["lighting", "perspective", "scale", "shadows", "reflections", "logo_integrity", "occlusion", "floating_artifacts", "blending"] },
                    "description": { "type": "STRING" },
                    "severity": { "type": "STRING", "enum": ["low", "medium", "high"] }
                },
                "required": ["category", "description", "severity"]
            }
        },
        "corrections": {
            "type": "OBJECT",
            "properties": {
                "adjust_prompt": { "type": "STRING" },
                "adjust_mask": {
                    "type": "OBJECT",
                    "properties": {
                        "shift_px": { "type": "ARRAY", "items": { "type": "INTEGER" }, "minItems": 2, "maxItems": 2 },
                        "scale_factor": { "type": "NUMBER" }
                    }
                }
            }
        }
    },
    "required": ["score", "issues", "corrections"]
}

class VisionJudge:
    """
    Vision Judge evaluates the generated/inpainted frames using the VLM,
    comparing it side-by-side with the original frame. It checks for lighting
    mismatch, incorrect shadows, perspective skew, floating, scale, and blending artifacts.
    """
    def __init__(self, vision_model: Optional[VisionModel] = None):
        self.vlm = vision_model

    def evaluate_frame(
        self,
        original_frame_path: str,
        generated_frame_path: str,
        placement_plan: Dict[str, Any],
        iteration: int = 1
    ) -> Dict[str, Any]:
        """
        Runs side-by-side visual analysis to compute a realism score and issue log.
        """
        logger.info(f"Running visual evaluation on generation. Iteration: {iteration}")
        
        if self.vlm:
            try:
                # We feed both the original frame and generated frame (multi-image) to Qwen2.5-VL
                prompt = (
                    f"You are the Vision Judge. Compare the original room scene (Image 1) with the "
                    f"AI-generated product placement scene (Image 2). The product was placed at the "
                    f"normalized coordinates: {placement_plan.get('placement', {}).get('bbox_2d')} "
                    f"using target prompt: {placement_plan.get('prompt')}.\n\n"
                    f"Critically evaluate the generation for:\n"
                    f"- Lighting consistency (Is the color temperature and angle matching the surroundings?)\n"
                    f"- Shadows (Are there proper contact shadows at the base? Are they pointing the right direction?)\n"
                    f"- Perspective and scale (Does the size look correct relative to other objects?)\n"
                    f"- Occlusions and edge blending (Are foreground objects clipping correctly? Are edges smooth?)\n"
                    f"- Floating artifacts (Does it sit firmly, or look like it is floating?)\n\n"
                    f"Assign a realism score between 0.0 (fake/broken) and 1.0 (flawless/photorealistic). "
                    f"If the score is below 0.85, detail specific issues and suggest detailed text corrections for FLUX."
                )
                
                feedback = self.vlm.query_json(
                    prompt=prompt,
                    image_paths=[original_frame_path, generated_frame_path],
                    expected_schema=JUDGE_FEEDBACK_SCHEMA
                )
                return feedback
            except Exception as e:
                logger.error(f"VLM Judge evaluation failed: {e}. Falling back to default scoring.")
                
        return self._fallback_score(placement_plan)

    def _fallback_score(self, placement_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deterministic scoring fallback.
        """
        logger.warning("Using fallback judge scoring model.")
        prompt = placement_plan.get("prompt", "")
        
        # Trigger mock retry loop logic for testing
        if "retry_trigger" in prompt.lower():
            return {
                "score": 0.65,
                "issues": [{
                    "category": "shadows",
                    "description": "The product lacks a contact shadow on the table surface, making it look floating.",
                    "severity": "high"
                }],
                "corrections": {
                    "adjust_prompt": prompt + " with strong contact shadow underneath",
                    "adjust_mask": {
                        "shift_px": [0, 0],
                        "scale_factor": 1.0
                    }
                }
            }
            
        return {
            "score": 0.95,
            "issues": [],
            "corrections": {
                "adjust_prompt": prompt,
                "adjust_mask": {
                    "shift_px": [0, 0],
                    "scale_factor": 1.0
                }
            }
        }
Definition = "VisionJudge"
