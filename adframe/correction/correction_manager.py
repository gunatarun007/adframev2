import logging
from typing import Dict, Any, List

logger = logging.getLogger("adframe.correction")

class CorrectionManager:
    """
    Correction Manager controls the retry loop. If the Vision Judge rejects a generated
    frame, the Correction Manager processes the feedback issues and formulates an
    adjusted placement plan, modifying prompts or refining boundaries for the next FLUX retry.
    """
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.attempts: Dict[str, int] = {}  # Tracks retries per frame index

    def should_retry(self, frame_id: str, score: float, threshold: float) -> bool:
        """
        Determines whether the loop should execute a retry based on score and budget.
        """
        current_attempt = self.attempts.get(frame_id, 0)
        
        if score >= threshold:
            logger.info(f"Frame '{frame_id}' passed with realism score {score:.2f} >= threshold {threshold:.2f}.")
            return False
            
        if current_attempt >= self.max_retries:
            logger.warning(f"Frame '{frame_id}' failed ({score:.2f}) but max retries ({self.max_retries}) are exhausted.")
            return False
            
        logger.info(f"Frame '{frame_id}' realism score {score:.2f} < threshold {threshold:.2f}. Retry scheduled.")
        return True

    def formulate_retry(self, original_plan: Dict[str, Any], judge_feedback: Dict[str, Any], frame_id: str) -> Dict[str, Any]:
        """
        Adjusts the placement plan (bounding boxes, prompts) based on the Judge's feedback.
        """
        self.attempts[frame_id] = self.attempts.get(frame_id, 0) + 1
        attempt_num = self.attempts[frame_id]
        
        logger.info(f"Formulating retry plan. Attempt: {attempt_num}/{self.max_retries}")
        
        # Deep copy original plan
        new_plan = json_deepcopy(original_plan)
        
        corrections = judge_feedback.get("corrections", {})
        
        # 1. Update text prompt
        adjust_prompt = corrections.get("adjust_prompt")
        if adjust_prompt:
            logger.info(f"Adjusting prompt to: '{adjust_prompt}'")
            new_plan["prompt"] = adjust_prompt
            
        # 2. Adjust bounding box / mask position/scale
        adjust_mask = corrections.get("adjust_mask", {})
        shift_px = adjust_mask.get("shift_px", [0, 0])
        scale_factor = adjust_mask.get("scale_factor", 1.0)
        
        if shift_px != [0, 0] or scale_factor != 1.0:
            bbox = new_plan["placement"]["bbox_2d"]
            # Convert normalized bbox to adjusted values
            # bbox layout: [y1, x1, y2, x2] or [x1, y1, x2, y2]
            # Standard is [x1, y1, x2, y2] (from our schema)
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            # Apply scale factor around the box center
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            new_width = width * scale_factor
            new_height = height * scale_factor
            
            # Apply shifts (assuming relative coordinate scaling for simplicity, e.g. shifts are in 1/1000th units if pixels)
            # Normalizing shift coordinates: divide by 1000 as a placeholder or apply directly if normalized
            shift_x_rel = shift_px[0] / 1000.0
            shift_y_rel = shift_px[1] / 1000.0
            
            new_cx = cx + shift_x_rel
            new_cy = cy + shift_y_rel
            
            # Clamp new coordinates to [0.0, 1.0]
            new_x1 = max(0.0, min(1.0, new_cx - new_width / 2))
            new_y1 = max(0.0, min(1.0, new_cy - new_height / 2))
            new_x2 = max(0.0, min(1.0, new_cx + new_width / 2))
            new_y2 = max(0.0, min(1.0, new_cy + new_height / 2))
            
            logger.info(f"Adjusted bbox coordinates from {[x1, y1, x2, y2]} to {[new_x1, new_y1, new_x2, new_y2]}")
            new_plan["placement"]["bbox_2d"] = [new_x1, new_y1, new_x2, new_y2]
            
        return new_plan

def json_deepcopy(data: Dict[str, Any]) -> Dict[str, Any]:
    import json
    return json.loads(json.dumps(data))
