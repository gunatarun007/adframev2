import os
import json
import logging
from typing import Dict, Any, List
from PIL import Image

from adframe.config import config
from adframe.vision.vision_model import VisionModel
from adframe.scene_memory.memory_store import SceneMemory
from adframe.planner.placement_planner import PlacementPlanner
from adframe.generation.flux_generator import FluxGenerator
from adframe.judge.vision_judge import VisionJudge
from adframe.correction.correction_manager import CorrectionManager
from adframe.renderer.blend_renderer import BlendRenderer
from adframe.utils.video_helpers import extract_keyframes, compile_output_video

logger = logging.getLogger("adframe.pipeline")

class OrchestrationPipeline:
    """
    Orchestration Pipeline drives the high-level flow of the AdFrame v2 architecture.
    It coordinates keyframe extraction, scene memory accumulation, product planning,
    inpainting, evaluation, and the correction loop to compile the final visual assets.
    """
    def __init__(self, use_mock: bool = True):
        # Initialize modules
        self.vision_model = VisionModel(
            model_id=config.vlm_model_id,
            device=config.vlm_device,
            use_mock=use_mock
        )
        
        self.scene_memory = SceneMemory()
        
        self.planner = PlacementPlanner(vision_model=self.vision_model)
        
        self.generator = FluxGenerator(
            model_id=config.flux_model_id,
            device=config.flux_device,
            use_mock=use_mock
        )
        
        self.judge = VisionJudge(vision_model=self.vision_model)
        
        self.correction = CorrectionManager(max_retries=config.max_retries)
        
        self.renderer = BlendRenderer(
            model_id=config.sam_model_id,
            device=config.sam_device,
            use_mock=use_mock
        )

    def run_pipeline(self, video_path: str, product_metadata: Dict[str, Any]) -> str:
        """
        Executes the visual insertion pipeline for the target video.
        """
        logger.info(f"Starting orchestration pipeline for {video_path}")
        
        # 1. Frame Extraction
        keyframes = extract_keyframes(
            video_path=video_path,
            output_dir=config.cache_dir,
            flow_threshold=config.optical_flow_threshold,
            max_skip=config.max_frames_to_skip
        )
        
        processed_frames: List[str] = []
        
        # 2. Keyframe Processing loop
        for idx, keyframe_path in enumerate(keyframes):
            frame_id = f"frame_{idx:03d}"
            logger.info(f"Processing keyframe {idx+1}/{len(keyframes)}: {keyframe_path}")
            
            # Step A: VLM Scene Analysis
            scene_analysis_prompt = (
                "Analyze this room scene. Identify all horizontal and vertical surface structures, "
                "materials (e.g. wood, marble, metal), empty placement regions, and dominant lighting directions. "
                "Provide the result in SceneMemory format."
            )
            from adframe.planner.placement_planner import PLACEMENT_PLAN_SCHEMA
            # To get initial SceneMemory structure
            from adframe.judge.vision_judge import JUDGE_FEEDBACK_SCHEMA
            # Query the VLM for scene memory schema
            from adframe.vision.vision_model import VisionModel
            # Create a mock schema trigger
            expected_scene_schema = {
                "title": "SceneMemory",
                "type": "OBJECT"
            }
            
            frame_analysis = self.vision_model.query_json(
                prompt=scene_analysis_prompt,
                image_paths=[keyframe_path],
                expected_schema=expected_scene_schema
            )
            
            # Step B: Build/accumulate Scene Memory
            self.scene_memory.update_state(frame_analysis)
            
            # Step C: Call Placement Planner
            placement_plan = self.planner.plan_placement(
                scene_memory_state=self.scene_memory.get_state(),
                product_metadata=product_metadata
            )
            
            # Step D: Correction & Generation Loop
            score = 0.0
            attempt = 0
            final_frame_path = keyframe_path
            current_plan = placement_plan
            
            # Temporary paths for generation intermediate outputs
            mask_path = os.path.join(config.cache_dir, f"{frame_id}_mask.png")
            gen_path = os.path.join(config.cache_dir, f"{frame_id}_gen.png")
            blend_path = os.path.join(config.output_dir, f"{frame_id}_final.png")
            
            original_image = Image.open(keyframe_path).convert("RGB")
            
            while True:
                attempt += 1
                logger.info(f"Execution Loop iteration {attempt} for {frame_id}")
                
                # Draw / refine binary mask
                bbox = current_plan["placement"]["bbox_2d"]
                mask_image = self.renderer.generate_binary_mask(original_image, bbox)
                mask_image.save(mask_path)
                
                # Run FLUX Fill inpainting
                generated_image = self.generator.generate_fill(
                    image_path=keyframe_path,
                    mask_path=mask_path,
                    prompt=current_plan["prompt"],
                    negative_prompt=current_plan.get("negative_prompt")
                )
                generated_image.save(gen_path)
                
                # Blend generated pixels into the original frame background
                blended_image = self.renderer.blend_edges(original_image, generated_image, mask_image)
                blended_image.save(blend_path)
                
                # Run Vision Judge
                judge_feedback = self.judge.evaluate_frame(
                    original_frame_path=keyframe_path,
                    generated_frame_path=blend_path,
                    placement_plan=current_plan,
                    iteration=attempt
                )
                
                score = judge_feedback["score"]
                
                # Save intermediate status files for transparency
                status_path = os.path.join(config.output_dir, f"{frame_id}_status.json")
                with open(status_path, "w") as sf:
                    json.dump({
                        "attempt": attempt,
                        "plan": current_plan,
                        "feedback": judge_feedback
                    }, sf, indent=2)
                
                # Check retry status
                if self.correction.should_retry(frame_id, score, config.judge_threshold):
                    # Formulate retry
                    current_plan = self.correction.formulate_retry(current_plan, judge_feedback, frame_id)
                else:
                    logger.info(f"Generation loop completed for {frame_id} after {attempt} attempts. Score: {score:.2f}")
                    # Update Scene memory with final placement
                    self.scene_memory.record_placement(
                        frame_idx=idx,
                        product_id=product_metadata.get("product_id", "prod"),
                        bbox_2d=current_plan["placement"]["bbox_2d"]
                    )
                    final_frame_path = blend_path
                    break
                    
            processed_frames.append(final_frame_path)
            
        # 3. Compile output video from blended frames
        output_video_path = os.path.join(config.output_dir, f"output_{os.path.basename(video_path)}")
        compile_output_video(
            original_video_path=video_path,
            frame_paths=processed_frames,
            output_path=output_video_path
        )
        
        # Save final scene memory log
        self.scene_memory.save_to_file(os.path.join(config.output_dir, "final_scene_memory.json"))
        
        return output_video_path
