import gc
import logging
import sys
import torch

logger = logging.getLogger(__name__)

VLM_STAGES = {"filter", "evaluate", "queries"}


class PipelineRunner:
    def __init__(self, skip_stages=None):
        from .filter_images import ImageFilter
        from .generate_queries import QueryGenerator
        from .evaluate_mllm import MLLMEvaluator
        from .judge_safety import SafetyJudge
        from .analyze_results import ResultsAnalyzer
        from config import FILTER_MODEL_PATH

        self.skip_stages = skip_stages or []
        self.model_path = FILTER_MODEL_PATH
        self.shared_model = None
        self.shared_processor = None

        self.stages = [
            ("filter", "Image Filtering", ImageFilter),
            ("queries", "Query Generation", QueryGenerator),
            ("evaluate", "MLLM Evaluation", MLLMEvaluator),
            ("judge", "Safety Judging", SafetyJudge),
            ("analyze", "Results Analysis", ResultsAnalyzer),
        ]

    def needs_vlm(self):
        """Check if any VLM stage will run."""
        return any(s not in self.skip_stages for s in VLM_STAGES)

    def unload_shared_model(self):
        if self.shared_model is not None:
            del self.shared_model
            self.shared_model = None
        if self.shared_processor is not None:
            del self.shared_processor
            self.shared_processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def load_shared_model(self):
        if self.shared_model is not None:
            return

        import os
        from pathlib import Path
        from vllm import LLM
        from transformers import AutoProcessor

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        if not torch.cuda.is_available():
            raise RuntimeError("GPU required")

        gpu_count = torch.cuda.device_count()
        logger.info(f"Loading vLLM model on {gpu_count} GPU(s): {self.model_path}")

        try:
            # Load vLLM model
            self.shared_model = LLM(
                model=str(self.model_path),
                trust_remote_code=True,
                dtype="bfloat16",
                gpu_memory_utilization=0.9,
                max_model_len=8192,
                tensor_parallel_size=gpu_count if gpu_count > 1 else 1,
                limit_mm_per_prompt={"image": 1},  # Support for vision models
            )
            
            # Load processor for chat template
            self.shared_processor = AutoProcessor.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            
            logger.info("vLLM model and processor loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load vLLM model: {e}")
            raise

    def run_stage(self, stage_id, stage_name, stage_class):
        if stage_id in self.skip_stages:
            logger.info(f"Skipping: {stage_name}")
            return True

        logger.info(f"Starting: {stage_name}")

        try:
            if stage_id in VLM_STAGES:
                stage = stage_class(
                    shared_model=self.shared_model,
                    shared_processor=self.shared_processor
                )
            else:
                stage = stage_class()
            stage.run()
            logger.info(f"Completed: {stage_name}")

            # Unload VLM after evaluation, before judge (which uses text-only model)
            if stage_id == "evaluate":
                self.unload_shared_model()
            return True

        except KeyboardInterrupt:
            logger.warning(f"Interrupted: {stage_name}")
            return False
        except Exception as e:
            logger.error(f"Error in {stage_name}: {e}", exc_info=True)
            return False

    def run(self):
        logger.info("MLLM Safety Evaluation Pipeline")
        if self.skip_stages:
            logger.info(f"Skipping: {self.skip_stages}")

        # Only load VLM if needed
        if self.needs_vlm():
            self.load_shared_model()

        for stage_id, stage_name, stage_class in self.stages:
            if not self.run_stage(stage_id, stage_name, stage_class):
                logger.error(f"Pipeline stopped at: {stage_name}")
                sys.exit(1)

        logger.info("Pipeline completed")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip", nargs="+",
        choices=["filter", "queries", "evaluate", "judge", "analyze"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    PipelineRunner(skip_stages=args.skip or []).run()


if __name__ == "__main__":
    main()
