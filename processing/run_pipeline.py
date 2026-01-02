import gc
import logging
import sys
import torch

logger = logging.getLogger(__name__)

VLM_STAGES = {"filter", "evaluate"}


class PipelineRunner:
    def __init__(self, skip_stages=None, force_queries=False):
        from .filter_images import ImageFilter
        from .generate_queries import QueryGenerator
        from .evaluate_mllm import MLLMEvaluator
        from .judge_safety import SafetyJudge
        from .analyze_results import ResultsAnalyzer
        from config import FILTER_MODEL_PATH

        self.skip_stages = skip_stages or []
        self.force_queries = force_queries
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
        from transformers import AutoProcessor, AutoModel

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

        try:
            from transformers import Qwen3VLMoeForConditionalGeneration
            QWEN3_AVAILABLE = True
        except ImportError:
            QWEN3_AVAILABLE = False

        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        if not torch.cuda.is_available():
            raise RuntimeError("GPU required")

        gpu_count = torch.cuda.device_count()
        device_map = "auto" if gpu_count > 1 else {"": "cuda:0"}
        logger.info(f"Loading model on {gpu_count} GPU(s): {self.model_path}")

        is_qwen3 = "qwen3" in self.model_path.lower()
        try:
            if is_qwen3 and QWEN3_AVAILABLE:
                self.shared_model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    device_map=device_map,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    attn_implementation="sdpa",
                )
            else:
                self.shared_model = AutoModel.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    device_map=device_map,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    attn_implementation="sdpa",
                )
        except Exception as e:
            # Fallback to 8-bit quantization
            logger.warning(f"bfloat16 failed, using 8-bit: {e}")
            from transformers import BitsAndBytesConfig
            self.shared_model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                self.model_path,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                device_map=device_map,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

        self.shared_processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        logger.info("Model loaded")

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
            
            if stage_id == "queries":
                stage.run(force=self.force_queries)
            else:
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
            logger.error(f"Error in {stage_name}: {e}")
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
        help="Skip specified stages"
    )
    parser.add_argument(
        "--force-queries", action="store_true",
        help="Force regeneration of queries even if they already exist"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    PipelineRunner(skip_stages=args.skip or [], force_queries=args.force_queries).run()


if __name__ == "__main__":
    main()
