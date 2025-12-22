import logging
import sys
import torch

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Orchestrates the complete MLLM safety evaluation pipeline."""

    def __init__(self, skip_stages=None):
        from .filter_images import ImageFilter
        from .generate_queries import QueryGenerator
        from .evaluate_mllm import MLLMEvaluator
        from .judge_safety import SafetyJudge
        from .analyze_results import ResultsAnalyzer
        from config import FILTER_MODEL_PATH

        self.skip_stages = skip_stages or []
        self.model_path = FILTER_MODEL_PATH

        # Shared model instance to avoid reloading
        self.shared_model = None
        self.shared_processor = None

        self.stages = [
            ("filter", "Image Filtering", ImageFilter),
            ("queries", "Query Generation", QueryGenerator),
            ("evaluate", "MLLM Evaluation", MLLMEvaluator),
            ("judge", "Safety Judging", SafetyJudge),
            ("analyze", "Results Analysis", ResultsAnalyzer),
        ]

    def unload_shared_model(self):
        """Unload shared model to free GPU memory."""
        if self.shared_model is not None:
            logger.info("Unloading shared model to free GPU memory")
            del self.shared_model
            self.shared_model = None

        if self.shared_processor is not None:
            del self.shared_processor
            self.shared_processor = None

        # Aggressive GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        logger.info("Shared model unloaded successfully")

    def load_shared_model(self):
        """Load shared model once for all stages to avoid reloading."""
        if self.shared_model is not None:
            return

        import os
        from pathlib import Path
        from transformers import AutoProcessor, AutoModel

        # Set PyTorch memory optimization for GPU-only processing
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Non-blocking CUDA launches

        try:
            from transformers import Qwen3VLMoeForConditionalGeneration
            QWEN3_AVAILABLE = True
        except ImportError:
            QWEN3_AVAILABLE = False

        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at: {self.model_path}")

        logger.info(f"Loading shared model: {self.model_path}")

        # Check GPU availability - refuse to run on CPU
        if not torch.cuda.is_available():
            raise RuntimeError("GPU not available. This pipeline requires GPU for processing.")

        gpu_count = torch.cuda.device_count()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Available GPUs: {gpu_count}, GPU 0 memory: {gpu_memory:.1f}GB")

        try:
            # Force GPU-only processing
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                # Multi-GPU setup - distribute across all available GPUs
                device_map = "auto"
                logger.info(f"Using {gpu_count} GPUs with automatic distribution")
            else:
                # Single GPU setup
                device_map = {"": "cuda:0"}
                logger.info("Using single GPU: cuda:0")

            is_qwen3 = "qwen3" in self.model_path.lower()
            if is_qwen3 and QWEN3_AVAILABLE:
                # Try bfloat16 first, fallback to 8-bit quantization if needed
                try:
                    logger.info("Attempting to load Qwen3 with bfloat16 across GPUs...")
                    self.shared_model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                        self.model_path,
                        dtype=torch.bfloat16,
                        device_map=device_map,  # Use multi-GPU distribution
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    logger.info("Qwen3 loaded successfully with bfloat16")
                except RuntimeError as dtype_error:
                    if "scatter" in str(dtype_error).lower() or "dtype" in str(dtype_error).lower():
                        logger.warning(f"Dtype issue detected with bfloat16: {dtype_error}")
                        logger.info("Falling back to float32 for Qwen3...")
                        # Try float32 instead of bfloat16 to avoid dtype issues
                        self.shared_model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                            self.model_path,
                            dtype=torch.float32,
                            device_map=device_map,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True
                        )
                        logger.info("Qwen3 loaded successfully with float32")
                    else:
                        raise
                except RuntimeError as oom_error:
                    logger.warning(f"bfloat16 loading failed: {oom_error}")
                    logger.info("Falling back to 8-bit quantization...")

                    # 8-bit quantization as fallback - now can use multi-GPU
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_enable_fp32_cpu_offload=False
                    )

                    self.shared_model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                        self.model_path,
                        quantization_config=quantization_config,
                        device_map=device_map,  # Use multi-GPU distribution
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    logger.info("Qwen3 loaded successfully with 8-bit quantization")
                except Exception as dtype_error:
                    if "scatter" in str(dtype_error).lower() or "dtype" in str(dtype_error).lower():
                        logger.warning(f"Dtype issue detected in Qwen3 loading: {dtype_error}")
                        logger.info("Attempting final fallback with float32...")
                        # Final fallback: use float32 to ensure dtype consistency
                        self.shared_model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                            self.model_path,
                            dtype=torch.float32,
                            device_map=device_map,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True
                        )
                        logger.info("Qwen3 loaded successfully with float32 fallback")
                    else:
                        raise
            else:
                self.shared_model = AutoModel.from_pretrained(
                    self.model_path,
                    dtype=torch.bfloat16,
                    device_map=device_map,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    use_cache=True
                )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("GPU memory insufficient for model loading")
                logger.error("Try: 1) Reduce model size, 2) Use CPU, 3) Clear GPU cache")
                raise RuntimeError("Insufficient GPU memory for model loading") from e
            else:
                raise

        try:
            self.shared_processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            # Ensure processor uses GPU if available
            if hasattr(self.shared_processor, 'tokenizer') and torch.cuda.is_available():
                logger.info("Processor loaded with GPU support")
        except Exception as e:
            logger.error(f"Failed to load processor: {e}")
            raise

        # Optional model compilation for performance (skip for Qwen3 due to dtype issues)
        if not is_qwen3:
            try:
                self.shared_model = torch.compile(self.shared_model)
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile failed (non-critical): {e}")
        else:
            logger.info("Skipping torch.compile for Qwen3 model (known dtype issues)")

        logger.info("Shared model loaded successfully")

    def run_stage(self, stage_id, stage_name, stage_class):
        """Run a single pipeline stage."""
        if stage_id in self.skip_stages:
            logger.info(f"⏭️  Skipping stage: {stage_name}")
            return True

        logger.info("=" * 60)
        logger.info(f"🚀 Starting: {stage_name}")
        logger.info("=" * 60)

        # Monitor GPU memory usage
        if torch.cuda.is_available():
            gpu_memory_before = torch.cuda.mem_get_info()[0] / (1024**3)
            logger.info(f"GPU memory before {stage_name}: {gpu_memory_before:.1f}GB free")

        try:
            # Pass shared model to stages that need it
            if stage_id in ["filter", "queries", "evaluate"]:
                stage = stage_class(shared_model=self.shared_model, shared_processor=self.shared_processor)
            else:
                stage = stage_class()
            stage.run()

            # Report memory usage
            if torch.cuda.is_available():
                gpu_memory_after = torch.cuda.mem_get_info()[0] / (1024**3)
                memory_used = gpu_memory_before - gpu_memory_after
                logger.info(f"GPU memory after {stage_name}: {gpu_memory_after:.1f}GB free ({memory_used:.1f}GB used)")

            logger.info(f"✅ Completed: {stage_name}\n")

            if stage_id == "evaluate":
                self.unload_shared_model()

            return True

        except KeyboardInterrupt:
            logger.warning(f"\n⚠️  Interrupted: {stage_name}")
            logger.info("Checkpoint saved. You can resume later.")
            return False
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"❌ OOM Error in {stage_name}: {e}")
                logger.error("Try reducing batch sizes or using smaller model")
                return False
            else:
                logger.error(f"❌ Runtime Error in {stage_name}: {e}")
                logger.exception(e)
                return False
        except Exception as e:
            logger.error(f"❌ Error in {stage_name}: {e}")
            logger.exception(e)
            return False

    def run(self):
        """Run the entire pipeline."""
        logger.info("=" * 60)
        logger.info("🎯 MLLM Safety Evaluation Pipeline")
        logger.info("=" * 60)
        logger.info(f"Stages to skip: {self.skip_stages if self.skip_stages else 'None'}\n")

        # Load shared model once
        self.load_shared_model()

        for stage_id, stage_name, stage_class in self.stages:
            success = self.run_stage(stage_id, stage_name, stage_class)
            if not success:
                logger.error(f"Pipeline stopped at: {stage_name}")
                sys.exit(1)

        logger.info("=" * 60)
        logger.info("🎉 Pipeline completed successfully!")
        logger.info("=" * 60)
        logger.info("\nOutput files:")
        logger.info("  - results/benchmark_queries.json")
        logger.info("  - results/evaluation_responses.json")
        logger.info("  - results/evaluation_results.json")
        logger.info("  - results/report.json")
        logger.info("  - results/filtering_log.json")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run MLLM safety evaluation pipeline")
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=["filter", "queries", "evaluate", "judge", "analyze"],
        help="Stages to skip"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    runner = PipelineRunner(skip_stages=args.skip or [])
    runner.run()


if __name__ == "__main__":
    main()
