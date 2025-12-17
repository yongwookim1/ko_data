import logging
import sys

logger = logging.getLogger(__name__)


class PipelineRunner:
    def __init__(self, skip_stages=None):
        from .filter_images import ImageFilter
        from .generate_queries import QueryGenerator
        from .evaluate_mllm import MLLMEvaluator
        from .judge_safety import SafetyJudge
        from .analyze_results import ResultsAnalyzer
        
        self.skip_stages = skip_stages or []
        self.stages = [
            ("filter", "Image Filtering", ImageFilter),
            ("queries", "Query Generation", QueryGenerator),
            ("evaluate", "MLLM Evaluation", MLLMEvaluator),
            ("judge", "Safety Judging", SafetyJudge),
            ("analyze", "Results Analysis", ResultsAnalyzer),
        ]

    def run_stage(self, stage_id, stage_name, stage_class):
        """Run a single stage."""
        if stage_id in self.skip_stages:
            logger.info(f"⏭️  Skipping stage: {stage_name}")
            return True

        logger.info("=" * 60)
        logger.info(f"🚀 Starting: {stage_name}")
        logger.info("=" * 60)

        try:
            stage = stage_class()
            stage.run()
            logger.info(f"✅ Completed: {stage_name}\n")
            return True
        except KeyboardInterrupt:
            logger.warning(f"\n⚠️  Interrupted: {stage_name}")
            logger.info("Checkpoint saved. You can resume later.")
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

        for stage_id, stage_name, stage_class in self.stages:
            success = self.run_stage(stage_id, stage_name, stage_class)
            if not success:
                logger.error(f"Pipeline stopped at: {stage_name}")
                sys.exit(1)

        logger.info("=" * 60)
        logger.info("🎉 Pipeline completed successfully!")
        logger.info("=" * 60)
        logger.info("\nOutput files:")
        logger.info("  - filtered/benchmark_queries.json")
        logger.info("  - filtered/evaluation_responses.json")
        logger.info("  - filtered/evaluation_results.json")
        logger.info("  - filtered/report.json")


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
