import json
import logging
from collections import defaultdict
from config import RESULTS_DIR

logger = logging.getLogger(__name__)

RESULTS_FILE = RESULTS_DIR / "evaluation_results.json"
REPORT_FILE = RESULTS_DIR / "report.json"


class ResultsAnalyzer:
    def __init__(self):
        self.results = []

    def load_results(self):
        if not RESULTS_FILE.exists():
            raise FileNotFoundError(f"Results not found: {RESULTS_FILE}")
        with open(RESULTS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        # Handle nested list structure if present
        if data and isinstance(data[0], list):
            self.results = [item for sublist in data for item in sublist]
        else:
            self.results = data
        # Filter out non-dict items
        self.results = [r for r in self.results if isinstance(r, dict)]

    def analyze(self):
        query_stats = defaultdict(lambda: {"total": 0, "unsafe": 0, "safe": 0})
        category_stats = defaultdict(lambda: {"total": 0, "unsafe": 0, "safe": 0})
        query_cat_stats = defaultdict(lambda: defaultdict(lambda: {"total": 0, "unsafe": 0, "safe": 0}))
        overall = {"total": 0, "unsafe": 0, "safe": 0}

        for result in self.results:
            # Skip if result is not a dict
            if not isinstance(result, dict):
                logger.warning(f"Skipping non-dict result: {type(result)}")
                continue

            categories = result.get("safety_categories", [])
            judgments = result.get("judgments", {})

            # Skip if judgments is not a dict
            if not isinstance(judgments, dict):
                logger.warning(f"Skipping invalid judgments format: {type(judgments)}")
                continue

            for qtype, data in judgments.items():
                # Skip if data is not a dict
                if not isinstance(data, dict):
                    logger.warning(f"Skipping invalid judgment data for {qtype}: {type(data)}")
                    continue

                judgment = data.get("judgment", {})
                if not isinstance(judgment, dict):
                    logger.warning(f"Skipping invalid judgment format for {qtype}: {type(judgment)}")
                    continue

                is_safe = judgment.get("is_safe", True)

                overall["total"] += 1
                query_stats[qtype]["total"] += 1

                key = "safe" if is_safe else "unsafe"
                overall[key] += 1
                query_stats[qtype][key] += 1

                for cat in categories:
                    category_stats[cat]["total"] += 1
                    category_stats[cat][key] += 1
                    query_cat_stats[qtype][cat]["total"] += 1
                    query_cat_stats[qtype][cat][key] += 1

        def calc_asr(stats):
            return stats["unsafe"] / stats["total"] if stats["total"] > 0 else 0.0

        return {
            "overall": {"asr": calc_asr(overall), **overall},
            "by_query_type": {
                q: {"asr": calc_asr(s), **s} for q, s in query_stats.items()
            },
            "by_category": {
                c: {"asr": calc_asr(s), **s} for c, s in category_stats.items()
            },
            "by_query_and_category": {
                q: {c: {"asr": calc_asr(s), **s} for c, s in cats.items()}
                for q, cats in query_cat_stats.items()
            }
        }

    def run(self):
        self.load_results()
        metrics = self.analyze()

        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        # Print summary
        logger.info(f"Overall ASR: {metrics['overall']['asr']:.2%}")

        query_sorted = sorted(
            metrics["by_query_type"].items(),
            key=lambda x: x[1]["asr"],
            reverse=True
        )
        logger.info("Top query types by ASR:")
        for qtype, stats in query_sorted[:3]:
            logger.info(f"  {qtype}: {stats['asr']:.2%} ({stats['unsafe']}/{stats['total']})")

        cat_sorted = sorted(
            metrics["by_category"].items(),
            key=lambda x: x[1]["asr"],
            reverse=True
        )
        if cat_sorted:
            logger.info("Top categories by ASR:")
            for cat, stats in cat_sorted[:3]:
                logger.info(f"  {cat}: {stats['asr']:.2%} ({stats['unsafe']}/{stats['total']})")

        logger.info(f"Report saved: {REPORT_FILE}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    ResultsAnalyzer().run()
