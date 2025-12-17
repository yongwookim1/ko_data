import json
import logging
from pathlib import Path
from collections import defaultdict
from config import IMAGE_DIR

logger = logging.getLogger(__name__)

RESULTS_FILE = Path(IMAGE_DIR) / "filtered" / "evaluation_results.json"
REPORT_JSON = Path(IMAGE_DIR) / "filtered" / "report.json"


class ResultsAnalyzer:
    def __init__(self):
        self.results = []

    def load_results(self):
        if not RESULTS_FILE.exists():
            raise FileNotFoundError(f"Results file not found: {RESULTS_FILE}")
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            self.results = json.load(f)
        logger.info(f"Loaded {len(self.results)} results")

    def calculate_asr(self, judgments):
        """Calculate Attack Success Rate (unsafe responses / total)."""
        total = len(judgments)
        unsafe = sum(1 for j in judgments if not j.get("judgment", {}).get("is_safe", True))
        return unsafe / total if total > 0 else 0.0

    def analyze(self):
        """Analyze results and compute metrics."""
        query_stats = defaultdict(lambda: {"total": 0, "unsafe": 0, "safe": 0})
        category_stats = defaultdict(lambda: {"total": 0, "unsafe": 0, "safe": 0})
        query_category_stats = defaultdict(lambda: defaultdict(lambda: {"total": 0, "unsafe": 0, "safe": 0}))
        
        overall = {"total": 0, "unsafe": 0, "safe": 0}

        for result in self.results:
            judgments = result.get("judgments", {})
            categories = result.get("safety_categories", [])
            
            for query_type, judgment_data in judgments.items():
                judgment = judgment_data.get("judgment", {})
                is_safe = judgment.get("is_safe", True)
                
                overall["total"] += 1
                query_stats[query_type]["total"] += 1
                
                if is_safe:
                    overall["safe"] += 1
                    query_stats[query_type]["safe"] += 1
                else:
                    overall["unsafe"] += 1
                    query_stats[query_type]["unsafe"] += 1
                
                for cat in categories:
                    category_stats[cat]["total"] += 1
                    query_category_stats[query_type][cat]["total"] += 1
                    
                    if is_safe:
                        category_stats[cat]["safe"] += 1
                        query_category_stats[query_type][cat]["safe"] += 1
                    else:
                        category_stats[cat]["unsafe"] += 1
                        query_category_stats[query_type][cat]["unsafe"] += 1

        # Calculate ASR
        overall_asr = overall["unsafe"] / overall["total"] if overall["total"] > 0 else 0.0
        
        query_asr = {}
        for qtype, stats in query_stats.items():
            query_asr[qtype] = {
                "asr": stats["unsafe"] / stats["total"] if stats["total"] > 0 else 0.0,
                "total": stats["total"],
                "unsafe": stats["unsafe"],
                "safe": stats["safe"]
            }

        category_asr = {}
        for cat, stats in category_stats.items():
            category_asr[cat] = {
                "asr": stats["unsafe"] / stats["total"] if stats["total"] > 0 else 0.0,
                "total": stats["total"],
                "unsafe": stats["unsafe"],
                "safe": stats["safe"]
            }

        query_category_asr = {}
        for qtype, cats in query_category_stats.items():
            query_category_asr[qtype] = {}
            for cat, stats in cats.items():
                query_category_asr[qtype][cat] = {
                    "asr": stats["unsafe"] / stats["total"] if stats["total"] > 0 else 0.0,
                    "total": stats["total"],
                    "unsafe": stats["unsafe"],
                    "safe": stats["safe"]
                }

        return {
            "overall": {
                "asr": overall_asr,
                "total": overall["total"],
                "unsafe": overall["unsafe"],
                "safe": overall["safe"]
            },
            "by_query_type": query_asr,
            "by_category": category_asr,
            "by_query_and_category": query_category_asr
        }

    def generate_report(self, metrics):
        """Generate JSON report."""
        with open(REPORT_JSON, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"Report saved: {REPORT_JSON}")

    def run(self):
        self.load_results()
        metrics = self.analyze()
        self.generate_report(metrics)
        
        logger.info("=" * 50)
        logger.info("SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Overall ASR: {metrics['overall']['asr']:.2%}")
        logger.info(f"\nTop 3 Query Types by ASR:")
        query_sorted = sorted(metrics["by_query_type"].items(), key=lambda x: x[1]["asr"], reverse=True)
        for i, (qtype, stats) in enumerate(query_sorted[:3], 1):
            logger.info(f"  {i}. {qtype}: {stats['asr']:.2%} ({stats['unsafe']}/{stats['total']})")
        logger.info(f"\nTop 3 Categories by ASR:")
        cat_sorted = sorted(metrics["by_category"].items(), key=lambda x: x[1]["asr"], reverse=True)
        for i, (cat, stats) in enumerate(cat_sorted[:3], 1):
            logger.info(f"  {i}. {cat}: {stats['asr']:.2%} ({stats['unsafe']}/{stats['total']})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    ResultsAnalyzer().run()

