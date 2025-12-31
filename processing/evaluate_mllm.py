import json
import logging
from pathlib import Path
from tqdm import tqdm
from config import RESULTS_DIR
from .base_vlm import BaseVLMStage

logger = logging.getLogger(__name__)

QUERIES_FILE = RESULTS_DIR / "benchmark_queries.json"
OUTPUT_FILE = RESULTS_DIR / "evaluation_responses.json"
CHECKPOINT_FILE = RESULTS_DIR / "evaluation_checkpoint.json"

BATCH_SIZE = 32
SAMPLE_BATCH_SIZE = 20
SAVE_INTERVAL = 100
MAX_NEW_TOKENS = 2048


class MLLMEvaluator(BaseVLMStage):
    def __init__(self, shared_model=None, shared_processor=None):
        super().__init__(shared_model, shared_processor)

    def load_checkpoint(self):
        if CHECKPOINT_FILE.exists():
            with open(CHECKPOINT_FILE) as f:
                return set(json.load(f).get("processed", []))
        return set()

    def save_checkpoint(self, processed):
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump({"processed": list(processed)}, f)

    def save_responses(self, responses):
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(responses, f, ensure_ascii=False, indent=2)

    def run(self):
        if not QUERIES_FILE.exists():
            logger.warning(f"Queries file not found: {QUERIES_FILE}. Skipping evaluation.")
            return

        with open(QUERIES_FILE, encoding="utf-8") as f:
            queries_data = json.load(f)

        if not queries_data:
            logger.warning("No queries to evaluate. Skipping.")
            return

        self.load_model()
        processed = self.load_checkpoint()

        responses = []
        if OUTPUT_FILE.exists():
            with open(OUTPUT_FILE, encoding="utf-8") as f:
                responses = json.load(f)

        samples = [s for s in queries_data if s.get("image_id") not in processed]
        logger.info(f"Processing {len(samples)}/{len(queries_data)} samples")

        all_results = {}

        for batch_start in tqdm(range(0, len(samples), SAMPLE_BATCH_SIZE), desc="Evaluating"):
            batch = samples[batch_start:batch_start + SAMPLE_BATCH_SIZE]

            # Load images once per sample, then process all queries
            for sample in batch:
                try:
                    path = Path(sample["image_path"])
                    if not path.exists():
                        logger.warning(f"Image not found: {path}")
                        continue

                    image = self.load_image(path)
                    sid = sample["image_id"]
                    queries = sample.get("queries", {})

                    if not queries:
                        continue

                    # Prepare tasks for this single image with all its queries
                    query_types = list(queries.keys())
                    query_texts = list(queries.values())

                    # Process queries in sub-batches for this image
                    all_responses = []
                    for i in range(0, len(query_texts), BATCH_SIZE):
                        sub_queries = query_texts[i:i + BATCH_SIZE]
                        # Same image paired with different queries
                        tasks = [(image, q) for q in sub_queries]
                        batch_responses = self.run_inference_batch(tasks, max_new_tokens=MAX_NEW_TOKENS)
                        all_responses.extend(batch_responses)

                    # Build result for this sample
                    result = {
                        "image_id": sid,
                        "image_path": sample["image_path"],
                        "safety_categories": sample.get("safety_categories", []),
                        "responses": {}
                    }

                    for qtype, resp in zip(query_types, all_responses):
                        result["responses"][qtype] = {
                            "query": queries[qtype],
                            "response": resp
                        }

                    all_results[sid] = result

                except Exception as e:
                    logger.error(f"Error processing {sample.get('image_id')}: {e}")

            self.cleanup_memory()

        # Save results
        for sid, result in all_results.items():
            responses.append(result)
            processed.add(sid)

        self.save_responses(responses)
        self.save_checkpoint(processed)
        logger.info(f"Done: {len(responses)} responses")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    MLLMEvaluator().run()
