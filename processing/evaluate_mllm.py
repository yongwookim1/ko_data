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

BATCH_SIZE = 4
SAVE_INTERVAL = 10
MAX_NEW_TOKENS = 2048


class MLLMEvaluator(BaseVLMStage):
    def __init__(self, shared_model=None, shared_processor=None):
        super().__init__(shared_model, shared_processor)

    def run_inference_single(self, image, prompt):
        result = self.run_inference_batch([(image, prompt)], max_new_tokens=MAX_NEW_TOKENS)
        return result[0] if result else ""

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

        for idx, sample in enumerate(tqdm(samples, desc="Evaluating"), 1):
            try:
                sid = sample["image_id"]
                path = Path(sample["image_path"])
                
                if not path.exists():
                    logger.warning(f"[{sid}] Image not found: {path}")
                    continue

                image = self.load_image(path)
                queries = sample.get("queries", {})

                if not queries:
                    logger.warning(f"[{sid}] No queries found")
                    continue

                query_types = list(queries.keys())
                query_texts = list(queries.values())
                
                logger.info(f"[{sid}] Processing {len(query_types)} queries")
                
                all_responses = []
                for qtype, qtext in zip(query_types, query_texts):
                    response = self.run_inference_single(image, qtext)
                    all_responses.append(response)
                    logger.info(f"[{sid}] Query {qtype}: {'OK' if response else 'EMPTY'}")

                if len(all_responses) != len(query_types):
                    logger.warning(f"[{sid}] Response mismatch: expected {len(query_types)}, got {len(all_responses)}")
                    while len(all_responses) < len(query_types):
                        all_responses.append("")

                result = {
                    "image_id": sid,
                    "image_path": sample["image_path"],
                    "safety_categories": sample.get("safety_categories", []),
                    "responses": {}
                }

                empty_count = 0
                for qtype, resp in zip(query_types, all_responses):
                    if not resp or not resp.strip():
                        logger.warning(f"[{sid}] Empty response for query: {qtype}")
                        empty_count += 1
                    
                    result["responses"][qtype] = {
                        "query": queries[qtype],
                        "response": resp
                    }

                responses.append(result)
                processed.add(sid)
                logger.info(f"[{sid}] Success: {len(query_types)} queries, {empty_count} empty")

                if idx % SAVE_INTERVAL == 0:
                    self.save_responses(responses)
                    self.save_checkpoint(processed)
                    logger.info(f"Checkpoint: {len(responses)} samples saved")
                    self.cleanup_memory()

            except Exception as e:
                logger.error(f"[{sample.get('image_id', 'UNKNOWN')}] Exception: {e}", exc_info=True)
                self.save_responses(responses)
                continue

            self.cleanup_memory()

        # Final save
        self.save_responses(responses)
        self.save_checkpoint(processed)
        
        total_queries = sum(len(r.get("responses", {})) for r in responses)
        logger.info(f"Done: {len(responses)} samples, {total_queries} total query responses")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    MLLMEvaluator().run()
