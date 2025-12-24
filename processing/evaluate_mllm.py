import gc
import json
import torch
import logging
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
from config import FILTER_MODEL_PATH, RESULTS_DIR

try:
    from transformers import Qwen3VLMoeForConditionalGeneration
    QWEN3_AVAILABLE = True
except ImportError:
    QWEN3_AVAILABLE = False

logger = logging.getLogger(__name__)

QUERIES_FILE = RESULTS_DIR / "benchmark_queries.json"
OUTPUT_FILE = RESULTS_DIR / "evaluation_responses.json"
CHECKPOINT_FILE = RESULTS_DIR / "evaluation_checkpoint.json"

BATCH_SIZE = 16
SAMPLE_BATCH_SIZE = 10
SAVE_INTERVAL = 100
MAX_IMAGE_SIZE = 1024


class MLLMEvaluator:
    def __init__(self, shared_model=None, shared_processor=None):
        self.model = shared_model
        self.processor = shared_processor

    def cleanup_memory(self):
        torch.cuda.empty_cache()
        gc.collect()

    def load_model(self):
        if self.model is not None:
            return

        model_path = Path(FILTER_MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {FILTER_MODEL_PATH}")

        is_qwen3 = "qwen3" in FILTER_MODEL_PATH.lower()
        model_cls = Qwen3VLMoeForConditionalGeneration if is_qwen3 and QWEN3_AVAILABLE else AutoModel
        self.model = model_cls.from_pretrained(
            FILTER_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        self.processor = AutoProcessor.from_pretrained(FILTER_MODEL_PATH, trust_remote_code=True)
        logger.info("Model loaded")

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

    def load_image(self, path):
        image = Image.open(path).convert("RGB")
        if max(image.size) > MAX_IMAGE_SIZE:
            ratio = MAX_IMAGE_SIZE / max(image.size)
            image = image.resize(
                (int(image.width * ratio), int(image.height * ratio)),
                Image.Resampling.LANCZOS
            )
        return image

    def generate_batch(self, image_query_pairs):
        if not image_query_pairs:
            return []

        images, queries = zip(*image_query_pairs)
        images = list(images)

        messages_batch = [
            [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q}]}]
            for img, q in zip(images, queries)
        ]
        texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_batch
        ]
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        is_qwen3 = "qwen3" in str(type(self.model)).lower()
        gen_kwargs = {
            "max_new_tokens": 2048,
            "do_sample": False,
            "pad_token_id": self.processor.tokenizer.eos_token_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }
        if not is_qwen3:
            gen_kwargs["use_cache"] = True

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        responses = []
        for inp_ids, gen_ids in zip(inputs.input_ids, generated_ids):
            trimmed = gen_ids[len(inp_ids):]
            responses.append(self.processor.decode(trimmed, skip_special_tokens=True).strip())

        del inputs, generated_ids
        self.cleanup_memory()
        return responses

    def run(self):
        if not QUERIES_FILE.exists():
            raise FileNotFoundError(f"Queries not found: {QUERIES_FILE}")

        with open(QUERIES_FILE, encoding="utf-8") as f:
            queries_data = json.load(f)

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

            # Prepare tasks
            tasks, metadata = [], []
            for sample in batch:
                try:
                    path = Path(sample["image_path"])
                    if not path.exists():
                        continue
                    image = self.load_image(path)
                    for qtype, qtext in sample.get("queries", {}).items():
                        tasks.append((image, qtext))
                        metadata.append((qtype, sample))
                except Exception as e:
                    logger.error(f"Error loading {sample.get('image_id')}: {e}")

            # Process in sub-batches
            for i in range(0, len(tasks), BATCH_SIZE):
                sub_tasks = tasks[i:i + BATCH_SIZE]
                sub_meta = metadata[i:i + BATCH_SIZE]

                try:
                    batch_responses = self.generate_batch(sub_tasks)
                    for (qtype, sample), resp in zip(sub_meta, batch_responses):
                        sid = sample["image_id"]
                        if sid not in all_results:
                            all_results[sid] = {
                                "image_id": sid,
                                "image_path": sample["image_path"],
                                "safety_categories": sample.get("safety_categories", []),
                                "responses": {}
                            }
                        all_results[sid]["responses"][qtype] = {
                            "query": sample["queries"][qtype],
                            "response": resp
                        }
                except Exception as e:
                    logger.error(f"Batch error: {e}")

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
