import json
import torch
import logging
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
from config import FILTER_MODEL_PATH, IMAGE_DIR

try:
    from transformers import Qwen3VLMoeForConditionalGeneration
    QWEN3_AVAILABLE = True
except ImportError:
    QWEN3_AVAILABLE = False

logger = logging.getLogger(__name__)

QUERIES_FILE = Path(IMAGE_DIR) / "filtered" / "benchmark_queries.json"
OUTPUT_FILE = Path(IMAGE_DIR) / "filtered" / "evaluation_responses.json"
CHECKPOINT_FILE = Path(IMAGE_DIR) / "filtered" / "evaluation_checkpoint.json"
SAVE_INTERVAL = 50


class MLLMEvaluator:
    def __init__(self):
        self.model = None
        self.processor = None

    def load_model(self):
        model_path = Path(FILTER_MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at: {FILTER_MODEL_PATH}")
        logger.info(f"Loading target MLLM: {FILTER_MODEL_PATH}")
        
        is_qwen3 = "qwen3" in FILTER_MODEL_PATH.lower()
        if is_qwen3 and QWEN3_AVAILABLE:
            self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                FILTER_MODEL_PATH,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModel.from_pretrained(
                FILTER_MODEL_PATH,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
        self.processor = AutoProcessor.from_pretrained(
            FILTER_MODEL_PATH, trust_remote_code=True
        )
        logger.info("Target MLLM loaded successfully")

    def load_checkpoint(self):
        if CHECKPOINT_FILE.exists():
            with open(CHECKPOINT_FILE, "r") as f:
                data = json.load(f)
                return set(data.get("processed", []))
        return set()

    def save_checkpoint(self, processed):
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump({"processed": list(processed)}, f)

    def save_responses(self, responses):
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(responses, f, ensure_ascii=False, indent=2)

    def generate_response(self, image, query):
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": query}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
            return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def run(self):
        if not QUERIES_FILE.exists():
            raise FileNotFoundError(f"Queries file not found: {QUERIES_FILE}")

        with open(QUERIES_FILE, "r", encoding="utf-8") as f:
            queries_data = json.load(f)

        self.load_model()
        processed = self.load_checkpoint()
        logger.info(f"Checkpoint: {len(processed)} samples already processed")

        responses = []
        if OUTPUT_FILE.exists():
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                responses = json.load(f)

        samples = [s for s in queries_data if s.get("image_id") not in processed]
        logger.info(f"Found {len(queries_data)} total, {len(samples)} remaining")

        for idx, sample in enumerate(tqdm(samples, desc="Evaluating")):
            try:
                image_path = Path(sample["image_path"])
                if not image_path.exists():
                    logger.warning(f"Image not found: {image_path}")
                    continue

                image = Image.open(image_path).convert("RGB")
                result = {
                    "image_id": sample["image_id"],
                    "image_path": sample["image_path"],
                    "safety_categories": sample.get("safety_categories", []),
                    "responses": {}
                }

                queries = sample.get("queries", {})
                for query_type, query_text in queries.items():
                    response = self.generate_response(image, query_text)
                    result["responses"][query_type] = {
                        "query": query_text,
                        "response": response
                    }

                responses.append(result)
                processed.add(sample["image_id"])

                if (idx + 1) % SAVE_INTERVAL == 0:
                    self.save_responses(responses)
                    self.save_checkpoint(processed)
                    logger.info(f"Saved at {idx + 1}/{len(samples)}")

            except Exception as e:
                logger.error(f"Error processing {sample.get('image_id')}: {e}")
                continue

        self.save_responses(responses)
        self.save_checkpoint(processed)
        logger.info(f"Evaluation complete. Saved {len(responses)} responses.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    MLLMEvaluator().run()

