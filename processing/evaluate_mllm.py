import json
import torch
import logging
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
from config import FILTER_MODEL_PATH, IMAGE_DIR, RESULTS_DIR

try:
    from transformers import Qwen3VLMoeForConditionalGeneration
    QWEN3_AVAILABLE = True
except ImportError:
    QWEN3_AVAILABLE = False

logger = logging.getLogger(__name__)

QUERIES_FILE = Path(RESULTS_DIR) / "benchmark_queries.json"
OUTPUT_FILE = Path(RESULTS_DIR) / "evaluation_responses.json"
CHECKPOINT_FILE = Path(RESULTS_DIR) / "evaluation_checkpoint.json"
SAVE_INTERVAL = 100  # Reduced I/O frequency

BATCH_SIZE = 2


class MLLMEvaluator:
    """Evaluates MLLM responses to benchmark queries."""

    def __init__(self, shared_model=None, shared_processor=None):
        # Use shared model if provided to avoid reloading
        self.model = shared_model
        self.processor = shared_processor

    def cleanup_memory(self):
        """Aggressive memory cleanup after inference."""
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    def load_model(self):
        if self.model is not None:
            logger.info("Using shared model")
            return

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
        return self.generate_responses_batch([(image, query)])[0]

    def generate_responses_batch(self, image_query_pairs):
        if not image_query_pairs:
            return []

        images, queries = zip(*image_query_pairs)
        images = list(images)

        messages_batch = []
        for image, query in zip(images, queries):
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": query}]}]
            messages_batch.append(messages)

        texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages_batch]

        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt").to(self.model.device)

        generated_ids = None
        try:
            # Check model type and quantization
            is_qwen3 = hasattr(self.model, 'config') and "qwen3" in str(self.model.config.__class__).lower()
            is_quantized = hasattr(self.model, 'config') and getattr(self.model.config, 'quantization_config', None) is not None

            # Qwen3: Disable autocast completely to prevent scatter dtype issues
            if is_qwen3 or is_quantized:
                # Qwen3 or quantized models: Use full precision without autocast
                inference_context = torch.no_grad()
            else:
                # Other models: Use autocast for efficiency
                inference_context = torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16) if torch.cuda.is_available() else torch.no_grad()

            with torch.no_grad(), inference_context:
                # Enhanced generation parameters for complete responses
                generate_kwargs = {
                    "max_new_tokens": 2048,
                    "do_sample": False,
                    "pad_token_id": self.processor.tokenizer.eos_token_id,
                    "eos_token_id": self.processor.tokenizer.eos_token_id,
                }

                # Add use_cache only for non-Qwen3 models
                if not is_qwen3:
                    generate_kwargs["use_cache"] = True

                generated_ids = self.model.generate(**inputs, **generate_kwargs)

                responses = []
                for i, (inp_ids, gen_ids) in enumerate(zip(inputs.input_ids, generated_ids)):
                    trimmed = gen_ids[len(inp_ids):]
                    response = self.processor.decode(trimmed, skip_special_tokens=True).strip()
                    responses.append(response)
        finally:
            # Aggressive memory cleanup
            del inputs
            if generated_ids is not None:
                del generated_ids
            self.cleanup_memory()

        return responses

    def run(self):
        """Evaluate MLLM responses for all benchmark queries."""
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

        # Process samples in smaller batches to prevent memory accumulation
        SAMPLE_BATCH_SIZE = 10

        all_results = {}
        for sample_batch_start in tqdm(range(0, len(samples), SAMPLE_BATCH_SIZE), desc="Processing sample batches"):
            sample_batch_end = min(sample_batch_start + SAMPLE_BATCH_SIZE, len(samples))
            current_sample_batch = samples[sample_batch_start:sample_batch_end]

            # Load images only for current batch
            batch_tasks = []
            task_metadata = []

            for sample_idx, sample in enumerate(current_sample_batch):
                try:
                    image_path = Path(sample["image_path"])
                    if not image_path.exists():
                        logger.warning(f"Image not found: {image_path}")
                        continue

                    image = Image.open(image_path).convert("RGB")

                    # Resize image to limit memory usage (max size: 1024px on longest side)
                    max_size = 1024
                    if max(image.size) > max_size:
                        ratio = max_size / max(image.size)
                        new_size = (int(image.width * ratio), int(image.height * ratio))
                        image = image.resize(new_size, Image.Resampling.LANCZOS)
                        logger.debug(f"Resized {Path(image_path).name} from {image.size} to {new_size}")

                    queries = sample.get("queries", {})

                    for query_type, query_text in queries.items():
                        batch_tasks.append((image, query_text))
                        task_metadata.append((sample_idx, query_type, sample))

                except Exception as e:
                    logger.error(f"Error preparing {sample.get('image_id')}: {e}")
                    continue

            # Process queries in sub-batches
            for batch_start in range(0, len(batch_tasks), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(batch_tasks))
                current_batch = batch_tasks[batch_start:batch_end]
                current_metadata = task_metadata[batch_start:batch_end]

                try:
                    batch_responses = self.generate_responses_batch(current_batch)

                    for (sample_idx, query_type, sample), response in zip(current_metadata, batch_responses):
                        sample_id = sample["image_id"]
                        if sample_id not in all_results:
                            all_results[sample_id] = {
                                "image_id": sample_id,
                                "image_path": sample["image_path"],
                                "safety_categories": sample.get("safety_categories", []),
                                "responses": {}
                            }

                        all_results[sample_id]["responses"][query_type] = {
                            "query": sample["queries"][query_type],
                            "response": response
                        }

                except Exception as e:
                    logger.error(f"Error in sub-batch {batch_start//BATCH_SIZE}: {e}")
                    continue

            # Memory cleanup after processing each sample batch
            self.cleanup_memory()

        for sample_id, result in all_results.items():
            responses.append(result)
            processed.add(sample_id)

            if len(responses) % SAVE_INTERVAL == 0:
                self.save_responses(responses)
                self.save_checkpoint(processed)
                logger.info(f"Saved at {len(responses)}/{len(samples)} samples")

        self.save_responses(responses)
        self.save_checkpoint(processed)
        logger.info(f"Evaluation complete. Saved {len(responses)} responses.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    MLLMEvaluator().run()

