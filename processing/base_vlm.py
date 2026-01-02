import gc
import json
import torch
import logging
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModel
from config import FILTER_MODEL_PATH

try:
    from transformers import Qwen3VLMoeForConditionalGeneration
    QWEN3_AVAILABLE = True
except ImportError:
    QWEN3_AVAILABLE = False

logger = logging.getLogger(__name__)

MAX_IMAGE_SIZE = 1024


class BaseVLMStage:
    """Base class for VLM-based pipeline stages."""

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

    def load_image(self, path, max_size=MAX_IMAGE_SIZE):
        image = Image.open(path).convert("RGB")
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            image = image.resize(
                (int(image.width * ratio), int(image.height * ratio)),
                Image.Resampling.LANCZOS
            )
        return image

    def run_inference_batch(self, image_prompt_pairs, max_new_tokens=512):
        if not image_prompt_pairs:
            return []

        images, prompts = zip(*image_prompt_pairs)
        images = list(images)

        messages_batch = [
            [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": p}]}]
            for img, p in zip(images, prompts)
        ]
        texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_batch
        ]
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        is_qwen3 = "qwen3" in str(type(self.model)).lower()
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": self.processor.tokenizer.eos_token_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }
        if not is_qwen3:
            gen_kwargs["use_cache"] = True

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        # Validate output length matches input
        if len(generated_ids) != len(inputs.input_ids):
            logger.warning(f"Batch size mismatch: input {len(inputs.input_ids)}, output {len(generated_ids)}")

        responses = []
        prefixes = ["Assistant:", "Description:", "Question:", "Rewritten question:",
                    "assistant:", "description:", "question:", "rewritten question:"]
        for i, (inp_ids, gen_ids) in enumerate(zip(inputs.input_ids, generated_ids)):
            trimmed = gen_ids[len(inp_ids):]
            text = self.processor.decode(trimmed, skip_special_tokens=True).strip()
            for p in prefixes:
                if text.startswith(p):
                    text = text[len(p):].strip()
                    break
            responses.append(text)

        # Pad with error markers if responses are fewer than inputs
        while len(responses) < len(image_prompt_pairs):
            responses.append("[ERROR_GENERATION_FAILED]")

        del inputs, generated_ids
        self.cleanup_memory()
        return responses

    def parse_json(self, text, default=None):
        try:
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text.strip())
        except (json.JSONDecodeError, IndexError):
            return default if default is not None else {}

