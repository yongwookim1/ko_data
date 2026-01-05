import gc
import json
import torch
import logging
from pathlib import Path
from PIL import Image
from vllm import LLM, SamplingParams
from config import FILTER_MODEL_PATH

logger = logging.getLogger(__name__)

MAX_IMAGE_SIZE = 1024


class BaseVLMStage:
    """Base class for VLM-based pipeline stages using vLLM."""

    def __init__(self, shared_model=None):
        self.model = shared_model

    def cleanup_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def load_model(self):
        if self.model is not None:
            return

        model_path = Path(FILTER_MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {FILTER_MODEL_PATH}")

        logger.info(f"Loading vLLM model from {FILTER_MODEL_PATH}")
        self.model = LLM(
            model=str(FILTER_MODEL_PATH),
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            max_model_len=8192,
        )
        logger.info("vLLM model loaded")

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
        
        # Prepare messages in OpenAI format for vLLM
        messages_batch = []
        for img, prompt in zip(images, prompts):
            messages_batch.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt}
                    ]
                }
            ])

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
            stop_token_ids=None,
        )

        try:
            outputs = self.model.chat(messages_batch, sampling_params=sampling_params)
            
            responses = []
            prefixes = ["Assistant:", "Description:", "Question:", "Rewritten question:",
                        "assistant:", "description:", "question:", "rewritten question:"]
            
            for output in outputs:
                if output.outputs:
                    text = output.outputs[0].text.strip()
                    for p in prefixes:
                        if text.startswith(p):
                            text = text[len(p):].strip()
                            break
                    responses.append(text)
                else:
                    responses.append("[ERROR_GENERATION_FAILED]")
            
            while len(responses) < len(image_prompt_pairs):
                responses.append("[ERROR_GENERATION_FAILED]")
                
            return responses
            
        except Exception as e:
            logger.error(f"vLLM inference error: {e}")
            return ["[ERROR_GENERATION_FAILED]"] * len(image_prompt_pairs)

    def parse_json(self, text, default=None):
        try:
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text.strip())
        except (json.JSONDecodeError, IndexError):
            return default if default is not None else {}

