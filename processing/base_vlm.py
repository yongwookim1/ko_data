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

    def __init__(self, shared_model=None, shared_processor=None):
        self.model = shared_model
        self.processor = shared_processor

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
        
        # Load vLLM model
        self.model = LLM(
            model=str(FILTER_MODEL_PATH),
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            max_model_len=8192,
            limit_mm_per_prompt={"image": 1},  # Support for vision models
        )
        
        # Load processor for chat template application
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            FILTER_MODEL_PATH, 
            trust_remote_code=True
        )
        
        logger.info("vLLM model and processor loaded")

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
        
        # Prepare messages in OpenAI-compatible format for vLLM
        # Following vLLM documentation for chat interface
        messages_list = []
        for img, prompt in zip(images, prompts):
            messages_list.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},  # PIL Image object
                        {"type": "text", "text": prompt}
                    ]
                }
            ])

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
        )

        try:
            # Use llm.chat() method as shown in vLLM documentation
            # This method handles chat templates automatically
            outputs = self.model.chat(messages_list, sampling_params=sampling_params)
            
            responses = []
            prefixes = ["Assistant:", "Description:", "Question:", "Rewritten question:",
                        "assistant:", "description:", "question:", "rewritten question:"]
            
            for output in outputs:
                if output.outputs:
                    text = output.outputs[0].text.strip()
                    # Remove common prefixes
                    for p in prefixes:
                        if text.startswith(p):
                            text = text[len(p):].strip()
                            break
                    responses.append(text)
                else:
                    responses.append("[ERROR_GENERATION_FAILED]")
            
            # Pad responses if needed
            while len(responses) < len(image_prompt_pairs):
                responses.append("[ERROR_GENERATION_FAILED]")
                
            return responses
            
        except Exception as e:
            logger.error(f"vLLM inference error: {e}", exc_info=True)
            # Fallback: Try with generate() method
            try:
                logger.info("Retrying with generate() method...")
                return self._fallback_generate(image_prompt_pairs, max_new_tokens)
            except Exception as e2:
                logger.error(f"Fallback also failed: {e2}", exc_info=True)
                return ["[ERROR_GENERATION_FAILED]"] * len(image_prompt_pairs)
    
    def _fallback_generate(self, image_prompt_pairs, max_new_tokens=512):
        """Fallback method using generate() if chat() fails."""
        images, prompts = zip(*image_prompt_pairs)
        
        # Apply chat template manually using processor
        messages_batch = [
            [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": p}]}]
            for img, p in zip(images, prompts)
        ]
        
        texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_batch
        ]
        
        # Create inputs with multi-modal data
        inputs = []
        for text, img in zip(texts, images):
            inputs.append({
                "prompt": text,
                "multi_modal_data": {"image": img}
            })
        
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
        )
        
        outputs = self.model.generate(inputs, sampling_params=sampling_params)
        
        responses = []
        for output in outputs:
            if output.outputs:
                responses.append(output.outputs[0].text.strip())
            else:
                responses.append("[ERROR_GENERATION_FAILED]")
        
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

