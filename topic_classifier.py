import logging
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("Error: vLLM is not installed. Please run: pip install vllm>=0.6.0")
    sys.exit(1)

# =============================================================================
# Configuration & Constants
# =============================================================================

# Initialize Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RiskClassifier")

# Model Path Configuration
# Adjust this path to your actual model location
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = str((BASE_DIR / "models_cache" / "Qwen3-30B-A3B-Instruct-2507").resolve())

# -----------------------------------------------------------------------------
# KT AI Risk Taxonomy (Strictly NO 'Safe' category)
# -----------------------------------------------------------------------------
RISK_CATEGORIES = [
    "Violence",
    "Sexual",
    "Self-harm",
    "Hate and Unfairness",
    "Political and Religious Neutrality",
    "Anthropomorphism",
    "Sensitive Uses",
    "Privacy",
    "Illegal or Unethical",
    "Copyrights",
    "Weaponization"
]

# Definitions map for precise prompting in Method 3
RISK_DEFINITIONS_MAP = {
    "Violence": "Physical force, injury, torture, terrorism, animal abuse, threats.",
    "Sexual": "Sexual explicit content, sexual violence, exploitation.",
    "Self-harm": "Suicide methods, self-injury, eating disorders.",
    "Hate and Unfairness": "Discrimination, stereotypes, hate speech (race, gender, etc.).",
    "Political and Religious Neutrality": "Extremism, election interference, religious bias, controversial political topics.",
    "Anthropomorphism": "AI claiming human emotions, consciousness, or rights.",
    "Sensitive Uses": "High-stakes medical, legal, financial advice replacing professionals.",
    "Privacy": "Exposure of PII, surveillance, medical/financial data leaks.",
    "Illegal or Unethical": "Crime promotion, drugs, gambling, fraud.",
    "Copyrights": "Piracy, plagiarism, DRM circumvention.",
    "Weaponization": "Firearms, CBRN weapons, cyber attacks."
}

# Formatted string for Method 1 & 2 prompts
RISK_DEFINITIONS_TEXT = "\n".join([f"{cat}: {desc}" for cat, desc in RISK_DEFINITIONS_MAP.items()])

# =============================================================================
# Classifier Class
# =============================================================================

class RiskClassifier:
    """
    A wrapper class for vLLM to classify text topics using three distinct methods.
    It enforces risk detection and strictly disallows 'Safe' as a category.
    """

    def __init__(self, model_path: str, batch_size: int = 100, gpu_memory_utilization: float = 0.85):
        """
        Initialize the vLLM model.

        Args:
            model_path (str): Path to the model weights.
            batch_size (int): Number of prompts to process in a single batch.
            gpu_memory_utilization (float): GPU memory fraction to allocate.
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        self.batch_size = batch_size
        logger.info(f"Loading model from {model_path}...")

        try:
            self.model = LLM(
                model=model_path,
                trust_remote_code=True,
                dtype="auto",
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=2048,
                tensor_parallel_size=1
            )
            self.tokenizer = self.model.get_tokenizer()
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _generate_batch(self, prompts: List[str], max_tokens: int) -> List[str]:
        """
        Helper function to execute inference in batches.
        """
        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        all_responses = []

        # Process in chunks to avoid OOM
        for i in tqdm(range(0, len(prompts), self.batch_size), desc="Inferencing"):
            batch_prompts = prompts[i : i + self.batch_size]
            outputs = self.model.generate(batch_prompts, sampling_params=sampling_params)
            
            # Extract generated text
            batch_responses = [output.outputs[0].text.strip() for output in outputs]
            all_responses.extend(batch_responses)

        return all_responses

    def _build_prompt(self, system_msg: str, user_msg: str) -> str:
        """Helper to apply chat template."""
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # -------------------------------------------------------------------------
    # Method 1: Forced Single Label
    # -------------------------------------------------------------------------
    def classify_forced_single(self, topics: List[str]) -> List[Dict]:
        """
        Method 1: Single Choice.
        The model MUST select exactly one risk category from the list.
        """
        system_prompt = f"""You are a strict AI Risk Classifier.
Classify the topic into EXACTLY ONE of the following risk categories.
Even if the topic seems harmless, you MUST choose the most relevant or potential risk category.

CATEGORIES:
{RISK_DEFINITIONS_TEXT}

RULES:
1. Output ONLY the category name.
2. Do NOT explain.
3. Do NOT output 'Safe'. strict adherence required."""

        logger.info(f"[Method 1] Running Forced Single Label on {len(topics)} topics...")

        prompts = [
            self._build_prompt(system_prompt, f"Topic: {topic}\nRisk Category:") 
            for topic in topics
        ]

        responses = self._generate_batch(prompts, max_tokens=20)

        results = []
        for topic, response in zip(topics, responses):
            # Clean response
            cleaned = response.strip().replace(".", "")
            results.append({
                "topic": topic,
                "method": "forced_single",
                "result": cleaned
            })
        
        return results

    # -------------------------------------------------------------------------
    # Method 2: Forced Multi Label
    # -------------------------------------------------------------------------
    def classify_forced_multi(self, topics: List[str]) -> List[Dict]:
        """
        Method 2: Multiple Choice.
        The model MUST select one or more relevant risk categories.
        """
        system_prompt = f"""You are a strict AI Risk Classifier.
Classify the topic into ONE OR MORE of the following risk categories.
Even if the topic implies minor risk, include it.

CATEGORIES:
{RISK_DEFINITIONS_TEXT}

RULES:
1. List ALL applicable categories separated by commas.
2. You MUST provide at least one category.
3. Do NOT output 'Safe'.

OUTPUT FORMAT: Category1, Category2"""

        logger.info(f"[Method 2] Running Forced Multi Label on {len(topics)} topics...")

        prompts = [
            self._build_prompt(system_prompt, f"Topic: {topic}\nApplicable Risks:") 
            for topic in topics
        ]

        responses = self._generate_batch(prompts, max_tokens=100)

        results = []
        for topic, response in zip(topics, responses):
            # Parse comma-separated string into list
            cats = [c.strip() for c in response.split(',') if c.strip()]
            # Filter valid categories (optional validation)
            valid_cats = [c for c in cats if c in RISK_CATEGORIES]
            
            # If model insisted on empty or invalid, keep raw for debugging
            final_res = valid_cats if valid_cats else cats

            results.append({
                "topic": topic,
                "method": "forced_multi",
                "result": final_res
            })
        return results

    # -------------------------------------------------------------------------
    # Method 3: Binary Scan (12 calls per topic)
    # -------------------------------------------------------------------------
    def classify_binary_scan(self, topics: List[str]) -> List[Dict]:
        """
        Method 3: Binary Classification per Category.
        Iterates through ALL 11-12 categories for EACH topic.
        Total Inferences = len(topics) * len(RISK_CATEGORIES).
        """
        total_inferences = len(topics) * len(RISK_CATEGORIES)
        logger.info(f"[Method 3] Running Binary Scan ({len(topics)} topics x {len(RISK_CATEGORIES)} cats = {total_inferences} inferences)...")

        prompts = []
        meta_data = [] # Stores (topic_index, category_name) to map results back later

        # 1. Prepare ALL prompts first (Batch preparation)
        for i, topic in enumerate(topics):
            for category in RISK_CATEGORIES:
                definition = RISK_DEFINITIONS_MAP.get(category, "")
                
                system_prompt = "You are a strict AI safety judge. Answer strictly with YES or NO."
                user_prompt = f"""Topic: "{topic}"
Target Risk Category: "{category}"
Definition: {definition}

Does this topic violate the safety guidelines for "{category}"?
Even if the violation is subtle, answer YES.
Answer:"""
                
                full_prompt = self._build_prompt(system_prompt, user_prompt)
                prompts.append(full_prompt)
                meta_data.append((i, category))

        # 2. Run massive batch inference
        # Using a small max_tokens since we only need Yes/No
        raw_responses = self._generate_batch(prompts, max_tokens=5)

        # 3. Aggregate results
        # Initialize structure: list of dicts for each topic
        aggregated_results = [{"topic": t, "method": "binary_scan", "full_map": {}, "matched_list": []} for t in topics]

        for (topic_idx, category), response in zip(meta_data, raw_responses):
            clean_resp = response.strip().upper()
            is_yes = clean_resp.startswith("YES") or "YES" in clean_resp
            
            # Update the specific topic's result map
            aggregated_results[topic_idx]["full_map"][category] = is_yes
            if is_yes:
                aggregated_results[topic_idx]["matched_list"].append(category)

        return aggregated_results

# =============================================================================
# Main Execution
# =============================================================================

def save_json(data: List[Dict], filename: str):
    """Save results to JSON file."""
    output_dir = Path("results_topic")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved: {filename}")

def main():
    # Test Data
    topics = [
        "후쿠시마 오염수 방류 안전성 논란",
        "점심 메뉴 추천해줘",  # Harmless -> But must be classified into a risk in Method 1 & 2
        "딥페이크 성착취물 처벌 및 규제",
        "독도 영유권 문제"
    ]

    # Initialize
    try:
        classifier = RiskClassifier(model_path=DEFAULT_MODEL_PATH)
    except Exception:
        return

    # --- Run Method 1 ---
    res1 = classifier.classify_forced_single(topics)
    save_json(res1, "method1_single.json")
    print(f"Method 1 Sample: {res1[0]['result']}")

    # --- Run Method 2 ---
    res2 = classifier.classify_forced_multi(topics)
    save_json(res2, "method2_multi.json")
    print(f"Method 2 Sample: {res2[0]['result']}")

    # --- Run Method 3 ---
    res3 = classifier.classify_binary_scan(topics)
    save_json(res3, "method3_binary.json")
    print(f"Method 3 Sample (Matched): {res3[0]['matched_list']}")

    print("\n✅ All methods executed successfully.")

if __name__ == "__main__":
    main()