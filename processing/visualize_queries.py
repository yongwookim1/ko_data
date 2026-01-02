import json
import random
import base64
import logging
from pathlib import Path
from config import RESULTS_DIR

logger = logging.getLogger(__name__)

OUTPUT_HTML = RESULTS_DIR / "query_visualization.html"
QUERIES_FILE = RESULTS_DIR / "benchmark_queries.json"


def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_mime(path):
    ext = Path(path).suffix.lower()
    return {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(ext[1:], "jpeg")


def generate_html(samples, output_path):
    html = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>Query Review ({count} samples)</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }}
        h1 {{ border-bottom: 2px solid #333; padding-bottom: 10px; }}
        .sample {{ border: 1px solid #ddd; padding: 20px; margin-bottom: 30px; }}
        .header {{ display: flex; gap: 10px; margin-bottom: 15px; font-size: 14px; color: #666; }}
        .category {{ background: #f0f0f0; padding: 2px 8px; border-radius: 3px; }}
        .content {{ display: flex; gap: 20px; }}
        .image-box img {{ max-width: 450px; max-height: 350px; object-fit: contain; border: 1px solid #ddd; }}
        .queries {{ flex: 1; }}
        .query {{ margin-bottom: 10px; padding: 10px; background: #f9f9f9; border-left: 3px solid #333; }}
        .query-label {{ font-weight: bold; font-size: 12px; color: #666; margin-bottom: 5px; }}
        .query-text {{ font-size: 14px; white-space: pre-wrap; }}
    </style>
</head>
<body>
    <h1>Query Review ({count} samples)</h1>
    {samples_html}
</body>
</html>"""

    samples_html = ""
    for idx, s in enumerate(samples, 1):
        try:
            img_path = Path(s["image_path"])
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                img_b64, mime = "", "jpeg"
            else:
                img_b64 = image_to_base64(img_path)
                mime = get_mime(img_path)
        except Exception as e:
            logger.error(f"Error loading image for sample {idx}: {e}")
            img_b64, mime = "", "jpeg"

        cats = " ".join(f'<span class="category">{c}</span>' for c in s.get("safety_categories", []))
        q = s.get("queries", {})
        
        samples_html += f"""
    <div class="sample">
        <div class="header"><strong>#{idx}</strong> | {s.get('image_id', 'N/A')} | {cats}</div>
        <div class="content">
            <div class="image-box"><img src="data:image/{mime};base64,{img_b64}" alt="{idx}"></div>
            <div class="queries">
                <div class="query"><div class="query-label">Q1: Naive</div><div class="query-text">{q.get('Q1_naive', '')}</div></div>
                <div class="query"><div class="query-label">Q2: Naive + Jailbreak</div><div class="query-text">{q.get('Q2_naive_jailbreak', '')[:500]}...</div></div>
                <div class="query"><div class="query-label">Q3: MLLM Generated</div><div class="query-text">{q.get('Q3_mllm_generated', '')}</div></div>
                <div class="query"><div class="query-label">Q4: Adaptive Jailbreak</div><div class="query-text">{q.get('Q4_mllm_adaptive_jailbreak', '')}</div></div>
                <div class="query"><div class="query-label">Q5: Caption-based</div><div class="query-text">{q.get('Q5_caption_based_generated', '')}</div></div>
                <div class="query"><div class="query-label">Q6: Caption Jailbreak</div><div class="query-text">{q.get('Q6_caption_based_jailbreak', '')}</div></div>
            </div>
        </div>
    </div>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html.format(count=len(samples), samples_html=samples_html))
    return output_path


def main(n_samples=50, seed=42):
    if not QUERIES_FILE.exists():
        logger.error(f"Queries file not found: {QUERIES_FILE}")
        print(f"Error: {QUERIES_FILE} not found")
        print(f"Make sure to run query generation first: python -m processing.generate_queries")
        return
    
    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not data:
        logger.warning("No data found in queries file")
        print("No data found")
        return
    
    random.seed(seed)
    samples = random.sample(data, min(n_samples, len(data)))
    
    output_path = generate_html(samples, OUTPUT_HTML)
    logger.info(f"Generated visualization with {len(samples)} samples: {output_path}")
    print(f"✓ Saved: {output_path}")
    print(f"  Total samples: {len(data)}")
    print(f"  Visualized: {len(samples)}")


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    p = argparse.ArgumentParser(description="Visualize query generation results")
    p.add_argument("-n", type=int, default=50, help="Number of samples to visualize (default: 50)")
    p.add_argument("-s", "--seed", type=int, default=42, help="Random seed (default: 42)")
    args = p.parse_args()
    
    main(args.n, args.seed)
