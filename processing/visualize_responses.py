import json
import random
import base64
import logging
from pathlib import Path
from config import RESULTS_DIR

logger = logging.getLogger(__name__)

OUTPUT_HTML = RESULTS_DIR / "response_visualization.html"
JUDGMENTS_FILE = RESULTS_DIR / "evaluation_results.json"


def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_mime(path):
    ext = Path(path).suffix.lower()
    return {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(ext[1:], "jpeg")


def fix_image_path(image_path):
    """Fix image path by replacing old paths with current workspace path"""
    img_path_str = str(image_path)
    
    if "/home/work/MLLM_Safety/ko_data" in img_path_str:
        img_path_str = img_path_str.replace("/home/work/MLLM_Safety/ko_data", "/home/kyw1654/ko_data")
    
    img_path = Path(img_path_str)
    
    if not img_path.exists():
        filename = img_path.name
        alternative_path = Path("/home/kyw1654/ko_data/images/crawled_images") / filename
        if alternative_path.exists():
            return alternative_path
    
    return img_path


def generate_html(samples, output_path):
    html = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>Response Review ({count} samples)</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1600px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        h1 {{ border-bottom: 2px solid #333; padding-bottom: 10px; }}
        .sample {{ border: 1px solid #ddd; padding: 20px; margin-bottom: 40px; background: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .header {{ display: flex; gap: 10px; margin-bottom: 15px; font-size: 14px; color: #666; }}
        .category {{ background: #f0f0f0; padding: 2px 8px; border-radius: 3px; }}
        .content {{ display: flex; gap: 20px; margin-bottom: 20px; }}
        .image-box img {{ max-width: 400px; max-height: 300px; object-fit: contain; border: 1px solid #ddd; }}
        .responses {{ flex: 1; }}
        .response-item {{ margin-bottom: 25px; padding: 15px; background: #f9f9f9; border-left: 4px solid #333; }}
        .query-label {{ font-weight: bold; font-size: 13px; color: #0066cc; margin-bottom: 8px; }}
        .query-text {{ font-size: 13px; background: #e8f4fd; padding: 10px; margin-bottom: 10px; border-radius: 4px; white-space: pre-wrap; }}
        .response-label {{ font-weight: bold; font-size: 13px; color: #009900; margin-bottom: 8px; }}
        .response-text {{ font-size: 13px; line-height: 1.6; white-space: pre-wrap; max-height: 400px; overflow-y: auto; margin-bottom: 10px; }}
        .judgment {{ background: #fff3cd; border: 1px solid #ffc107; padding: 10px; margin-top: 10px; border-radius: 4px; }}
        .judgment-safe {{ background: #d4edda; border-color: #28a745; }}
        .judgment-unsafe {{ background: #f8d7da; border-color: #dc3545; }}
        .judgment-header {{ font-weight: bold; font-size: 13px; margin-bottom: 5px; }}
        .judgment-details {{ font-size: 12px; color: #555; }}
        .badge {{ display: inline-block; padding: 3px 8px; border-radius: 3px; font-size: 11px; font-weight: bold; margin-right: 5px; }}
        .badge-safe {{ background: #28a745; color: white; }}
        .badge-unsafe {{ background: #dc3545; color: white; }}
        .badge-category {{ background: #6c757d; color: white; }}
        .badge-severity {{ background: #ffc107; color: #333; }}
        .collapsible {{ cursor: pointer; user-select: none; }}
        .collapsible:hover {{ background: #e0e0e0; }}
        .collapsed {{ max-height: 100px; overflow: hidden; position: relative; }}
        .collapsed::after {{ content: '... (click to expand)'; display: block; text-align: center; background: linear-gradient(transparent, #f9f9f9); padding-top: 20px; }}
    </style>
    <script>
        function toggleCollapse(elem) {{
            elem.classList.toggle('collapsed');
        }}
    </script>
</head>
<body>
    <h1>Response Review with Judge Results ({count} samples)</h1>
    {samples_html}
</body>
</html>"""

    samples_html = ""
    for idx, s in enumerate(samples, 1):
        try:
            img_path = fix_image_path(s["image_path"])
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
        
        responses_html = ""
        for q_key in ["Q1_naive", "Q2_naive_jailbreak", "Q3_mllm_generated", 
                      "Q4_mllm_adaptive_jailbreak", "Q5_caption_based_generated", 
                      "Q6_caption_based_jailbreak"]:
            judgments = s.get("judgments", {})
            if q_key not in judgments:
                continue
                
            resp_data = judgments[q_key]
            query_text = resp_data.get("query", "")
            response_text = resp_data.get("response", "")
            judgment = resp_data.get("judgment", {})
            
            label_map = {
                "Q1_naive": "Q1: Naive",
                "Q2_naive_jailbreak": "Q2: Naive + Jailbreak",
                "Q3_mllm_generated": "Q3: MLLM Generated",
                "Q4_mllm_adaptive_jailbreak": "Q4: Adaptive Jailbreak",
                "Q5_caption_based_generated": "Q5: Caption-based",
                "Q6_caption_based_jailbreak": "Q6: Caption Jailbreak"
            }
            
            is_long = len(response_text) > 500
            collapse_class = "collapsed" if is_long else ""
            
            # Build judgment HTML
            is_safe = judgment.get("is_safe", True)
            categories = judgment.get("categories", [])
            severity = judgment.get("severity", "0")
            reason = judgment.get("reason", "")
            
            judgment_class = "judgment-safe" if is_safe else "judgment-unsafe"
            safety_badge = f'<span class="badge badge-safe">SAFE</span>' if is_safe else f'<span class="badge badge-unsafe">UNSAFE</span>'
            
            judgment_html = f"""
                <div class="judgment {judgment_class}">
                    <div class="judgment-header">
                        Judge: {safety_badge}
                    </div>
                </div>"""
            
            responses_html += f"""
                <div class="response-item">
                    <div class="query-label">{label_map.get(q_key, q_key)}</div>
                    <div class="query-text">{query_text[:300]}{'...' if len(query_text) > 300 else ''}</div>
                    <div class="response-label">Response:</div>
                    <div class="response-text {collapse_class} collapsible" onclick="toggleCollapse(this)">{response_text}</div>
                    {judgment_html}
                </div>"""
        
        samples_html += f"""
    <div class="sample">
        <div class="header"><strong>#{idx}</strong> | {s.get('image_id', 'N/A')} | {cats}</div>
        <div class="content">
            <div class="image-box"><img src="data:image/{mime};base64,{img_b64}" alt="{idx}"></div>
            <div class="responses">
                {responses_html}
            </div>
        </div>
    </div>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html.format(count=len(samples), samples_html=samples_html))
    return output_path


def main(n_samples=50, seed=42):
    if not JUDGMENTS_FILE.exists():
        logger.error(f"Judgments file not found: {JUDGMENTS_FILE}")
        print(f"Error: {JUDGMENTS_FILE} not found")
        print(f"Please run judge first: python -m processing.judge_safety")
        return
    
    with open(JUDGMENTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not data:
        logger.warning("No data found in judgments file")
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
    
    p = argparse.ArgumentParser(description="Visualize evaluation responses with judge results")
    p.add_argument("-n", type=int, default=50, help="Number of samples to visualize (default: 50)")
    p.add_argument("-s", "--seed", type=int, default=42, help="Random seed (default: 42)")
    args = p.parse_args()
    
    main(args.n, args.seed)
