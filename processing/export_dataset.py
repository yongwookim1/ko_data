import json
import shutil
import random
import logging
from pathlib import Path
from config import RESULTS_DIR, CRAWLED_DIR

logger = logging.getLogger(__name__)

QUERIES_FILE = RESULTS_DIR / "benchmark_queries.json"


def export_dataset(n_samples=100, seed=42, output_dir="shared_dataset"):
    """
    Export dataset for sharing (Google Drive)
    - Copies images (no original data modification)
    - Creates annotations.json
    - Generates README.md
    """
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    
    output_path.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    
    if not QUERIES_FILE.exists():
        logger.error(f"Queries file not found: {QUERIES_FILE}")
        print(f"Error: {QUERIES_FILE} not found")
        return
    
    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not data:
        logger.warning("No data found")
        print("No data found")
        return
    
    random.seed(seed)
    samples = random.sample(data, min(n_samples, len(data)))
    
    exported = []
    for idx, sample in enumerate(samples, 1):
        img_path = Path(sample["image_path"])
        
        # Try absolute path first, then relative to CRAWLED_DIR
        if img_path.is_absolute() and img_path.exists():
            src_img = img_path
        else:
            # Try finding in crawled_images directory
            src_img = CRAWLED_DIR / img_path.name
            if not src_img.exists():
                # Try original path as fallback
                src_img = img_path
        
        if not src_img.exists():
            logger.warning(f"Image not found: {src_img}")
            continue
        
        img_ext = src_img.suffix
        new_img_name = f"{idx:03d}{img_ext}"
        dst_img = images_dir / new_img_name
        shutil.copy2(src_img, dst_img)
        
        exported.append({
            "id": idx,
            "image_id": sample.get("image_id", "N/A"),
            "image": f"images/{new_img_name}",
            "safety_categories": sample.get("safety_categories", []),
            "queries": sample.get("queries", {})
        })
        
        if idx % 10 == 0:
            print(f"Processed: {idx}/{len(samples)}")
    
    annotations_file = output_path / "annotations.json"
    with open(annotations_file, "w", encoding="utf-8") as f:
        json.dump(exported, f, ensure_ascii=False, indent=2)
    
    readme_content = f"""# Korean Safety MLLM Benchmark Dataset

## Dataset Overview
- **Total samples**: {len(exported)}
- **Queries per image**: 6 types
- **Random seed**: {seed}

## Structure
```
{output_dir}/
├── images/          # {len(exported)} images
├── annotations.json # Metadata and queries
└── README.md       # This file
```

## Annotations Format
Each entry contains:
- `id`: Sequential ID (1-{len(exported)})
- `image_id`: Original image identifier
- `image`: Relative path to image file
- `safety_categories`: List of safety categories
- `queries`: Dictionary with 6 query types
  - `Q1_naive`: Direct naive query
  - `Q2_naive_jailbreak`: Naive query with jailbreak prompt
  - `Q3_mllm_generated`: MLLM-generated query
  - `Q4_mllm_adaptive_jailbreak`: Adaptive jailbreak query
  - `Q5_caption_based_generated`: Caption-based query
  - `Q6_caption_based_jailbreak`: Caption-based jailbreak query

## Usage
```python
import json
from pathlib import Path

# Load annotations
with open("annotations.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Access sample
sample = data[0]
image_path = sample["image"]
queries = sample["queries"]
```

## Citation
[Add your citation information here]
"""
    
    readme_file = output_path / "README.md"
    with open(readme_file, "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    logger.info(f"Export completed: {output_path.absolute()}")
    print(f"\n✓ Export completed!")
    print(f"  Output directory: {output_path.absolute()}")
    print(f"  Images: {len(exported)}")
    print(f"  Files:")
    print(f"    - {annotations_file}")
    print(f"    - {readme_file}")
    print(f"    - images/ ({len(list(images_dir.glob('*')))} files)")


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    p = argparse.ArgumentParser(description="Export dataset for sharing")
    p.add_argument("-n", type=int, default=100, help="Number of samples (default: 100)")
    p.add_argument("-s", "--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("-o", "--output", default="shared_dataset", help="Output directory (default: shared_dataset)")
    args = p.parse_args()
    
    export_dataset(args.n, args.seed, args.output)

