import shutil
import logging
from pathlib import Path
from config import IMAGE_DIR

logger = logging.getLogger(__name__)


def merge_images(source_dirs, output_dir=None):
    output_dir = Path(output_dir) if output_dir else IMAGE_DIR / "merged"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for src in source_dirs:
        src = Path(src)
        if not src.exists():
            logger.warning(f"Source directory not found: {src}")
            continue
        for img in src.iterdir():
            if img.is_file():
                shutil.copy2(str(img), str(output_dir / img.name))
                count += 1
    
    logger.info(f"Merged {count} images to {output_dir}")
    return count

