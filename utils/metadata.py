import json
import logging
from datetime import datetime
from config import METADATA_FILE

logger = logging.getLogger(__name__)


class MetadataManager:
    def __init__(self, metadata_file=None):
        self.file = metadata_file or METADATA_FILE
        self.data = self._load()

    def _load(self):
        if self.file.exists():
            try:
                with open(self.file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Metadata load error: {e}")
        return []

    def save(self):
        try:
            with open(self.file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Metadata save error: {e}")

    def add(self, filename, source_url, image_url, source_type, **extra):
        entry = {
            "filename": filename,
            "source_url": source_url,
            "image_url": image_url,
            "source_type": source_type,
            "crawled_at": datetime.now().isoformat(),
            **extra
        }
        self.data.append(entry)
        return entry

    def get_downloaded_urls(self):
        return {item.get('image_url') for item in self.data if item.get('image_url')}

    def __len__(self):
        return len(self.data)

