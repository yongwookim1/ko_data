import re
import time
import random
import hashlib
import logging
from abc import ABC, abstractmethod
from config import IMAGE_DIR, DELAY_MIN, DELAY_MAX, SKIP_PATTERNS, IMAGE_EXTENSIONS
from utils import HTTPClient, MetadataManager

logger = logging.getLogger(__name__)


class BaseCrawler(ABC):
    source_type = "base"

    def __init__(self):
        self.http = HTTPClient()
        self.metadata = MetadataManager()
        self.visited = set()
        self.downloaded_urls = self.metadata.get_downloaded_urls()

    def random_sleep(self, lo=DELAY_MIN, hi=DELAY_MAX):
        time.sleep(random.uniform(lo, hi))

    def clean_visited(self, limit=10000, keep=5000):
        if len(self.visited) > limit:
            logger.info("Cleaning visited links...")
            self.visited = set(list(self.visited)[-keep:])

    def generate_filename(self, url, prefix=""):
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        ext = self._get_extension(url)
        prefix = re.sub(r'[\\/*?:"<>|]', '', prefix)[:40]
        if prefix:
            return f"{prefix}_{url_hash}.{ext}"
        return f"{url_hash}.{ext}"

    def _get_extension(self, url):
        path = url.split('?')[0].lower()
        for ext in ['png', 'webp', 'gif', 'jpeg', 'jpg']:
            if ext in path:
                return 'jpg' if ext == 'jpeg' else ext
        return 'jpg'

    def should_skip_image(self, url):
        url_lower = url.lower()
        return any(p in url_lower for p in SKIP_PATTERNS)

    def save_image(self, content, filename):
        filepath = IMAGE_DIR / filename
        if filepath.exists():
            return False
        with open(filepath, 'wb') as f:
            f.write(content)
        return True

    def download_and_save(self, img_url, source_url, prefix="", referer=None):
        if img_url in self.downloaded_urls or self.should_skip_image(img_url):
            return None

        content = self.http.download_image(img_url, referer=referer or source_url)
        if not content:
            return None

        filename = self.generate_filename(img_url, prefix)
        if self.save_image(content, filename):
            self.metadata.add(filename, source_url, img_url, self.source_type)
            self.downloaded_urls.add(img_url)
            logger.info(f"Downloaded: {filename}")
            return filename
        return None

    def save_metadata(self):
        self.metadata.save()

    @abstractmethod
    def run(self):
        pass

