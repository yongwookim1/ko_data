import logging
from bs4 import BeautifulSoup
from .base_crawler import BaseCrawler
from config import ILBE_BASE_URL, ILBE_LIST_URL, BATCH_DELAY
import time

logger = logging.getLogger(__name__)


class IlbeCrawler(BaseCrawler):
    source_type = "ilbe"

    def __init__(self):
        super().__init__()
        self.base_url = ILBE_BASE_URL
        self.list_url = ILBE_LIST_URL

    def get_post_links(self):
        resp = self.http.get(self.list_url)
        if not resp:
            return []

        soup = BeautifulSoup(resp.text, 'html.parser')
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if '/view/' in href:
                if not href.startswith('http'):
                    href = self.base_url + href if href.startswith('/') else self.base_url + '/' + href
                href = href.split('?')[0].split('#')[0]
                links.append(href)
        return list(set(links))

    def download_from_post(self, post_url):
        if post_url in self.visited:
            return
        self.visited.add(post_url)

        resp = self.http.get(post_url, referer=self.list_url)
        if not resp:
            return

        soup = BeautifulSoup(resp.text, 'html.parser')
        content_div = soup.select_one("div.post_content") or soup.select_one("div.content_body")
        if not content_div:
            return

        images = content_div.find_all('img', src=True)
        if not images:
            return

        logger.info(f"[Ilbe] Processing: {post_url} ({len(images)} images)")

        for img in images:
            img_url = img['src']
            if not img_url.startswith('http'):
                img_url = ('https:' + img_url) if img_url.startswith('//') else (self.base_url + img_url)

            if self.should_skip_image(img_url):
                continue

            self.download_and_save(img_url, post_url, referer=post_url)
            time.sleep(0.2)

    def run(self):
        logger.info("[Ilbe] Starting crawler...")

        try:
            while True:
                links = self.get_post_links()
                new_links = [l for l in links if l not in self.visited]
                logger.info(f"[Ilbe] Found {len(new_links)} new posts")

                for link in new_links:
                    self.download_from_post(link)
                    self.random_sleep()

                self.clean_visited()
                self.save_metadata()
                logger.info(f"[Ilbe] Batch done. Waiting {BATCH_DELAY}s...")
                time.sleep(BATCH_DELAY)

        except KeyboardInterrupt:
            logger.info("[Ilbe] Stopped by user")
            self.save_metadata()

