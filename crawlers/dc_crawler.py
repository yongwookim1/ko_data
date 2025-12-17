import logging
from bs4 import BeautifulSoup
from .base_crawler import BaseCrawler
from config import DC_BASE_URL, BATCH_DELAY
import time

logger = logging.getLogger(__name__)


class DCCrawler(BaseCrawler):
    source_type = "dcinside"

    def __init__(self):
        super().__init__()
        self.base_url = DC_BASE_URL

    def get_main_page_links(self):
        resp = self.http.get(self.base_url)
        if not resp:
            return []

        soup = BeautifulSoup(resp.text, 'html.parser')
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if 'board/view' in href and 'javascript' not in href:
                if not href.startswith('http'):
                    href = "https:" + href
                links.append(href)
        return list(set(links))

    def download_from_post(self, post_url):
        if post_url in self.visited:
            return
        self.visited.add(post_url)

        resp = self.http.get(post_url, referer=self.base_url)
        if not resp:
            return

        soup = BeautifulSoup(resp.text, 'html.parser')
        image_list = soup.select("div.appending_file_box ul li")
        if not image_list:
            return

        logger.info(f"[DC] Processing: {post_url} ({len(image_list)} images)")

        for li in image_list:
            img_tag = li.find('a', href=True)
            if not img_tag:
                continue
            img_url = img_tag['href']
            self.download_and_save(img_url, post_url, referer=post_url)
            time.sleep(0.2)

    def run(self):
        logger.info("[DC] Starting crawler...")

        try:
            while True:
                links = self.get_main_page_links()
                new_links = [l for l in links if l not in self.visited]
                logger.info(f"[DC] Found {len(new_links)} new posts")

                for link in new_links:
                    self.download_from_post(link)
                    self.random_sleep()

                self.clean_visited()
                self.save_metadata()
                logger.info(f"[DC] Batch done. Waiting {BATCH_DELAY}s...")
                time.sleep(BATCH_DELAY)

        except KeyboardInterrupt:
            logger.info("[DC] Stopped by user")
            self.save_metadata()

