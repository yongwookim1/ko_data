import json
import logging
from urllib.parse import quote, urlparse
from bs4 import BeautifulSoup
from seleniumbase import SB
from .base_crawler import BaseCrawler
from config import NAMU_BASE_URL, DATA_DIR, MIN_IMAGE_SIZE

logger = logging.getLogger(__name__)


class GoogleImageCrawler(BaseCrawler):
    source_type = "google"

    def __init__(self, sb=None):
        super().__init__()
        self.sb = sb
        self.titles_file = DATA_DIR / "controversy_titles.json"

    def set_browser(self, sb):
        self.sb = sb

    def get_titles(self):
        if self.titles_file.exists():
            try:
                with open(self.titles_file, 'r', encoding='utf-8') as f:
                    titles = json.load(f)
                logger.info(f"Loaded {len(titles)} titles from cache")
                return titles
            except Exception:
                pass

        logger.info("Fetching titles from NamuWiki...")
        url = f"{NAMU_BASE_URL}/w/%EB%B6%84%EB%A5%98%3A%EB%85%BC%EB%9E%80"
        self.sb.uc_open_with_reconnect(url, reconnect_time=6)
        self.sb.sleep(3)

        if "Access denied" in self.sb.get_page_title():
            self.sb.sleep(30)
            self.sb.uc_open_with_reconnect(url, reconnect_time=10)

        soup = BeautifulSoup(self.sb.get_page_source(), 'html.parser')
        content = soup.select_one("div[class*='wiki-content']") or soup.select_one("article")

        titles = []
        skip = ['분류:', '파일:', '틀:', '사용자:']
        if content:
            for a in content.select("ul li a[href^='/w/']"):
                text = a.text.strip()
                if text and not any(s in text for s in skip):
                    titles.append(text)

        logger.info(f"Found {len(titles)} titles")
        with open(self.titles_file, 'w', encoding='utf-8') as f:
            json.dump(titles, f, ensure_ascii=False, indent=2)
        return titles

    def search_images(self, query, count=5):
        url = f"https://www.google.com/search?q={quote(query)}&tbm=isch&hl=ko"
        self.sb.uc_open_with_reconnect(url, reconnect_time=5)
        self.sb.sleep(2)

        try:
            self.sb.click("button#L2AGLb", timeout=1)
        except Exception:
            pass

        urls = self.sb.execute_script("""
            const urls = new Set();
            for (const script of document.scripts) {
                const text = script.textContent || '';
                const regex = /\\["(https?:\\/\\/[^"]+)",(\\d+),(\\d+)\\]/g;
                let m;
                while ((m = regex.exec(text)) !== null) {
                    const [, url, w, h] = m;
                    if (parseInt(w) > 400 && parseInt(h) > 400 &&
                        !url.includes('encrypted-tbn') && 
                        !url.includes('google.com') && 
                        !url.includes('gstatic.com') &&
                        /\\.(jpg|jpeg|png|webp|gif)/i.test(url)) {
                        urls.add(url.replace(/\\\\u003d/g,'=').replace(/\\\\u0026/g,'&'));
                    }
                }
            }
            for (const a of document.querySelectorAll('a[href*="imgurl="]')) {
                const m = a.href.match(/imgurl=([^&]+)/);
                if (m) urls.add(decodeURIComponent(m[1]));
            }
            return [...urls];
        """) or []

        downloaded = []
        for img_url in urls[:count * 2]:
            if len(downloaded) >= count or img_url in self.downloaded_urls:
                continue
            fname = self._download(img_url, query)
            if fname:
                downloaded.append(fname)
            self.random_sleep(0.3, 0.7)
        return downloaded

    def _download(self, url, query):
        if 'encrypted-tbn' in url or 'gstatic' in url:
            return None

        content = self.http.download_image(url, referer="https://www.google.com/")
        if not content or len(content) < MIN_IMAGE_SIZE:
            return None

        filename = self.generate_filename(url, query)
        if self.save_image(content, filename):
            self.metadata.add(filename, "google_search", url, self.source_type, query=query)
            self.downloaded_urls.add(url)
            logger.info(f"Downloaded: {filename} ({len(content)//1024}KB)")
            return filename
        return None

    def run(self):
        logger.info("[Google] Starting crawler...")

        try:
            with SB(uc=True, test=True, headless=True) as sb:
                self.set_browser(sb)
                titles = self.get_titles()
                if not titles:
                    logger.error("No titles found")
                    return

                for i, title in enumerate(titles):
                    logger.info(f"[{i+1}/{len(titles)}] Searching: {title}")
                    self.search_images(title, count=3)
                    self.save_metadata()
                    self.random_sleep(2, 4)

        except KeyboardInterrupt:
            logger.info("[Google] Stopped by user")
        finally:
            self.save_metadata()

