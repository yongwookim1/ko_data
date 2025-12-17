import re
import json
import base64
import logging
from urllib.parse import urljoin, quote
from bs4 import BeautifulSoup
from seleniumbase import SB
from .base_crawler import BaseCrawler
from config import NAMU_BASE_URL, IMAGE_DIR, DATA_DIR

logger = logging.getLogger(__name__)


class NamuCrawler(BaseCrawler):
    source_type = "namuwiki"

    def __init__(self, sb=None):
        super().__init__()
        self.sb = sb
        self.base_url = NAMU_BASE_URL
        self.start_category = "분류:논란"
        self.data_file = DATA_DIR / "namu_data.jsonl"

    def set_browser(self, sb):
        self.sb = sb
        self._sync_cookies()

    def _sync_cookies(self):
        if not self.sb:
            return
        try:
            cookies = self.sb.get_cookies()
            self.http.set_cookies(cookies)
            logger.debug(f"Synced {len(cookies)} cookies")
        except Exception as e:
            logger.warning(f"Cookie sync failed: {e}")

    def fetch_page(self, url):
        logger.info(f"[Namu] Fetching: {url}")
        self.sb.uc_open_with_reconnect(url, reconnect_time=6)
        self.sb.sleep(2)

        if "Access denied" in self.sb.get_page_title():
            logger.warning("Blocked. Waiting 30s...")
            self.sb.sleep(30)
            self.sb.uc_open_with_reconnect(url, reconnect_time=10)

        self._sync_cookies()
        self._scroll_page()
        return BeautifulSoup(self.sb.get_page_source(), 'html.parser')

    def _scroll_page(self):
        self.sb.execute_script("""
            (async () => {
                const delay = ms => new Promise(r => setTimeout(r, ms));
                const totalHeight = document.body.scrollHeight;
                const step = window.innerHeight;
                for (let pos = 0; pos < totalHeight; pos += step) {
                    window.scrollTo({ top: pos, behavior: 'smooth' });
                    await delay(400);
                }
            })();
        """)
        self.sb.sleep(3)

    def get_category_articles(self):
        url = f"{self.base_url}/w/{quote(self.start_category)}"
        soup = self.fetch_page(url)
        content = soup.select_one("div[class*='wiki-content']") or soup.select_one("article")
        if not content:
            return []

        articles = []
        skip = ['분류:', '파일:', '틀:', '사용자:']
        for a in content.select("ul li a"):
            href = a.get('href')
            text = a.text.strip()
            if href and href.startswith('/w/') and not any(s in text for s in skip):
                articles.append({"url": urljoin(self.base_url, href), "title": text})

        logger.info(f"[Namu] Found {len(articles)} articles")
        return articles

    def _get_image_urls(self):
        try:
            return self.sb.execute_script("""
                const content = document.querySelector("div[class*='wiki-content']") || document.querySelector("article");
                if (!content) return [];
                const urls = [];
                for (const img of content.querySelectorAll('img')) {
                    let src = img.currentSrc || img.src;
                    if (img.srcset) {
                        for (const part of img.srcset.split(',')) {
                            const url = part.trim().split(' ')[0];
                            if (url && url.includes('i.namu.wiki')) { src = url; break; }
                        }
                    }
                    if (src && src.includes('i.namu.wiki') && img.naturalWidth > 50) urls.push(src);
                }
                return [...new Set(urls)];
            """) or []
        except Exception as e:
            logger.error(f"Failed to get image URLs: {e}")
            return []

    def _download_via_canvas(self, img_url, filepath):
        try:
            result = self.sb.execute_async_script(f"""
                var callback = arguments[arguments.length - 1];
                var imgs = document.querySelectorAll('img');
                var target = null;
                for (var img of imgs) {{
                    if (img.currentSrc === '{img_url}' || img.src === '{img_url}') {{ target = img; break; }}
                }}
                if (!target || !target.complete || target.naturalWidth < 50) {{ callback('ERROR'); return; }}
                try {{
                    var canvas = document.createElement('canvas');
                    canvas.width = target.naturalWidth;
                    canvas.height = target.naturalHeight;
                    canvas.getContext('2d').drawImage(target, 0, 0);
                    callback(canvas.toDataURL('image/png').split(',')[1]);
                }} catch(e) {{ callback('ERROR'); }}
            """)
            if result and result != 'ERROR':
                with open(filepath, 'wb') as f:
                    f.write(base64.b64decode(result))
                return True
        except Exception:
            pass
        return False

    def download_image(self, img_url, article_title, article_url):
        if self.should_skip_image(img_url) or 'i.namu.wiki' not in img_url:
            return None

        filename = self.generate_filename(img_url, article_title)
        filepath = IMAGE_DIR / filename

        if filepath.exists():
            return filename

        if self._download_via_canvas(img_url, filepath):
            self.metadata.add(filename, article_url, img_url, self.source_type)
            logger.info(f"Downloaded: {filename}")
            return filename

        content = self.http.download_image(img_url, referer=article_url)
        if content:
            with open(filepath, 'wb') as f:
                f.write(content)
            self.metadata.add(filename, article_url, img_url, self.source_type)
            logger.info(f"Downloaded: {filename}")
            return filename

        return None

    def parse_article(self, article):
        url, title = article['url'], article['title']
        if url in self.visited:
            return None
        self.visited.add(url)

        soup = self.fetch_page(url)
        content = soup.select_one("div[class*='wiki-content']") or soup.select_one("article")
        if not content:
            return None

        for tag in content(["script", "style", "iframe"]):
            tag.decompose()

        body = re.sub(r'\[\d+\]|\[편집\]|\s+', ' ', content.get_text(separator='\n')).strip()
        images = []
        for img_url in self._get_image_urls():
            fname = self.download_image(img_url, title, url)
            if fname:
                images.append(fname)
            self.random_sleep(0.5, 1.5)

        return {"title": title, "url": url, "content": body[:5000], "images": images}

    def run(self):
        logger.info("[Namu] Starting crawler...")

        try:
            with SB(uc=True, test=True, headless=True) as sb:
                self.set_browser(sb)
                articles = self.get_category_articles()

                with open(self.data_file, 'a', encoding='utf-8') as f:
                    for i, article in enumerate(articles):
                        logger.info(f"[{i+1}/{len(articles)}] {article['title']}")
                        data = self.parse_article(article)
                        if data:
                            f.write(json.dumps(data, ensure_ascii=False) + "\n")
                            f.flush()
                        self.save_metadata()
                        self.clean_visited()
                        self.random_sleep(4, 8)

        except KeyboardInterrupt:
            logger.info("[Namu] Stopped by user")
        finally:
            self.save_metadata()

