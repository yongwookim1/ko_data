import requests
import logging
from config import USER_AGENT, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)


class HTTPClient:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            "Connection": "keep-alive",
            "Cache-Control": "max-age=0",
            "DNT": "1",
            "Upgrade-Insecure-Requests": "1",
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "ko-KR,ko;q=0.9",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-User": "?1",
            "Sec-Fetch-Dest": "document",
        }

    def get(self, url, referer=None, timeout=REQUEST_TIMEOUT):
        try:
            headers = self.headers.copy()
            if referer:
                headers['Referer'] = referer
            resp = self.session.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                return resp
            logger.warning(f"Status {resp.status_code}: {url}")
        except requests.RequestException as e:
            logger.error(f"Request failed: {url} | {e}")
        return None

    def download_image(self, url, referer=None):
        try:
            headers = self.headers.copy()
            headers['Accept'] = 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8'
            if referer:
                headers['Referer'] = referer
            resp = self.session.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200 and len(resp.content) > 1000:
                return resp.content
        except requests.RequestException as e:
            logger.debug(f"Image download failed: {url} | {e}")
        return None

    def set_cookies(self, cookies):
        for cookie in cookies:
            self.session.cookies.set(
                cookie['name'],
                cookie['value'],
                domain=cookie.get('domain', '')
            )

