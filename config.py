from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
IMAGE_DIR = BASE_DIR / "images" / "crawled_images"
DATA_DIR = BASE_DIR / "data"
METADATA_FILE = IMAGE_DIR / "metadata.json"
TITLES_CACHE = DATA_DIR / "controversy_titles.json"

# Create directories
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Crawler settings
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
REQUEST_TIMEOUT = 15
MIN_IMAGE_SIZE = 30 * 1024  # 30KB

# Rate limiting (seconds)
DELAY_MIN = 3.0
DELAY_MAX = 6.0
BATCH_DELAY = 30

# Image filtering
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
SKIP_PATTERNS = ['emoji', 'icon', 'logo', 'favicon', 'cc-by-nc-sa', 'assets', 'skins']

# Site URLs
NAMU_BASE_URL = "https://namu.wiki"
DC_BASE_URL = "https://www.dcinside.com/"
ILBE_BASE_URL = "https://www.ilbe.com"
ILBE_LIST_URL = "https://www.ilbe.com/list/ilbe"

# VLM Model for filtering
FILTER_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

