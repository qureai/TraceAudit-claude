from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
OUTPUT_DIR = PROJECT_ROOT / "output"
PROCESSED_DIR = OUTPUT_DIR / "processed"

JSONL_FILE = DATA_DIR / "Jan16_LAST2W_FIRST_50K.jsonl"
ANALYSIS_CACHE = OUTPUT_DIR / "analysis_cache.json"
TRACES_DB = OUTPUT_DIR / "traces.db"

PROMPTS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# Sampling settings for test run
TEST_SAMPLE_SIZE = 1000
FULL_SAMPLE_SIZE = 50000

# Server config
HOST = "127.0.0.1"
PORT = 8000
