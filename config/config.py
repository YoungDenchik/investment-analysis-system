import os
from dotenv import load_dotenv

# Завантажуємо .env
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, 'settings_example.env')
load_dotenv(ENV_PATH)

POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_PORT = int(os.getenv('POSTGRES_PORT', 5432))
POSTGRES_DB = os.getenv('POSTGRES_DB', 'market_data')
POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'secret')
CFG_PROD_FILE = os.getenv('CFG_PROD_FILE')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')

# REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
# REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
# REDIS_DB = int(os.getenv('REDIS_DB', 0))
# REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', '')
