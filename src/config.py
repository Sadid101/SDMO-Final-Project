import os
from dotenv import load_dotenv

load_dotenv()

DEBUG = os.getenv("DEBUG", "False").lower() == "true"
API_KEY = os.getenv("API_KEY", "")
