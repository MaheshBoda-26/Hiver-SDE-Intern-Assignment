"""Configuration module for loading and validating environment variables."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Path to project root directory
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

# Load environment variables from .env
load_dotenv(dotenv_path=ENV_PATH)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY or not GEMINI_API_KEY.strip():
    raise ValueError(
        "GEMINI_API_KEY is missing. Please set GEMINI_API_KEY in your .env file or environment variables."
    )


def get_gemini_api_key() -> str:
    """Return the validated GEMINI_API_KEY."""
    return GEMINI_API_KEY
