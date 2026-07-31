"""Configuration loading utilities."""

import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv


def load_env_file(env_path: Optional[str] = None) -> None:
    """Load environment variables from .env file."""
    if env_path is None:
        # Look for .env in project root
        project_root = Path(__file__).parent.parent.parent
        env_path = project_root / ".env"
    
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
    else:
        # Try .env.example as fallback
        example_path = Path(env_path).with_suffix(".env.example")
        if os.path.exists(example_path):
            load_dotenv(example_path, override=True)


def get_env(key: str, default: Any = None, required: bool = False) -> Any:
    """Get environment variable with validation."""
    value = os.getenv(key, default)
    
    if required and value is None:
        raise ValueError(f"Required environment variable {key} is not set")
    
    return value


def parse_bool(value: str) -> bool:
    """Parse boolean from string."""
    return value.lower() in ("true", "1", "yes", "on")


def parse_int(value: str, default: int = 0) -> int:
    """Parse integer from string."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def parse_float(value: str, default: float = 0.0) -> float:
    """Parse float from string."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def validate_config() -> dict[str, Any]:
    """Validate all required configuration."""
    required_vars = [
        "POSTGRES_HOST",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    ]
    
    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    
    return {var: os.getenv(var) for var in required_vars}