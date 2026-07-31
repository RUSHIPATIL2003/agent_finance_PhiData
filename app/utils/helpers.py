"""Helper functions and utilities."""

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any, Generator, Optional


def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())


def compute_hash(data: str | bytes, algorithm: str = "sha256") -> str:
    """Compute hash of data."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    
    hasher = hashlib.new(algorithm)
    hasher.update(data)
    return hasher.hexdigest()


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to max length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Replace multiple whitespace with single space
    text = re.sub(r"\s+", " ", text)
    # Remove zero-width characters
    text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)
    # Normalize quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    # Normalize dashes
    text = text.replace("\u2013", "-").replace("\u2014", "--")
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    separator: str = " "
) -> Generator[str, None, None]:
    """Split text into overlapping chunks."""
    if not text:
        return
    
    words = text.split(separator)
    if not words:
        return
    
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = separator.join(words[start:end])
        yield chunk
        
        if end == len(words):
            break
        start = end - chunk_overlap


def safe_json_loads(data: str, default: Any = None) -> Any:
    """Safely parse JSON."""
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(data: Any, default: str = "{}") -> str:
    """Safely serialize to JSON."""
    try:
        return json.dumps(data, ensure_ascii=False)
    except (TypeError, ValueError):
        return default


def get_file_info(filepath: str | Path) -> dict[str, Any]:
    """Get file information."""
    path = Path(filepath)
    stat = path.stat()
    
    return {
        "name": path.name,
        "size": stat.st_size,
        "extension": path.suffix.lower(),
        "modified": stat.st_mtime,
        "created": stat.st_ctime,
    }


def format_bytes(bytes_value: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_value < 1024:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024
    return f"{bytes_value:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format duration as human-readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,)
):
    """Retry a function with exponential backoff."""
    import time
    import random
    
    for attempt in range(max_retries):
        try:
            return func()
        except exceptions as e:
            if attempt == max_retries - 1:
                raise
            
            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
            time.sleep(delay)
    
    raise RuntimeError("Retry logic failed unexpectedly")


class LazyProperty:
    """Descriptor for lazy property evaluation."""
    
    def __init__(self, func):
        self.func = func
        self.name = func.__name__
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        
        value = self.func(instance)
        setattr(instance, self.name, value)
        return value