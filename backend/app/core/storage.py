from pathlib import Path
from uuid import uuid4

STORAGE_DIR = Path("static/generated")


def save_file(filename: str, content: bytes) -> str:
    """Writes content to STORAGE_DIR/<unique-prefix>-<filename> and returns its URL.

    TODO: swap this implementation for an S3 upload later without changing the
    signature (keep returning a publicly reachable URL string)."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    unique_filename = f"{uuid4().hex}-{filename}"
    path = STORAGE_DIR / unique_filename
    path.write_bytes(content)
    return f"/static/generated/{unique_filename}"
