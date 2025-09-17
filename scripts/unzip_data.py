#!/usr/bin/env python3
"""
Unzip a dataset archive into a target folder (default: ./data).

Usage:
  python scripts/unzip_data.py --zip MMFakeBench_test.zip --dest data

Features:
  - Creates the destination folder if it does not exist.
  - Performs a safety check to prevent Zip Slip (path traversal) attacks.
  - Overwrites existing files with the same path inside the destination.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import zipfile


def is_within_directory(directory: Path, target: Path) -> bool:
    try:
        directory = directory.resolve(strict=False)
        target = target.resolve(strict=False)
    except Exception:
        # If resolution fails for any reason, fall back to string-based check
        directory = Path(os.path.normpath(str(directory)))
        target = Path(os.path.normpath(str(target)))
    try:
        target.relative_to(directory)
        return True
    except Exception:
        return False


def safe_extract(zipf: zipfile.ZipFile, dest_dir: Path) -> None:
    for member in zipf.infolist():
        # Normalize the member path to avoid path traversal
        member_path = Path(member.filename)

        # Skip absolute paths and parent directory references explicitly
        if member_path.is_absolute() or any(part == ".." for part in member_path.parts):
            raise RuntimeError(f"Unsafe path detected in archive entry: {member.filename}")

        target_path = dest_dir / member_path

        if not is_within_directory(dest_dir, target_path):
            raise RuntimeError(f"Blocked path traversal attempt: {member.filename}")

        if member.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue

        # Ensure parent directories exist
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with zipf.open(member, "r") as src, open(target_path, "wb") as dst:
            # Stream copy to handle large files efficiently
            for chunk in iter(lambda: src.read(1024 * 1024), b""):
                dst.write(chunk)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unzip dataset into a destination folder.")
    p.add_argument(
        "--zip",
        "-z",
        dest="zip_path",
        default="MMFakeBench_test.zip",
        help="Path to the ZIP archive (default: MMFakeBench_test.zip)",
    )
    p.add_argument(
        "--dest",
        "-d",
        dest="dest_dir",
        default="data",
        help="Destination directory to extract into (default: data)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    zip_path = Path(args.zip_path)
    dest_dir = Path(args.dest_dir)

    if not zip_path.exists():
        print(f"Error: ZIP file not found: {zip_path}", file=sys.stderr)
        return 1
    if not zipfile.is_zipfile(zip_path):
        print(f"Error: Not a valid ZIP file: {zip_path}", file=sys.stderr)
        return 1

    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting '{zip_path}' into '{dest_dir}'...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            safe_extract(zf, dest_dir)
    except Exception as e:
        print(f"Extraction failed: {e}", file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

