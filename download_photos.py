#!/usr/bin/env python3
"""Download all photos from kuva.swiss event 196."""

import json
import os
import sys
import time
import urllib.request
import urllib.error

API_BASE = "https://api.kuva.swiss/public-events/196/pictures"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "photos")
METADATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metadata.json")

# Pagination settings for metadata fetch
META_BATCH_SIZE = 100

# Download settings
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds between retries
DELAY_BETWEEN_DOWNLOADS = 0.5  # seconds between successful downloads
MIN_FILE_SIZE = 1024  # 1KB minimum to consider a file valid


def fetch_all_metadata():
    """Fetch all photo metadata from the API, with caching."""
    if os.path.exists(METADATA_FILE):
        print(f"Loading cached metadata from {METADATA_FILE}")
        with open(METADATA_FILE, "r") as f:
            photos = json.load(f)
        print(f"  Loaded {len(photos)} photos from cache")
        return photos

    print("Fetching photo metadata from API...")
    all_photos = []
    skip = 0

    # First request to get total
    url = f"{API_BASE}?skip=0&limit=1"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
    total = data["total"]
    print(f"  Total photos: {total}")

    while skip < total:
        url = f"{API_BASE}?skip={skip}&limit={META_BATCH_SIZE}"
        for attempt in range(MAX_RETRIES):
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
                all_photos.extend(data["data"])
                fetched = len(all_photos)
                print(f"  Fetched metadata: {fetched}/{total}", end="\r")
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"\n  Retry metadata fetch at skip={skip}: {e}")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"\n  FAILED metadata fetch at skip={skip}: {e}")
                    raise
        skip += META_BATCH_SIZE
        time.sleep(0.2)  # Be nice to the API

    print(f"\n  Fetched all {len(all_photos)} photo metadata entries")

    # Cache metadata to disk
    with open(METADATA_FILE, "w") as f:
        json.dump(all_photos, f)
    print(f"  Saved metadata cache to {METADATA_FILE}")

    return all_photos


def make_filename(index, photo):
    """Generate filename like photo_0001_1198861_GAZ76775.webp"""
    seq = str(index + 1).zfill(4)
    pic_id = photo["id"]
    original = os.path.splitext(photo["filename"])[0]
    return f"photo_{seq}_{pic_id}_{original}.webp"


def is_already_downloaded(filepath):
    """Check if file exists and has a plausible size."""
    if not os.path.exists(filepath):
        return False
    size = os.path.getsize(filepath)
    if size < MIN_FILE_SIZE:
        return False
    return True


def download_photo(url, filepath):
    """Download a single photo with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if len(data) < MIN_FILE_SIZE:
                raise ValueError(f"Downloaded file too small: {len(data)} bytes")
            with open(filepath, "wb") as f:
                f.write(data)
            return len(data)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  Retry {attempt + 1}/{MAX_RETRIES}: {e}")
                time.sleep(RETRY_DELAY)
            else:
                raise


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    photos = fetch_all_metadata()
    total = len(photos)

    print(f"\nDownloading {total} photos to {OUTPUT_DIR}")
    print(f"  Rate limit: {DELAY_BETWEEN_DOWNLOADS}s between downloads")
    print()

    downloaded = 0
    skipped = 0
    failed = 0
    failed_list = []

    for i, photo in enumerate(photos):
        filename = make_filename(i, photo)
        filepath = os.path.join(OUTPUT_DIR, filename)

        if is_already_downloaded(filepath):
            skipped += 1
            # Print progress periodically for skipped files
            if skipped % 100 == 0 or i == total - 1:
                print(f"[{i+1}/{total}] Skipped {skipped} already downloaded", end="\r")
            continue

        url = photo.get("hdUrl")
        if not url:
            print(f"[{i+1}/{total}] {filename} — NO HD URL, skipping")
            failed += 1
            failed_list.append({"index": i, "id": photo["id"], "reason": "no hdUrl"})
            continue

        try:
            size = download_photo(url, filepath)
            downloaded += 1
            size_mb = size / (1024 * 1024)
            print(f"[{i+1}/{total}] {filename} — {size_mb:.1f} MB")
            time.sleep(DELAY_BETWEEN_DOWNLOADS)
        except Exception as e:
            failed += 1
            failed_list.append({"index": i, "id": photo["id"], "reason": str(e)})
            print(f"[{i+1}/{total}] {filename} — FAILED: {e}")

    print(f"\n{'='*60}")
    print(f"Done! Downloaded: {downloaded}, Skipped: {skipped}, Failed: {failed}")

    if failed_list:
        failed_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "failed.json")
        with open(failed_file, "w") as f:
            json.dump(failed_list, f, indent=2)
        print(f"Failed downloads saved to {failed_file}")
        print("Re-run the script to retry failed downloads.")


if __name__ == "__main__":
    main()
