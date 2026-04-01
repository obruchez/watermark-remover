#!/usr/bin/env python3
"""Remove KUVA watermark from event photos using statistical estimation."""

import glob
import multiprocessing
import os
import random
import sys
import time

import numpy as np
from PIL import Image, ImageFilter

# === CONFIGURATION ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "photos")
OUTPUT_DIR = os.path.join(BASE_DIR, "photos_clean")
MASK_DIR = os.path.join(BASE_DIR, "watermark_masks")
VERIFY_DIR = os.path.join(BASE_DIR, "watermark_verify")

SAMPLE_COUNT = 500          # photos to sample per orientation for mask estimation
NUM_WORKERS = 10            # parallel workers for removal
WEBP_QUALITY = 90           # output quality
ALPHA_THRESHOLD = 0.02      # below this, alpha is set to 0
ALPHA_MAX = 0.95            # clamp alpha for numerical stability
BLUR_RADIUS = 1.5           # Gaussian blur on alpha mask


def classify_photos(input_dir):
    """Group photos by (width, height)."""
    files = sorted(glob.glob(os.path.join(input_dir, "*.webp")))
    groups = {}
    for f in files:
        img = Image.open(f)
        size = img.size  # (width, height)
        img.close()
        groups.setdefault(size, []).append(f)
    return groups


def estimate_alpha_mask(photo_paths, sample_count, label):
    """Estimate watermark alpha by computing per-pixel minimum across all photos."""
    if sample_count <= 0:
        n = len(photo_paths)
        samples = photo_paths
    else:
        n = min(sample_count, len(photo_paths))
        samples = random.sample(photo_paths, n)

    print(f"  [{label}] Estimating alpha from {n} samples...")
    running_min = None

    for i, path in enumerate(samples):
        img = np.array(Image.open(path), dtype=np.uint8)
        if running_min is None:
            running_min = img.copy()
        else:
            np.minimum(running_min, img, out=running_min)
        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"    Processed {i+1}/{n}", end="\r")

    print()

    raw_alpha = running_min.astype(np.float64) / 255.0
    return raw_alpha


def refine_alpha_mask(raw_alpha):
    """Estimate watermark color from per-channel minimums, then derive true alpha.

    raw_alpha has shape (H, W, 3) where raw_alpha_c ≈ (Wc/255) * alpha.
    At peak watermark pixels, the ratio R:G:B reveals the watermark color.
    """
    # Step 1: Find peak watermark region (where combined signal is strongest)
    gray = np.mean(raw_alpha, axis=2)
    nonzero = gray[gray > 0.05]
    if len(nonzero) > 0:
        peak_threshold = np.percentile(nonzero, 90)
    else:
        peak_threshold = 0.3
    peak_mask = gray >= peak_threshold

    # Step 2: Estimate watermark color from peak pixel ratios
    # At peak: raw_c ≈ (Wc/255) * alpha_peak, so ratios give Wr:Wg:Wb
    peak_r = np.mean(raw_alpha[peak_mask, 0])
    peak_g = np.mean(raw_alpha[peak_mask, 1])
    peak_b = np.mean(raw_alpha[peak_mask, 2])
    max_peak = max(peak_r, peak_g, peak_b)

    if max_peak > 0:
        # Watermark color normalized so brightest channel = 255
        wcolor = np.array([peak_r / max_peak, peak_g / max_peak, peak_b / max_peak])
    else:
        wcolor = np.array([1.0, 1.0, 1.0])

    print(f"    Watermark color: RGB({wcolor[0]*255:.0f}, {wcolor[1]*255:.0f}, {wcolor[2]*255:.0f})")

    # Step 3: Compute true alpha per channel, then average
    # raw_alpha_c = (Wc/255) * alpha → alpha = raw_alpha_c / (Wc/255)
    alpha_per_ch = np.zeros_like(raw_alpha)
    for c in range(3):
        if wcolor[c] > 0.01:
            alpha_per_ch[:, :, c] = raw_alpha[:, :, c] / wcolor[c]
        else:
            alpha_per_ch[:, :, c] = raw_alpha[:, :, c]
    alpha_gray = np.mean(alpha_per_ch, axis=2)

    # Step 4: Threshold, then blur (same as v1)
    alpha_gray[alpha_gray < ALPHA_THRESHOLD] = 0.0

    alpha_img = Image.fromarray((np.clip(alpha_gray, 0, 1) * 255).astype(np.uint8), mode="L")
    alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
    alpha_gray = np.array(alpha_img).astype(np.float64) / 255.0

    alpha_gray[alpha_gray < ALPHA_THRESHOLD] = 0.0

    # Return single-channel alpha (expanded to 3) + watermark color
    alpha = np.stack([alpha_gray] * 3, axis=2)
    return alpha, wcolor


def save_mask(alpha, mask_dir, label):
    """Save alpha mask as PNG for inspection."""
    os.makedirs(mask_dir, exist_ok=True)
    # Save grayscale visualization (alpha averaged across channels)
    alpha_gray = alpha[:, :, 0]
    # Scale to full 0-255 range for visibility
    vis = (alpha_gray * 255).astype(np.uint8)
    Image.fromarray(vis, mode="L").save(
        os.path.join(mask_dir, f"alpha_{label}.png")
    )
    # Save raw numpy array for potential reuse
    np.save(os.path.join(mask_dir, f"alpha_{label}.npy"), alpha)
    print(f"  [{label}] Mask saved to {mask_dir}/alpha_{label}.png")


# Global state for multiprocessing workers
_masks = {}
_wcolors = {}


def worker_init(masks_dict, wcolors_dict):
    global _masks, _wcolors
    _masks = masks_dict
    _wcolors = wcolors_dict


def process_one(args):
    input_path, output_path, size_key = args
    try:
        img = np.array(Image.open(input_path)).astype(np.float64)
        alpha = _masks[size_key]
        wcolor = _wcolors[size_key]  # (3,) array in [0, 1]

        alpha_safe = np.clip(alpha, 0, ALPHA_MAX)
        # Use estimated watermark color: Wc = wcolor * 255
        wc = wcolor * 255.0  # shape (3,), broadcasts to (1, 1, 3)
        result = (img - wc * alpha_safe) / (1.0 - alpha_safe)

        # Where alpha is 0, keep original pixel exactly (avoid float rounding)
        no_watermark = alpha[:, :, 0] == 0
        result[no_watermark] = img[no_watermark]

        result = np.clip(result, 0, 255).astype(np.uint8)
        Image.fromarray(result).save(output_path, "WEBP", quality=WEBP_QUALITY)
        return None
    except Exception as e:
        return (input_path, str(e))


def remove_watermarks(groups, masks, wcolors, output_dir, num_workers):
    """Remove watermark from all photos using multiprocessing."""
    os.makedirs(output_dir, exist_ok=True)

    # Build work list, skipping already-processed files
    work = []
    skipped = 0
    for size_key, paths in groups.items():
        for p in paths:
            fname = os.path.basename(p)
            out_path = os.path.join(output_dir, fname)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
                skipped += 1
                continue
            work.append((p, out_path, size_key))

    total = len(work)
    if skipped:
        print(f"  Skipped {skipped} already processed photos")
    if total == 0:
        print("  All photos already processed!")
        return []

    print(f"  Processing {total} photos with {num_workers} workers...")

    failed = []
    done = 0
    t0 = time.time()

    with multiprocessing.Pool(num_workers, initializer=worker_init,
                              initargs=(masks, wcolors)) as pool:
        for result in pool.imap_unordered(process_one, work, chunksize=10):
            done += 1
            if result is not None:
                failed.append(result)
                print(f"  FAILED: {result[0]} — {result[1]}")
            if done % 100 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  [{done}/{total}] {rate:.1f} img/s, ETA {eta:.0f}s", end="\r")

    print()
    return failed


def generate_verification(groups, output_dir, verify_dir, count=20):
    """Create side-by-side before/after comparison images."""
    os.makedirs(verify_dir, exist_ok=True)

    # Pick random samples from each orientation
    all_files = []
    for size_key, paths in groups.items():
        n = min(count // len(groups) + 1, len(paths))
        all_files.extend(random.sample(paths, n))
    random.shuffle(all_files)
    all_files = all_files[:count]

    for i, input_path in enumerate(all_files):
        fname = os.path.basename(input_path)
        output_path = os.path.join(output_dir, fname)
        if not os.path.exists(output_path):
            continue

        original = Image.open(input_path)
        cleaned = Image.open(output_path)

        w, h = original.size
        comparison = Image.new("RGB", (w * 2, h))
        comparison.paste(original, (0, 0))
        comparison.paste(cleaned, (w, 0))
        comparison.save(
            os.path.join(verify_dir, f"compare_{i+1:02d}_{fname.replace('.webp', '.jpg')}"),
            "JPEG", quality=85
        )
        original.close()
        cleaned.close()

    print(f"  Saved {len(all_files)} comparison images to {verify_dir}/")


def main():
    random.seed(42)  # reproducible sampling

    print("=== Phase 0: Classifying photos by orientation ===")
    groups = classify_photos(INPUT_DIR)
    for size_key, paths in sorted(groups.items()):
        print(f"  {size_key[0]}x{size_key[1]}: {len(paths)} photos")

    print("\n=== Phase 1: Estimating watermark alpha masks ===")
    masks = {}
    wcolors = {}
    for size_key, paths in sorted(groups.items()):
        label = f"{size_key[0]}x{size_key[1]}"
        raw_alpha = estimate_alpha_mask(paths, SAMPLE_COUNT, label)
        alpha, wcolor = refine_alpha_mask(raw_alpha)
        save_mask(alpha, MASK_DIR, label)
        masks[size_key] = alpha
        wcolors[size_key] = wcolor

        # Print stats
        nonzero = np.count_nonzero(alpha[:, :, 0])
        total_px = alpha.shape[0] * alpha.shape[1]
        max_alpha = alpha.max()
        print(f"  [{label}] Watermark covers {nonzero}/{total_px} pixels "
              f"({100*nonzero/total_px:.1f}%), max alpha={max_alpha:.3f}")

    print("\n=== Phase 2: Removing watermarks ===")
    failed = remove_watermarks(groups, masks, wcolors, OUTPUT_DIR, NUM_WORKERS)

    print("\n=== Phase 3: Generating verification images ===")
    generate_verification(groups, OUTPUT_DIR, VERIFY_DIR)

    print("\n=== Done! ===")
    if failed:
        print(f"  {len(failed)} photos failed — check output above")
    else:
        print("  All photos processed successfully")


if __name__ == "__main__":
    main()
