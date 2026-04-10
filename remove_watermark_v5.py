#!/usr/bin/env python3
"""Remove KUVA watermark using hybrid approach: statistical mask + reference pair calibration.

V5: Combines the best of both approaches:
  - V3 statistical mask (from 8000 photos) for watermark SHAPE/REGION
  - 3 purchased reference pairs to calibrate ALPHA VALUES per-pixel
  - Pure white watermark color (C=255) determined by optimization
"""

import os
import time

import numpy as np
from PIL import Image, ImageFilter

# === CONFIGURATION ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHOTOS_DIR = os.path.join(BASE_DIR, "photos")
CLEAN_DIR = os.path.join(PHOTOS_DIR, "without_watermarks")
WATERMARKED_DIR = os.path.join(PHOTOS_DIR, "with_watermarks")
OUTPUT_DIR = os.path.join(BASE_DIR, "photos_v5")
MASK_DIR = os.path.join(BASE_DIR, "watermark_masks_v5")

# V3 statistical masks (computed from 8000 photos)
V3_MASK_DIR = os.path.join(BASE_DIR, "watermark_masks")

WEBP_QUALITY = 90
WATERMARK_COLOR = 255.0  # Pure white, determined by grid search optimization
V3_ALPHA_SCALE = 1.03    # Fallback scale factor for V3 alpha where reference pairs are noisy
AGREEMENT_THRESHOLD = 0.1  # Max std across pairs to trust reference alpha
MIN_DENOM = 15.0          # Minimum (C - O) to compute alpha from a pair

# Reference pairs: (clean filename, watermarked filename)
REFERENCE_PAIRS = [
    ("GAZ70651.jpg", "photo_3983_1202843_GAZ70651.webp"),
    ("GAZ71833.jpg", "photo_4994_1203854_GAZ71833.webp"),
    ("GAZ79064.jpg", "photo_6530_1205390_GAZ79064.webp"),
]


def load_v3_mask():
    """Load the V3 statistical alpha mask (landscape)."""
    path = os.path.join(V3_MASK_DIR, "alpha_2160x1440.npy")
    alpha_3ch = np.load(path)
    alpha = alpha_3ch[:, :, 0]  # All 3 channels are identical
    print(f"  V3 mask: {alpha.shape}, range [{alpha.min():.4f}, {alpha.max():.4f}], "
          f"nonzero: {np.sum(alpha > 0.01)}")
    return alpha


def load_reference_pairs():
    """Load the 3 reference pairs, downscaling clean to match watermarked."""
    pairs = []
    for clean_name, wm_name in REFERENCE_PAIRS:
        clean_path = os.path.join(CLEAN_DIR, clean_name)
        wm_path = os.path.join(WATERMARKED_DIR, wm_name)

        clean_img = Image.open(clean_path)
        wm_img = Image.open(wm_path)
        target_size = wm_img.size

        clean_resized = clean_img.resize(target_size, Image.LANCZOS)
        O = np.array(clean_resized, dtype=np.float64)
        W = np.array(wm_img, dtype=np.float64)

        pairs.append((O, W))
        print(f"  Loaded: {clean_name} ({clean_img.size}) -> {target_size}")

    return pairs


def calibrate_alpha(alpha_v3, pairs):
    """Calibrate V3 alpha per-pixel using reference pairs.

    For each pixel in the watermark region:
    - Compute alpha from each pair: α = (W - O) / (C - O) with C = 255
    - Where all 3 pairs agree (low std), use the mean reference alpha
    - Elsewhere, fall back to V3 alpha * scale factor
    """
    C = WATERMARK_COLOR
    mask = alpha_v3 > 0.01

    # Per-pixel alpha from each reference pair
    alpha_per_pair = []
    for O, W in pairs:
        W_gray = np.mean(W, axis=2)
        O_gray = np.mean(O, axis=2)
        denom = C - O_gray
        alpha_pair = np.where(denom > MIN_DENOM,
                              (W_gray - O_gray) / denom, np.nan)
        alpha_per_pair.append(alpha_pair)

    stacked = np.stack(alpha_per_pair, axis=0)  # (3, H, W)
    alpha_mean = np.nanmean(stacked, axis=0)
    alpha_std = np.nanstd(stacked, axis=0)

    # Where all pairs agree, trust the reference alpha
    agreement = alpha_std < AGREEMENT_THRESHOLD
    valid_ref = agreement & mask & (alpha_mean > 0.01) & (alpha_mean < 0.95)

    # Build hybrid alpha
    alpha_hybrid = np.zeros_like(alpha_v3)
    alpha_hybrid[mask] = alpha_v3[mask] * V3_ALPHA_SCALE
    alpha_hybrid[valid_ref] = alpha_mean[valid_ref]
    alpha_hybrid = np.clip(alpha_hybrid, 0, 0.95)

    # Zero outside mask
    alpha_hybrid[~mask] = 0.0

    ref_count = np.sum(valid_ref)
    fallback_count = np.sum(mask) - ref_count
    print(f"  Reference-calibrated pixels: {ref_count} ({100*ref_count/np.sum(mask):.1f}%)")
    print(f"  V3 fallback pixels: {fallback_count} ({100*fallback_count/np.sum(mask):.1f}%)")
    print(f"  Hybrid alpha range: [{alpha_hybrid[mask].min():.4f}, {alpha_hybrid[mask].max():.4f}]")

    return alpha_hybrid


def load_portrait_mask():
    """Load the V3 statistical portrait mask and apply same calibration as landscape.

    The portrait watermark has a different layout than landscape (smaller, differently
    positioned), so we use the V3 statistical mask computed from actual portrait photos
    rather than deriving from the landscape mask.
    """
    port_path = os.path.join(V3_MASK_DIR, "alpha_1440x2160.npy")
    alpha_v3p = np.load(port_path)[:, :, 0]  # (2160, 1440), single channel

    # The V3 portrait mask has noise at low alpha (20% > 0.01 but only 4.3% > 0.15).
    # Threshold to keep only the real watermark signal.
    alpha_v3p[alpha_v3p < 0.10] = 0.0

    # Apply same scale factor as landscape calibration
    mask = alpha_v3p > 0.0
    alpha_v3p[mask] *= V3_ALPHA_SCALE
    alpha_v3p = np.clip(alpha_v3p, 0, 0.95)

    print(f"  V3 portrait mask: {alpha_v3p.shape}, nonzero: {np.sum(mask)}")
    print(f"  Alpha range: [{alpha_v3p[mask].min():.4f}, {alpha_v3p[mask].max():.4f}]")
    return alpha_v3p


def remove_watermark(img_arr, alpha):
    """Remove watermark: O = (W - α*C) / (1-α) with C = 255."""
    C = WATERMARK_COLOR
    alpha_3ch = np.stack([np.clip(alpha, 0, 0.95)] * 3, axis=2)

    result = (img_arr - alpha_3ch * C) / (1.0 - alpha_3ch)

    # Keep original pixels where no watermark
    no_wm = alpha < 0.01
    for ch in range(3):
        result[:, :, ch][no_wm] = img_arr[:, :, ch][no_wm]

    return np.clip(result, 0, 255).astype(np.uint8)


def save_masks(alpha_land, alpha_port, mask_dir):
    """Save mask visualizations."""
    os.makedirs(mask_dir, exist_ok=True)

    for label, alpha in [("landscape", alpha_land), ("portrait", alpha_port)]:
        vis = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(vis, mode="L").save(
            os.path.join(mask_dir, f"alpha_{label}.png"))

        vis_bright = (np.clip(alpha * 3, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(vis_bright, mode="L").save(
            os.path.join(mask_dir, f"alpha_{label}_bright.png"))

        np.save(os.path.join(mask_dir, f"alpha_{label}.npy"), alpha)
        print(f"  Saved {label} mask to {mask_dir}/")


def process_all_photos(alpha_land, alpha_port, output_dir):
    """Process all watermarked photos."""
    os.makedirs(output_dir, exist_ok=True)

    input_files = sorted([
        f for f in os.listdir(WATERMARKED_DIR) if f.endswith('.webp')
    ])

    total = len(input_files)
    print(f"  Processing {total} photos...")

    failed = []
    t0 = time.time()

    for i, fname in enumerate(input_files):
        input_path = os.path.join(WATERMARKED_DIR, fname)
        output_path = os.path.join(output_dir, fname)

        try:
            img = Image.open(input_path)
            img_arr = np.array(img, dtype=np.float64)
            img_h, img_w = img_arr.shape[:2]

            mask_h, mask_w = alpha_land.shape
            if img_w == mask_w and img_h == mask_h:
                alpha = alpha_land
            elif img_w == mask_h and img_h == mask_w:
                alpha = alpha_port
            else:
                print(f"  WARNING: {fname} size {img_w}x{img_h} unexpected, skipping")
                failed.append((fname, f"unexpected size {img_w}x{img_h}"))
                continue

            result = remove_watermark(img_arr, alpha)
            Image.fromarray(result).save(output_path, "WEBP", quality=WEBP_QUALITY)

        except Exception as e:
            failed.append((fname, str(e)))
            print(f"  FAILED: {fname} - {e}")

        if (i + 1) % 5 == 0 or i == total - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{total}] {rate:.1f} img/s", end="\r")

    print()
    return failed


def generate_comparisons(alpha_land, alpha_port, output_dir):
    """Generate comparison images."""
    comp_dir = os.path.join(output_dir, "comparisons")
    os.makedirs(comp_dir, exist_ok=True)

    # Reference pairs: 3-panel (watermarked | v5 removed | clean reference)
    for clean_name, wm_name in REFERENCE_PAIRS:
        wm_path = os.path.join(WATERMARKED_DIR, wm_name)
        wm_img = Image.open(wm_path)
        wm_arr = np.array(wm_img, dtype=np.float64)

        result = remove_watermark(wm_arr, alpha_land)
        result_img = Image.fromarray(result)

        clean_img = Image.open(os.path.join(CLEAN_DIR, clean_name))
        clean_img = clean_img.resize(wm_img.size, Image.LANCZOS)

        w, h = wm_img.size
        comp = Image.new("RGB", (w * 3, h))
        comp.paste(wm_img, (0, 0))
        comp.paste(result_img, (w, 0))
        comp.paste(clean_img, (w * 2, 0))

        out_name = f"ref_{clean_name.replace('.jpg', '.png')}"
        comp.save(os.path.join(comp_dir, out_name))
        print(f"  {out_name}")

    # Non-reference photos: 2-panel (watermarked | v5 removed)
    ref_wm_names = {p[1] for p in REFERENCE_PAIRS}
    non_ref = sorted([
        f for f in os.listdir(WATERMARKED_DIR)
        if f.endswith('.webp') and f not in ref_wm_names
    ])

    for fname in non_ref[:8]:
        wm_path = os.path.join(WATERMARKED_DIR, fname)
        wm_img = Image.open(wm_path)
        wm_arr = np.array(wm_img, dtype=np.float64)

        img_h, img_w = wm_arr.shape[:2]
        mask_h, mask_w = alpha_land.shape
        if img_w == mask_h and img_h == mask_w:
            alpha = alpha_port
        else:
            alpha = alpha_land

        result = remove_watermark(wm_arr, alpha)
        result_img = Image.fromarray(result)

        w, h = wm_img.size
        comp = Image.new("RGB", (w * 2, h))
        comp.paste(wm_img, (0, 0))
        comp.paste(result_img, (w, 0))

        out_name = f"other_{fname.replace('.webp', '.png')}"
        comp.save(os.path.join(comp_dir, out_name))
        print(f"  {out_name}")


def main():
    print("=== V5: Hybrid watermark removal (statistical mask + reference calibration) ===\n")

    print("Phase 1: Loading V3 statistical mask...")
    alpha_v3 = load_v3_mask()

    print("\nPhase 2: Loading reference pairs...")
    pairs = load_reference_pairs()

    print("\nPhase 3: Calibrating alpha with reference pairs...")
    alpha_land = calibrate_alpha(alpha_v3, pairs)

    print("\nPhase 4: Loading portrait mask...")
    alpha_port = load_portrait_mask()

    print("\nPhase 5: Saving masks...")
    save_masks(alpha_land, alpha_port, MASK_DIR)

    print("\nPhase 6: Removing watermarks...")
    failed = process_all_photos(alpha_land, alpha_port, OUTPUT_DIR)

    print("\nPhase 7: Generating comparisons...")
    generate_comparisons(alpha_land, alpha_port, OUTPUT_DIR)

    print("\n=== Done! ===")
    if failed:
        print(f"  {len(failed)} photos failed:")
        for f, err in failed:
            print(f"    {f}: {err}")
    else:
        print("  All photos processed successfully")
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"  Comparisons: {OUTPUT_DIR}/comparisons/")


if __name__ == "__main__":
    main()
