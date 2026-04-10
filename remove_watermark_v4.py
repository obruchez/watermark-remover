#!/usr/bin/env python3
"""Remove KUVA watermark using reference pairs (clean + watermarked) to compute exact mask.

V4: Uses 3 purchased reference photos to extract the watermark mask by comparing
watermarked vs clean versions of the same image. Two-pass approach:
  1. Identify watermark region via averaged differences (noise cancels out)
  2. Estimate precise alpha + color within that region
"""

import os
import time

import numpy as np
from PIL import Image, ImageFilter, ImageOps

# === CONFIGURATION ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHOTOS_DIR = os.path.join(BASE_DIR, "photos")
CLEAN_DIR = os.path.join(PHOTOS_DIR, "without_watermarks")
WATERMARKED_DIR = os.path.join(PHOTOS_DIR, "with_watermarks")
OUTPUT_DIR = os.path.join(BASE_DIR, "photos_v4")
MASK_DIR = os.path.join(BASE_DIR, "watermark_masks_v4")

WEBP_QUALITY = 90

# Reference pairs: (clean filename, watermarked filename)
REFERENCE_PAIRS = [
    ("GAZ70651.jpg", "photo_3983_1202843_GAZ70651.webp"),
    ("GAZ71833.jpg", "photo_4994_1203854_GAZ71833.webp"),
    ("GAZ79064.jpg", "photo_6530_1205390_GAZ79064.webp"),
]


def load_pair(clean_name, watermarked_name):
    """Load a reference pair, downscaling clean image to match watermarked size."""
    clean_path = os.path.join(CLEAN_DIR, clean_name)
    wm_path = os.path.join(WATERMARKED_DIR, watermarked_name)

    clean_img = Image.open(clean_path)
    wm_img = Image.open(wm_path)

    target_size = wm_img.size
    clean_resized = clean_img.resize(target_size, Image.LANCZOS)

    clean_arr = np.array(clean_resized, dtype=np.float64)
    wm_arr = np.array(wm_img, dtype=np.float64)

    print(f"  Loaded pair: {clean_name} ({clean_img.size}) -> {target_size}, "
          f"{watermarked_name} ({wm_img.size})")

    return clean_arr, wm_arr


def estimate_mask_from_pairs():
    """Two-pass watermark estimation from reference pairs.

    Pass 1: Identify watermark region by averaging differences across pairs.
            The watermark signal reinforces while noise cancels out.
    Pass 2: Estimate precise alpha within the identified region using
            the pair-difference method.
    """
    pairs = []
    for clean_name, wm_name in REFERENCE_PAIRS:
        clean_arr, wm_arr = load_pair(clean_name, wm_name)
        pairs.append((clean_arr, wm_arr))

    h, w, c = pairs[0][0].shape
    print(f"\n  Image dimensions: {w}x{h}")

    # === PASS 1: Identify watermark region ===
    print("\n  Pass 1: Identifying watermark region...")

    # For each pair, compute signed difference (watermarked - clean)
    # The watermark adds light pixels, so difference should be positive in watermark areas
    diffs = []
    for O, W in pairs:
        diff = W - O  # positive where watermark adds brightness
        diffs.append(diff)

    # Average differences across pairs - noise cancels, watermark reinforces
    avg_diff = np.mean(np.stack(diffs, axis=0), axis=0)  # (H, W, 3)
    avg_diff_gray = np.mean(avg_diff, axis=2)  # average across RGB channels

    # Also compute the consistency: how much do all 3 pairs agree?
    # If all pairs show positive diff at a pixel, it's likely watermark
    agreement = np.ones((h, w), dtype=bool)
    for diff in diffs:
        diff_gray = np.mean(diff, axis=2)
        agreement &= (diff_gray > 2)  # all pairs show brightness increase > 2

    print(f"    Average diff range: [{avg_diff_gray.min():.1f}, {avg_diff_gray.max():.1f}]")
    print(f"    Pixels where all 3 pairs agree (diff > 2): {np.sum(agreement)}/{h*w} "
          f"({100*np.sum(agreement)/(h*w):.1f}%)")

    # The watermark region: average diff significantly positive AND pairs agree
    # Use a threshold based on the distribution
    positive_diff = avg_diff_gray[avg_diff_gray > 0]
    if len(positive_diff) > 0:
        # Use a threshold that separates watermark from noise
        # The watermark should have much larger diffs than noise
        threshold = max(5.0, np.percentile(positive_diff, 75))
    else:
        threshold = 5.0

    watermark_region = (avg_diff_gray > threshold) & agreement
    print(f"    Threshold: {threshold:.1f}")
    print(f"    Initial watermark region: {np.sum(watermark_region)}/{h*w} "
          f"({100*np.sum(watermark_region)/(h*w):.1f}%)")

    # Morphological cleanup: dilate slightly to fill gaps, then clean up small spots
    from PIL import ImageMorph
    region_img = Image.fromarray(watermark_region.astype(np.uint8) * 255, mode="L")

    # Slight blur + threshold to fill small gaps in the text
    region_img = region_img.filter(ImageFilter.GaussianBlur(radius=3))
    region_arr = np.array(region_img)
    region_arr = (region_arr > 30).astype(np.uint8) * 255
    region_img = Image.fromarray(region_arr, mode="L")

    # Remove small isolated components (noise) using erosion then dilation
    region_img = region_img.filter(ImageFilter.MinFilter(5))  # erode
    region_img = region_img.filter(ImageFilter.MaxFilter(7))  # dilate more
    region_img = region_img.filter(ImageFilter.GaussianBlur(radius=2))  # smooth edges

    watermark_region_cleaned = np.array(region_img).astype(np.float64) / 255.0
    watermark_region_bool = watermark_region_cleaned > 0.1

    print(f"    Cleaned watermark region: {np.sum(watermark_region_bool)}/{h*w} "
          f"({100*np.sum(watermark_region_bool)/(h*w):.1f}%)")

    # === PASS 2: Estimate alpha within watermark region ===
    print("\n  Pass 2: Estimating alpha from pair differences...")

    # Method: (1-α) = (W_i - W_j) / (O_i - O_j)
    pair_indices = [(0, 1), (0, 2), (1, 2)]
    one_minus_alpha_estimates = []

    for i, j in pair_indices:
        O_i, W_i = pairs[i]
        O_j, W_j = pairs[j]

        delta_O = O_i - O_j
        delta_W = W_i - W_j

        one_minus_alpha = np.full((h, w, c), np.nan)
        sufficient_diff = np.abs(delta_O) > 20

        one_minus_alpha[sufficient_diff] = delta_W[sufficient_diff] / delta_O[sufficient_diff]
        one_minus_alpha_estimates.append(one_minus_alpha)

    stacked = np.stack(one_minus_alpha_estimates, axis=0)
    with np.errstate(all='ignore'):
        one_minus_alpha_avg = np.nanmedian(stacked, axis=0)

    alpha_per_channel = 1.0 - one_minus_alpha_avg
    alpha = np.nanmean(alpha_per_channel, axis=2)
    alpha = np.nan_to_num(alpha, nan=0.0)
    alpha = np.clip(alpha, 0, 1)

    # Zero out alpha outside the watermark region
    alpha[~watermark_region_bool] = 0.0

    print(f"    Alpha in watermark region: min={alpha[watermark_region_bool].min():.4f}, "
          f"max={alpha[watermark_region_bool].max():.4f}, "
          f"mean={alpha[watermark_region_bool].mean():.4f}")

    # Also use the direct difference method as a secondary estimate
    # α ≈ avg_diff / (C - O) where C is watermark color
    # For a white watermark (C=255): α ≈ avg_diff / (255 - O)
    # Average across pairs for O
    avg_O = np.mean(np.stack([O for O, W in pairs], axis=0), axis=0)
    avg_W = np.mean(np.stack([W for O, W in pairs], axis=0), axis=0)

    # Estimate watermark color from high-alpha pixels
    # C = (W - (1-α)*O) / α
    alpha_3ch = np.stack([alpha] * 3, axis=2)
    color_estimates = []
    for O, W in pairs:
        safe_alpha = np.where(alpha_3ch > 0.05, alpha_3ch, 1.0)
        C = (W - (1.0 - alpha_3ch) * O) / safe_alpha
        C = np.where(alpha_3ch > 0.05, C, 0)
        color_estimates.append(C)
    color = np.mean(np.stack(color_estimates, axis=0), axis=0)

    # Report watermark color in high-alpha region
    high_alpha_mask = alpha > 0.15
    if np.any(high_alpha_mask):
        for ch, name in enumerate(['R', 'G', 'B']):
            vals = color[:, :, ch][high_alpha_mask]
            vals_clean = vals[(vals > 0) & (vals < 256)]
            if len(vals_clean) > 0:
                print(f"    Watermark color {name}: mean={vals_clean.mean():.1f}, "
                      f"median={np.median(vals_clean):.1f}")

    # Smooth alpha within the watermark region
    alpha_uint8 = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    alpha_img = Image.fromarray(alpha_uint8, mode="L")
    alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=0.8))
    alpha_smoothed = np.array(alpha_img).astype(np.float64) / 255.0

    # Ensure we stay within watermark region (with soft boundary from region blur)
    alpha_smoothed *= watermark_region_cleaned
    alpha_smoothed[alpha_smoothed < 0.01] = 0.0

    # Smooth color map
    color_smoothed = np.zeros_like(color)
    for ch in range(3):
        ch_img = Image.fromarray(np.clip(color[:, :, ch], 0, 255).astype(np.uint8), mode="L")
        ch_img = ch_img.filter(ImageFilter.GaussianBlur(radius=0.8))
        color_smoothed[:, :, ch] = np.array(ch_img).astype(np.float64)

    print(f"\n  Final alpha stats:")
    print(f"    Nonzero pixels: {np.sum(alpha_smoothed > 0.01)}/{h*w} "
          f"({100*np.sum(alpha_smoothed > 0.01)/(h*w):.1f}%)")
    print(f"    Max alpha: {alpha_smoothed.max():.4f}")

    return alpha_smoothed, color_smoothed


def save_mask(alpha, color, mask_dir):
    """Save mask visualizations for inspection."""
    os.makedirs(mask_dir, exist_ok=True)

    vis = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(vis, mode="L").save(os.path.join(mask_dir, "alpha_mask.png"))

    vis_bright = (np.clip(alpha * 3, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(vis_bright, mode="L").save(os.path.join(mask_dir, "alpha_mask_bright.png"))

    color_uint8 = np.clip(color, 0, 255).astype(np.uint8)
    Image.fromarray(color_uint8).save(os.path.join(mask_dir, "watermark_color.png"))

    np.save(os.path.join(mask_dir, "alpha.npy"), alpha)
    np.save(os.path.join(mask_dir, "color.npy"), color)

    print(f"  Masks saved to {mask_dir}/")


def make_portrait_mask(alpha, color):
    """Create portrait mask by placing the watermark pattern centered on a portrait canvas.

    The watermark is always horizontal KUVA text centered in the image,
    regardless of orientation. So for portrait, we extract the pattern
    from the landscape mask and re-center it.
    """
    land_h, land_w = alpha.shape  # landscape: 1440 x 2160
    port_h, port_w = land_w, land_h  # portrait: 2160 x 1440

    # Find bounding box of watermark in landscape mask
    rows = np.any(alpha > 0.01, axis=1)
    cols = np.any(alpha > 0.01, axis=0)
    if not np.any(rows) or not np.any(cols):
        return np.zeros((port_h, port_w)), np.zeros((port_h, port_w, 3))

    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]

    # Extract watermark pattern
    pattern_alpha = alpha[r_min:r_max+1, c_min:c_max+1]
    pattern_color = color[r_min:r_max+1, c_min:c_max+1]
    pat_h, pat_w = pattern_alpha.shape

    # Center pattern on portrait canvas
    port_alpha = np.zeros((port_h, port_w))
    port_color = np.zeros((port_h, port_w, 3))

    r_offset = (port_h - pat_h) // 2
    c_offset = (port_w - pat_w) // 2

    # Compute valid ranges for both source and destination
    dst_r_start = max(r_offset, 0)
    dst_c_start = max(c_offset, 0)
    dst_r_end = min(r_offset + pat_h, port_h)
    dst_c_end = min(c_offset + pat_w, port_w)

    src_r_start = dst_r_start - r_offset
    src_c_start = dst_c_start - c_offset
    src_r_end = src_r_start + (dst_r_end - dst_r_start)
    src_c_end = src_c_start + (dst_c_end - dst_c_start)

    port_alpha[dst_r_start:dst_r_end, dst_c_start:dst_c_end] = \
        pattern_alpha[src_r_start:src_r_end, src_c_start:src_c_end]
    port_color[dst_r_start:dst_r_end, dst_c_start:dst_c_end] = \
        pattern_color[src_r_start:src_r_end, src_c_start:src_c_end]

    return port_alpha, port_color


def remove_watermark(img_arr, alpha, color):
    """Remove watermark from a single image.

    Inverse of: W = (1-α)*O + α*C
    => O = (W - α*C) / (1-α)
    """
    alpha_3ch = np.stack([alpha] * 3, axis=2)

    safe_denom = np.where(alpha_3ch > 0.01, 1.0 - alpha_3ch, 1.0)
    result = (img_arr - alpha_3ch * color) / safe_denom

    no_watermark = alpha < 0.01
    for ch in range(3):
        result[:, :, ch][no_watermark] = img_arr[:, :, ch][no_watermark]

    return np.clip(result, 0, 255).astype(np.uint8)


def process_all_photos(alpha, color, output_dir):
    """Process all watermarked photos."""
    os.makedirs(output_dir, exist_ok=True)

    # Pre-compute portrait mask
    port_alpha, port_color = make_portrait_mask(alpha, color)

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
            mask_h, mask_w = alpha.shape

            if img_w == mask_w and img_h == mask_h:
                a = alpha
                c = color
            elif img_w == mask_h and img_h == mask_w:
                # Portrait: use pre-computed portrait mask (centered, not rotated)
                a = port_alpha
                c = port_color
            else:
                print(f"  WARNING: {fname} has unexpected size {img_w}x{img_h}, skipping")
                failed.append((fname, f"unexpected size {img_w}x{img_h}"))
                continue

            result = remove_watermark(img_arr, a, c)
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


def generate_comparisons(alpha, color, output_dir):
    """Generate side-by-side comparisons."""
    comp_dir = os.path.join(output_dir, "comparisons")
    os.makedirs(comp_dir, exist_ok=True)

    # Reference pair comparisons (3-up: watermarked | cleaned | reference)
    for clean_name, wm_name in REFERENCE_PAIRS:
        wm_path = os.path.join(WATERMARKED_DIR, wm_name)
        wm_img = Image.open(wm_path)
        wm_arr = np.array(wm_img, dtype=np.float64)

        result = remove_watermark(wm_arr, alpha, color)
        result_img = Image.fromarray(result)

        clean_path = os.path.join(CLEAN_DIR, clean_name)
        clean_img = Image.open(clean_path).resize(wm_img.size, Image.LANCZOS)

        w, h = wm_img.size
        comparison = Image.new("RGB", (w * 3, h))
        comparison.paste(wm_img, (0, 0))
        comparison.paste(result_img, (w, 0))
        comparison.paste(clean_img, (w * 2, 0))

        out_name = f"compare_{clean_name.replace('.jpg', '.png')}"
        comparison.save(os.path.join(comp_dir, out_name))
        print(f"  Saved comparison: {out_name}")

    # Pre-compute portrait mask for comparisons
    port_alpha, port_color = make_portrait_mask(alpha, color)

    # Non-reference comparisons (2-up: watermarked | cleaned)
    non_ref_files = sorted([
        f for f in os.listdir(WATERMARKED_DIR)
        if f.endswith('.webp') and f not in [p[1] for p in REFERENCE_PAIRS]
    ])
    for fname in non_ref_files[:6]:
        wm_path = os.path.join(WATERMARKED_DIR, fname)
        wm_img = Image.open(wm_path)
        wm_arr = np.array(wm_img, dtype=np.float64)

        img_h, img_w = wm_arr.shape[:2]
        mask_h, mask_w = alpha.shape
        if img_w == mask_h and img_h == mask_w:
            a = port_alpha
            c = port_color
        else:
            a = alpha
            c = color

        result = remove_watermark(wm_arr, a, c)
        result_img = Image.fromarray(result)

        w, h = wm_img.size
        comparison = Image.new("RGB", (w * 2, h))
        comparison.paste(wm_img, (0, 0))
        comparison.paste(result_img, (w, 0))

        out_name = f"compare_{fname.replace('.webp', '.png')}"
        comparison.save(os.path.join(comp_dir, out_name))
        print(f"  Saved comparison: {out_name}")


def main():
    print("=== V4: Watermark removal using reference pairs ===\n")

    print("Phase 1: Computing watermark mask from reference pairs...")
    alpha, color = estimate_mask_from_pairs()

    print("\nPhase 2: Saving mask visualizations...")
    save_mask(alpha, color, MASK_DIR)

    print("\nPhase 3: Removing watermarks from all photos...")
    failed = process_all_photos(alpha, color, OUTPUT_DIR)

    print("\nPhase 4: Generating comparison images...")
    generate_comparisons(alpha, color, OUTPUT_DIR)

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
