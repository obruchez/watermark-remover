#!/usr/bin/env python3
"""Improved KUVA watermark removal using robust statistical estimation."""

import glob
import multiprocessing
import os
import random
import sys
import time

import numpy as np
from PIL import Image

# Configuration
INPUT_DIR = "photos"
OUTPUT_DIR = "photos_clean"
MASK_DIR = "watermark_masks"
VERIFY_DIR = "watermark_verify"

SAMPLE_COUNT = 1000         # More samples for better statistical recovery
PERCENTILE = 10             # 10th percentile is more robust than absolute minimum
ALPHA_MAX = 0.95            # Guard for division
NUM_WORKERS = 12

def classify_photos(input_dir):
    files = sorted(glob.glob(os.path.join(input_dir, "*.webp")))
    groups = {}
    for f in files:
        with Image.open(f) as img:
            size = img.size
            groups.setdefault(size, []).append(f)
    return groups

def estimate_watermark(photos, size, sample_count=SAMPLE_COUNT):
    """Estimate the fixed watermark signal using percentiles."""
    n = min(len(photos), sample_count)
    samples = random.sample(photos, n)
    print(f"  Estimating watermark from {n} photos for {size}...")
    
    # We process in chunks to avoid memory issues
    chunk_size = 100
    all_chunks_p10 = []
    
    for i in range(0, n, chunk_size):
        chunk_paths = samples[i:i+chunk_size]
        chunk_data = []
        for p in chunk_paths:
            img = np.array(Image.open(p).convert('RGB')).astype(np.float32) / 255.0
            chunk_data.append(img)
        
        # Intermediate percentile for this chunk
        chunk_p10 = np.percentile(np.array(chunk_data), PERCENTILE, axis=0)
        all_chunks_p10.append(chunk_p10)
        print(f"    Processed {min(i+chunk_size, n)}/{n}...")

    # Final estimate is the median of chunk percentiles (very robust)
    watermark_estimate = np.median(np.array(all_chunks_p10), axis=0)
    return watermark_estimate

def refine_watermark(est):
    """Derive alpha and color from the raw estimate."""
    # Watermark is whitish. Estimate its color from the brightest pixels in the estimate.
    # We look for pixels that are likely fully part of the watermark.
    gray = np.mean(est, axis=2)
    # Use a high percentile of the estimate to find the "core" watermark color
    threshold = np.percentile(gray, 99.9)
    core_mask = gray >= threshold
    
    if np.any(core_mask):
        wcolor = np.mean(est[core_mask], axis=0)
        # Normalize so max channel is 1.0 (watermark is usually white or gray)
        max_c = np.max(wcolor)
        if max_c > 0:
            wcolor = wcolor / max_c
    else:
        wcolor = np.array([1.0, 1.0, 1.0])
        
    print(f"    Estimated Watermark Color: {wcolor}")
    
    # Compute alpha: raw_est = alpha * wcolor
    # We use the channel that gives the most signal
    alpha = np.zeros_like(gray)
    for c in range(3):
        if wcolor[c] > 0.01:
            alpha = np.maximum(alpha, est[:, :, c] / wcolor[c])
            
    # CRITICAL: No blurring, no thresholding. 
    # Just a tiny bit of noise floor removal.
    alpha[alpha < 0.005] = 0
    alpha = np.clip(alpha, 0, ALPHA_MAX)
    
    return alpha, wcolor

def process_one(args):
    input_path, output_path, alpha, wcolor = args
    try:
        img = np.array(Image.open(input_path).convert('RGB')).astype(np.float32) / 255.0
        
        # alpha is (H, W), wcolor is (3,)
        # Reverse blending: C_out = (C_src - alpha * wcolor) / (1 - alpha)
        
        # Expand alpha for broadcasting
        a = alpha[:, :, np.newaxis]
        
        # We only apply the math where alpha > 0
        mask = alpha > 0
        result = img.copy()
        
        # The core formula
        denom = 1.0 - a
        # Subtraction
        result[mask] = (img[mask] - a[mask] * wcolor) / denom[mask]
        
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        
        # Save as original format (WEBP)
        Image.fromarray(result).save(output_path, "WEBP", quality=95)
        return True
    except Exception as e:
        print(f"Error {input_path}: {e}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0, help="Process only N photos per group")
    args_cli = parser.parse_args()

    groups = classify_photos(INPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)
    os.makedirs(VERIFY_DIR, exist_ok=True)
    
    for size, photos in groups.items():
        label = f"{size[0]}x{size[1]}"
        print(f"\nProcessing {label}...")
        
        est = estimate_watermark(photos, size)
        alpha, wcolor = refine_watermark(est)
        
        # Save mask for inspection
        vis = (alpha * 255).astype(np.uint8)
        Image.fromarray(vis).save(os.path.join(MASK_DIR, f"v3_alpha_{label}.png"))
        
        work = []
        target_photos = photos
        if args_cli.sample > 0:
            target_photos = random.sample(photos, min(args_cli.sample, len(photos)))
            
        for p in target_photos:
            out = os.path.join(OUTPUT_DIR, os.path.basename(p))
            work.append((p, out, alpha, wcolor))
            
        print(f"  Removing watermark from {len(work)} photos...")
        with multiprocessing.Pool(NUM_WORKERS) as pool:
            results = pool.map(process_one, work)
            
        print(f"  Done {sum(results)}/{len(work)}")

    # Create a few verifications
    print("\nGenerating verifications...")
    for size in groups:
        label = f"{size[0]}x{size[1]}"
        samples = random.sample(groups[size], min(5, len(groups[size])))
        for i, p in enumerate(samples):
            fname = os.path.basename(p)
            clean_p = os.path.join(OUTPUT_DIR, fname)
            if os.path.exists(clean_p):
                img1 = Image.open(p)
                img2 = Image.open(clean_p)
                combined = Image.new('RGB', (img1.width*2, img1.height))
                combined.paste(img1, (0,0))
                combined.paste(img2, (img1.width, 0))
                combined.save(os.path.join(VERIFY_DIR, f"v3_verify_{label}_{i}.jpg"), "JPEG", quality=85)

if __name__ == "__main__":
    main()
