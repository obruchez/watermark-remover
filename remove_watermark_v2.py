#!/usr/bin/env python3
"""Improved KUVA watermark removal using statistical estimation and OpenCV inpainting."""

import glob
import multiprocessing
import os
import random
import sys
import time

import numpy as np
import cv2
from PIL import Image

# Configuration
INPUT_DIR = "photos"
OUTPUT_DIR = "photos_clean"
MASK_DIR = "watermark_masks"
VERIFY_DIR = "watermark_verify"

SAMPLE_SIZE = 500       # Number of photos to estimate the watermark
ALPHA_THRESHOLD = 0.02  # Threshold for considering a pixel part of the watermark
ALPHA_MAX = 0.8         # Cap for alpha (safeguard)
DILATION_RADIUS = 2     # Dilate the mask to cover edges
INPAINT_RADIUS = 3      # Radius for OpenCV inpainting

def classify_photos(input_dir):
    """Group photos by dimensions."""
    photos = glob.glob(os.path.join(input_dir, "*.webp"))
    photos += glob.glob(os.path.join(input_dir, "*.jpg"))
    photos += glob.glob(os.path.join(input_dir, "*.jpeg"))
    
    groups = {}
    for p in photos:
        try:
            with Image.open(p) as img:
                size = img.size
                if size not in groups:
                    groups[size] = []
                groups[size].append(p)
        except Exception as e:
            print(f"Error opening {p}: {e}")
            
    return groups

def estimate_raw_watermark(photos, size, sample_size=SAMPLE_SIZE):
    """Estimate the watermark by taking the minimum across a sample of photos."""
    sample = random.sample(photos, min(len(photos), sample_size))
    print(f"  Estimating from {len(sample)} photos of size {size}...")
    
    stack = None
    for i, p in enumerate(sample):
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if (img.shape[1], img.shape[0]) != size:
            continue
            
        arr = img.astype(np.float64) / 255.0
        if stack is None:
            stack = arr
        else:
            stack = np.minimum(stack, arr)
            
        if (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{len(sample)}...")
            
    return stack

def refine_mask(raw_watermark):
    """Refine the raw watermark estimate into an alpha mask and color."""
    # The watermark is whitish. Estimate color from the brightest pixels.
    avg_per_pixel = np.mean(raw_watermark, axis=2)
    mask_indices = avg_per_pixel > 0.05
    if not np.any(mask_indices):
        return np.zeros_like(avg_per_pixel), np.array([1.0, 1.0, 1.0])
    
    # Estimate color as the 99th percentile of mask pixels
    mask_pixels = raw_watermark[mask_indices]
    wcolor = np.percentile(mask_pixels, 99, axis=0)
    print(f"  Estimated watermark color: {wcolor}")
    
    # Compute alpha = raw / wcolor
    alpha_per_ch = np.zeros_like(raw_watermark)
    for c in range(3):
        if wcolor[c] > 0.01:
            alpha_per_ch[:, :, c] = raw_watermark[:, :, c] / wcolor[c]
        else:
            alpha_per_ch[:, :, c] = raw_watermark[:, :, c]
    
    alpha_gray = np.mean(alpha_per_ch, axis=2)
    alpha_gray = np.clip(alpha_gray, 0, ALPHA_MAX)
    alpha_gray[alpha_gray < ALPHA_THRESHOLD] = 0
    
    return alpha_gray, wcolor

def process_one(photo_path, alpha_gray, wcolor, output_dir, verify_dir=None):
    """Process a single photo: remove watermark using subtraction + inpainting."""
    try:
        img_bgr = cv2.imread(photo_path)
        if img_bgr is None:
            return False
        
        # Ensure alpha_gray is 2D
        if len(alpha_gray.shape) == 3:
            alpha_gray_2d = np.mean(alpha_gray, axis=2)
        else:
            alpha_gray_2d = alpha_gray

        # Binary mask for inpainting
        mask = (alpha_gray_2d > 0.01).astype(np.uint8) * 255
        
        # Dilate the mask to cover the borders of letters
        if DILATION_RADIUS > 0:
            kernel = np.ones((DILATION_RADIUS*2+1, DILATION_RADIUS*2+1), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            
        # First, try subtraction for low-alpha areas (preserves some background)
        alpha_sub = np.clip(alpha_gray_2d, 0, ALPHA_MAX)
        wc_bgr = (wcolor[::-1] * 255.0).reshape(1, 1, 3)
        
        # Reverse alpha blending
        img_float = img_bgr.astype(np.float64)
        denom = 1.0 - alpha_sub[:, :, np.newaxis]
        denom = np.maximum(denom, 0.01)
        
        result_sub = (img_float - wc_bgr * alpha_sub[:, :, np.newaxis]) / denom
        result_sub = np.clip(result_sub, 0, 255).astype(np.uint8)
        
        # Now use inpainting for the main body of the watermark
        result = cv2.inpaint(result_sub, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
        
        filename = os.path.basename(photo_path)
        out_name = os.path.splitext(filename)[0] + ".jpg"
        cv2.imwrite(os.path.join(output_dir, out_name), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if verify_dir and random.random() < 0.01:
            combined = np.hstack((img_bgr, result))
            cv2.imwrite(os.path.join(verify_dir, f"compare_{filename}.jpg"), combined)
            
        return True
    except Exception as e:
        print(f"Error processing {photo_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)
    os.makedirs(VERIFY_DIR, exist_ok=True)
    
    print("=== KUVA Watermark Remover v2 ===")
    
    groups = classify_photos(INPUT_DIR)
    print(f"Found {sum(len(v) for v in groups.values())} photos in {len(groups)} orientations.")
    
    for size, photos in groups.items():
        w, h = size
        print(f"\nProcessing group {w}x{h} ({len(photos)} photos)...")
        
        mask_path = os.path.join(MASK_DIR, f"alpha_{w}x{h}.npy")
        color_path = os.path.join(MASK_DIR, f"color_{w}x{h}.npy")
        
        if os.path.exists(mask_path) and os.path.exists(color_path):
            print(f"  Loading existing mask and color for {w}x{h}...")
            alpha_gray = np.load(mask_path)
            wcolor = np.load(color_path)
        else:
            raw_watermark = estimate_raw_watermark(photos, size)
            if raw_watermark is None:
                print(f"  Failed to estimate watermark for {w}x{h}")
                continue
            alpha_gray, wcolor = refine_mask(raw_watermark)
            np.save(mask_path, alpha_gray)
            np.save(color_path, wcolor)
            # Save visual mask
            cv2.imwrite(os.path.join(MASK_DIR, f"alpha_{w}x{h}.png"), (alpha_gray * 255).astype(np.uint8))
            
        print(f"  Removing watermark from {len(photos)} photos...")
        
        # Use multiprocessing
        args = [(p, alpha_gray, wcolor, OUTPUT_DIR, VERIFY_DIR) for p in photos]
        
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.starmap(process_one, args)
            
        success_count = sum(results)
        print(f"  Successfully processed {success_count}/{len(photos)} photos.")

    print("\n=== Done! ===")

if __name__ == "__main__":
    main()
