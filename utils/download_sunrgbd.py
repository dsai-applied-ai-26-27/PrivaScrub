# utils/download_sunrgbd.py
"""
run "python -m utils.download_sunrgbd" to download the SUN RGB-D dataset 
from Hugging Face and save it locally in a structured format.
"""
import os
import numpy as np
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

# 1. Set your target output directory
OUTPUT_DIR = os.path.join("..", "data", "raw",  "SUNRGBD_local")
# create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
REMAP_TO_STRUCTSCAN = True  # Set False to keep all 37 original SUN RGB-D classes

# Lookup table: 0=background, 1=wall, 2=floor, 3=ceiling, 4=window, 5=door
# Original SUN RGB-D 37-class IDs: 1=wall, 2=floor, 8=door, 9=window, 22=ceiling
REMAP_LUT = np.zeros(256, dtype=np.uint8)
REMAP_LUT[1] = 1   # wall
REMAP_LUT[2] = 2   # floor
REMAP_LUT[22] = 3  # ceiling
REMAP_LUT[9] = 4   # window
REMAP_LUT[8] = 5   # door

def export_dataset():
    print("Downloading SUN RGB-D from Hugging Face...")
    # 'default' preserves raw 16-bit depth maps
    dataset = load_dataset("wyrx/SUNRGBD_seg", name="default")

    for split in ["train", "test"]:
        split_data = dataset[split]
        
        # Create structured subfolders
        rgb_dir = os.path.join(OUTPUT_DIR, split, "rgb")
        depth_dir = os.path.join(OUTPUT_DIR, split, "depth")
        mask_dir = os.path.join(OUTPUT_DIR, split, "mask")
        
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        print(f"Saving {len(split_data)} {split} samples to {OUTPUT_DIR}/{split}...")
        for idx, item in enumerate(tqdm(split_data)):
            filename = f"{idx:05d}"

            # 1. Save RGB image
            rgb_img = item["image"].convert("RGB")
            rgb_img.save(os.path.join(rgb_dir, f"{filename}.jpg"), quality=95)

            # 2. Save 16-bit Depth map (stored as uint16 PNG)
            depth_arr = np.array(item["depth"], dtype=np.uint16)
            Image.fromarray(depth_arr).save(os.path.join(depth_dir, f"{filename}.png"))

            # 3. Save Semantic Mask (stored as 8-bit indexed PNG)
            mask_arr = np.array(item["label"], dtype=np.uint8)
            if REMAP_TO_STRUCTSCAN:
                mask_arr = REMAP_LUT[mask_arr]
            Image.fromarray(mask_arr).save(os.path.join(mask_dir, f"{filename}.png"))

    print(f"\nDone! Dataset saved locally at: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    export_dataset()