"""
Data loader + split logic for the blood cell project

what this file does:
- reads labels_simplified.csv (Image id -> Category)
- maps each image id to an actual file on disk (supports .jpg and .jpeg)
- drops rows with missing labels or missing image files (no drama)
- bins ultra-rare classes (<4 samples) so stratified splitting won’t explode
- returns stratified 70/15/15 train/val/test splits (seeded for reproducibility)

outputs:
- three pandas DataFrames (train_df, val_df, test_df) with columns:
  ['Image', 'Category', 'image_path']
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_dataset(csv_path="../labels_simplified.csv", images_dir="../JPEGImages"):
    """
    Load labels CSV, resolve image paths (.jpg/.jpeg), drop missing labels & files,
    drop ultra-rare classes (<4 samples), then stratified split 70/15/15.
    Returns: train_df, val_df, test_df (each has Image, Category, image_path)
    """
    df = pd.read_csv(csv_path)

    # --- sanity check: make sure CSV has Image and Category columns ---
    if "Image" not in df.columns or "Category" not in df.columns:
        raise ValueError("CSV must contain 'Image' and 'Category' columns.")

    # --- drop rows with missing labels (we don't want blanks messing with stratify) ---
    df = df.dropna(subset=["Category"])

    # --- helper to find image path; prefers .jpg over .jpeg if both exist ---
    def find_image_path(image_id):
        jpg_path = os.path.join(images_dir, f"BloodImage_{int(image_id):05d}.jpg")
        jpeg_path = os.path.join(images_dir, f"BloodImage_{int(image_id):05d}.jpeg")
        return jpg_path if os.path.exists(jpg_path) else jpeg_path

    # --- map each image id to its actual path ---
    df["image_path"] = df["Image"].apply(find_image_path)

    # --- catch missing files (if any) and warn the user ---
    missing = [p for p in df["image_path"] if not os.path.exists(p)]
    if missing:
        print(f"Warning: {len(missing)} images are missing! Example: {missing[:5]}")
    df = df[df["image_path"].apply(lambda x: os.path.exists(x))]

    print(f"After filtering missing files: {len(df)} images remain")

    # --- drop classes that have less than 4 examples (keeps stratify stable) ---
    class_counts = df["Category"].value_counts()
    too_small = class_counts[class_counts < 4].index.tolist()
    if too_small:
        print(f"Dropping classes with fewer than 4 samples: {too_small}")
        df = df[~df["Category"].isin(too_small)]

    print("\nClass distribution after filtering:")
    print(df["Category"].value_counts())

    # --- stratified train/val/test split (70/15/15) using random_state=42 ---
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["Category"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["Category"], random_state=42
    )

    # --- all done: send back our three tidy splits ---
    return train_df, val_df, test_df

if __name__ == "__main__":
    # --- quick test run just to show the sizes of each split ---
    tr, va, te = load_dataset()
    print(f"Train: {len(tr)}, Val: {len(va)}, Test: {len(te)}")

