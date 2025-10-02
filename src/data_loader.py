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

    if "Image" not in df.columns or "Category" not in df.columns:
        raise ValueError("CSV must contain 'Image' and 'Category' columns.")

    df = df.dropna(subset=["Category"])

    def find_image_path(image_id):
        jpg_path = os.path.join(images_dir, f"BloodImage_{int(image_id):05d}.jpg")
        jpeg_path = os.path.join(images_dir, f"BloodImage_{int(image_id):05d}.jpeg")
        return jpg_path if os.path.exists(jpg_path) else jpeg_path

    df["image_path"] = df["Image"].apply(find_image_path)

    missing = [p for p in df["image_path"] if not os.path.exists(p)]
    if missing:
        print(f"Warning: {len(missing)} images are missing! Example: {missing[:5]}")
    df = df[df["image_path"].apply(lambda x: os.path.exists(x))]

    print(f"After filtering missing files: {len(df)} images remain")

    class_counts = df["Category"].value_counts()
    too_small = class_counts[class_counts < 4].index.tolist()
    if too_small:
        print(f"Dropping classes with fewer than 4 samples: {too_small}")
        df = df[~df["Category"].isin(too_small)]

    print("\nClass distribution after filtering:")
    print(df["Category"].value_counts())

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["Category"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["Category"], random_state=42
    )

    return train_df, val_df, test_df

if __name__ == "__main__":
    tr, va, te = load_dataset()
    print(f"Train: {len(tr)}, Val: {len(va)}, Test: {len(te)}")


