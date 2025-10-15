"""
clip_inference.py — runs CLIP either in zero-shot mode or using the trained linear head

what this file does (in plain english):
- decides whether to use CLIP straight out of the box (zero-shot) or the fine-tuned head (classifier mode)
- loads all blood cell images (train/val/test combined)
- runs them through CLIP, grabs the predictions and confidences
- calculates top-1 and top-2 accuracy while logging everything nicely
- saves a CSV of per-image results for later evaluation (in evaluation.py)

outputs:
- ../results/classifier_predictions.csv or clip_predictions.csv (depending on mode)
"""

import os
import sys
import torch
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from data_loader import load_dataset


# ----------------------------
# little helpers
# ----------------------------
def load_labels_txt(path):
    """reads labels.txt (one class per line) so we know what we're predicting"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"labels.txt not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    if not labels:
        raise ValueError("labels.txt is empty.")
    return labels


def pick_mode(auto_save_dir="../results/fine_tuned_head", requested_mode="auto"):
    """
    decides which mode to use:
    - 'classifier' if we've already trained and saved a head
    - 'zeroshot' if we’re just running raw CLIP
    - or honours whatever mode is forced via command line arg
    """
    rm = requested_mode.lower()
    if rm in {"classifier", "zeroshot"}:
        return rm
    head_ok = (
        os.path.exists(os.path.join(auto_save_dir, "classifier.pt"))
        and os.path.exists(os.path.join(auto_save_dir, "labels.txt"))
    )
    return "classifier" if head_ok else "zeroshot"


def build_full_dataframe():
    """grabs all train/val/test splits and stitches them together into one big dataframe"""
    train_df, val_df, test_df = load_dataset()
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    if "Category" in df.columns:
        df["Category"] = df["Category"].astype(str)
    return df


# ----------------------------
# main inference driver
# ----------------------------
def run_clip_inference(mode="auto"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    save_dir = "../results/fine_tuned_head"
    mode = pick_mode(save_dir, requested_mode=mode)
    print(f"Inference mode: {mode}")

    # --- build the combined dataframe of all images ---
    df = build_full_dataframe()
    print(f"Total images to evaluate: {len(df)}")

    # --- set up where we’ll save results ---
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(
        results_dir,
        "classifier_predictions.csv" if mode == "classifier" else "clip_predictions.csv",
    )
    print("Predictions will be saved to:", os.path.abspath(out_csv))

    # --- load CLIP + optional trained classifier head ---
    if mode == "classifier" and os.path.exists(save_dir):
        # load everything we saved from train.py
        processor = CLIPProcessor.from_pretrained(save_dir)
        clip_model = CLIPModel.from_pretrained(save_dir).to(device)
        labels = load_labels_txt(os.path.join(save_dir, "labels.txt"))
        label2id = {c: i for i, c in enumerate(labels)}
        classifier = torch.nn.Linear(clip_model.config.projection_dim, len(labels)).to(device)
        classifier.load_state_dict(
            torch.load(os.path.join(save_dir, "classifier.pt"), map_location=device)
        )
        classifier.eval()
        print(f"Loaded classifier with {len(labels)} classes:", labels)
    else:
        # no trained head found or user forced zero-shot
        print("Loading zero-shot CLIP model...")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        labels = sorted(df["Category"].unique())
        label2id = {c: i for i, c in enumerate(labels)}
        classifier = None
        print(f"Zero-shot classes ({len(labels)}):", labels)

    # --- tracking results and accuracy ---
    results = []
    correct_top1 = 0
    correct_top2 = 0
    total = 0

    print("\nStarting inference...\n")
    with torch.no_grad():
        for i, row in df.iterrows():
            image_path = row["image_path"]
            true_label = str(row["Category"])

            # open image safely
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"ERROR: Could not open image {image_path}. Skipping. Reason: {e}")
                continue

            # --- classifier mode: use our linear head ---
            if mode == "classifier":
                inputs = processor(images=image, return_tensors="pt").to(device)
                feats = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
                logits = classifier(feats)
                probs = torch.softmax(logits, dim=1)

            # --- zero-shot mode: compare image to all label prompts ---
            else:
                inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)
                outputs = clip_model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)

            # --- grab top-2 predictions ---
            top2_probs, top2_idx = torch.topk(probs, k=min(2, len(labels)), dim=1)
            pred1_idx = top2_idx[0, 0].item()
            pred2_idx = top2_idx[0, 1].item() if top2_idx.shape[1] > 1 else pred1_idx

            pred1_label = labels[pred1_idx]
            pred2_label = labels[pred2_idx]
            conf1 = float(top2_probs[0, 0].item())
            conf2 = float(top2_probs[0, 1].item()) if top2_probs.shape[1] > 1 else conf1
            margin = conf1 - conf2  # difference between top-1 and top-2 confidences

            # --- check if correct ---
            is_top1 = (pred1_label == true_label)
            is_top2 = is_top1 or (pred2_label == true_label)

            correct_top1 += int(is_top1)
            correct_top2 += int(is_top2)
            total += 1

            # --- save this image’s results ---
            results.append({
                "Image": row.get("Image", i),
                "True_Label": true_label,
                "Predicted_Label": pred1_label,
                "Top2_Label": pred2_label,
                "Confidence_Top1": round(conf1, 6),
                "Confidence_Top2": round(conf2, 6),
                "Margin_Top1_Top2": round(margin, 6),
                "Top1_Correct": bool(is_top1),
                "Top2_Correct": bool(is_top2),
            })

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(df)}")

    # --- overall accuracy stats ---
    top1_acc = (correct_top1 / total) if total else 0.0
    top2_acc = (correct_top2 / total) if total else 0.0
    print(f"\nFinal Accuracy ({mode}) — Top-1: {top1_acc:.2%} | Top-2: {top2_acc:.2%} on {total} images")

    # --- save everything to CSV ---
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Predictions saved to: {os.path.abspath(out_csv)}")


if __name__ == "__main__":
    # quick command-line override: python clip_inference.py [auto|classifier|zeroshot]
    req_mode = sys.argv[1] if len(sys.argv) > 1 else "auto"
    run_clip_inference(mode=req_mode)

