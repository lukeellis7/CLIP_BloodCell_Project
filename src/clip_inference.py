import os
import sys
import json
import torch
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from data_loader import load_dataset

def load_labels_txt(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"labels.txt not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    if not labels:
        raise ValueError("labels.txt is empty.")
    return labels

def pick_mode(auto_save_dir="../results/fine_tuned_head", requested_mode="auto"):
    if requested_mode.lower() == "classifier":
        return "classifier"
    if requested_mode.lower() == "zeroshot":
        return "zeroshot"
    head_ok = os.path.exists(os.path.join(auto_save_dir, "classifier.pt")) and \
              os.path.exists(os.path.join(auto_save_dir, "labels.txt"))
    return "classifier" if head_ok else "zeroshot"

def build_full_dataframe():
    train_df, val_df, test_df = load_dataset()
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    if "Category" in df.columns:
        df["Category"] = df["Category"].astype(str)
    return df

def run_clip_inference(mode="auto", tta=False, save_probs=True):
    """
    mode: 'auto' | 'classifier' | 'zeroshot'
    tta: if True, average prediction with a horizontally flipped version
    save_probs: if True, include per-class probability columns in CSV
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    save_dir = "../results/fine_tuned_head"
    mode = pick_mode(save_dir, requested_mode=mode)
    print(f"Inference mode: {mode}")

    df = build_full_dataframe()
    print(f"Total images to evaluate: {len(df)}")

    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(
        results_dir,
        "classifier_predictions.csv" if mode == "classifier" else "clip_predictions.csv"
    )
    print("Predictions will be saved to:", os.path.abspath(out_csv))

    if mode == "classifier" and os.path.exists(save_dir):
        processor = CLIPProcessor.from_pretrained(save_dir)
        clip_model = CLIPModel.from_pretrained(save_dir).to(device)
        labels = load_labels_txt(os.path.join(save_dir, "labels.txt"))
        label2id = {c: i for i, c in enumerate(labels)}
        classifier = torch.nn.Linear(clip_model.config.projection_dim, len(labels)).to(device)
        classifier.load_state_dict(torch.load(os.path.join(save_dir, "classifier.pt"), map_location=device))
        classifier.eval()
        print(f"Loaded classifier with {len(labels)} classes:", labels)
    else:
        print("Loading zero-shot CLIP model...")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        labels = sorted(df["Category"].unique())
        label2id = {c: i for i, c in enumerate(labels)}
        print(f"Zero-shot classes ({len(labels)}):", labels)

    results = []
    correct = total = 0

    # Simple TTA: horizontal flip
    def _predict_probs(image: Image.Image):
        if mode == "classifier":
            inputs = processor(images=image, return_tensors="pt").to(device)
            feats = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
            logits = classifier(feats)
            probs = torch.softmax(logits, dim=1)
            return probs.squeeze(0)
        else:
            inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)
            outputs = clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
            return probs.squeeze(0)

    print("\nStarting inference...\n")
    with torch.no_grad():
        for i, row in df.iterrows():
            image_path = row["image_path"]
            true_label = str(row["Category"])

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"ERROR: Could not open image {image_path}. Skipping. Reason: {e}")
                continue

            probs = _predict_probs(image)

            if tta:
                flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
                probs_flip = _predict_probs(flipped)
                probs = (probs + probs_flip) / 2.0

            pred_idx = int(torch.argmax(probs).item())
            pred_label = labels[pred_idx]
            confidence = float(probs[pred_idx].item())

            # Top-2 (for analysis)
            top2_conf, top2_idx = torch.topk(probs, k=min(2, len(labels)))
            top2_idx = top2_idx.tolist()
            top2_conf = [float(c) for c in top2_conf.tolist()]
            pred2_label = labels[top2_idx[1]] if len(top2_idx) > 1 else pred_label
            pred2_conf = top2_conf[1] if len(top2_conf) > 1 else confidence

            is_correct = (pred_label == true_label)
            correct += int(is_correct)
            total += 1

            row_out = {
                "Image": row.get("Image", i),
                "True_Label": true_label,
                "Predicted_Label": pred_label,
                "Confidence": round(confidence, 4),
                "Predicted_Label_2": pred2_label,
                "Confidence_2": round(pred2_conf, 4),
                "Correct": bool(is_correct)
            }

            if save_probs:
                for j, cls in enumerate(labels):
                    row_out[f"Prob_{cls}"] = round(float(probs[j].item()), 6)

            results.append(row_out)

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(df)}")

    acc = (correct / total) if total else 0.0
    print(f"\nFinal Accuracy ({mode}): {acc:.2%} on {total} images")

    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Predictions saved to: {os.path.abspath(out_csv)}")

if __name__ == "__main__":
    # usage:
    #   python src/clip_inference.py            -> auto
    #   python src/clip_inference.py zeroshot   -> force zero-shot
    #   python src/clip_inference.py classifier -> force classifier
    req_mode = "auto"
    tta = False
    if len(sys.argv) >= 2:
        req_mode = sys.argv[1]
    if len(sys.argv) >= 3:
        tta = sys.argv[2].lower() == "tta"
    run_clip_inference(mode=req_mode, tta=tta, save_probs=True)

