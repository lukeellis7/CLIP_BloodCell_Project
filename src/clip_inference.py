import os
import sys
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
    """Return 'classifier' if head exists (and not explicitly set to zero-shot), else 'zeroshot'."""
    if requested_mode.lower() == "classifier":
        return "classifier"
    if requested_mode.lower() == "zeroshot":
        return "zeroshot"
    head_ok = os.path.exists(os.path.join(auto_save_dir, "classifier.pt")) and \
              os.path.exists(os.path.join(auto_save_dir, "labels.txt"))
    return "classifier" if head_ok else "zeroshot"

def build_full_dataframe():
    """Concatenate train/val/test into one DataFrame for inference."""
    train_df, val_df, test_df = load_dataset()
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    for col in ["Category"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df

def run_clip_inference(mode="auto"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    save_dir = "../results/fine_tuned_head"
    mode = pick_mode(save_dir, requested_mode=mode)
    print(f"Inference mode: {mode}")

    df = build_full_dataframe()
    print(f"Total images to evaluate: {len(df)}")

    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir,
                           "classifier_predictions.csv" if mode == "classifier"
                           else "clip_predictions.csv")
    print("Predictions will be saved to:", os.path.abspath(out_csv))

    if mode == "classifier" and os.path.exists(save_dir):
        processor = CLIPProcessor.from_pretrained(save_dir)
        clip_model = CLIPModel.from_pretrained(save_dir).to(device)
        labels = load_labels_txt(os.path.join(save_dir, "labels.txt"))
        label2id = {c: i for i, c in enumerate(labels)}
        classifier = torch.nn.Linear(clip_model.config.projection_dim, len(labels)).to(device)
        classifier.load_state_dict(torch.load(os.path.join(save_dir, "classifier.pt"),
                                              map_location=device))
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

            if mode == "classifier":
                inputs = processor(images=image, return_tensors="pt").to(device)
                feats = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
                logits = classifier(feats)
                probs = torch.softmax(logits, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                pred_label = labels[pred_idx]
                confidence = probs[0, pred_idx].item()
            else:
                inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)
                outputs = clip_model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                pred_label = labels[pred_idx]
                confidence = probs[0, pred_idx].item()

            is_correct = (pred_label == true_label)
            correct += int(is_correct)
            total += 1

            results.append({
                "Image": row.get("Image", i),
                "True_Label": true_label,
                "Predicted_Label": pred_label,
                "Confidence": round(float(confidence), 4),
                "Correct": bool(is_correct)
            })

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(df)}")

    acc = (correct / total) if total else 0.0
    print(f"\nFinal Accuracy ({mode}): {acc:.2%} on {total} images")

    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Predictions saved to: {os.path.abspath(out_csv)}")

if __name__ == "__main__":
    req_mode = sys.argv[1] if len(sys.argv) > 1 else "auto"
    run_clip_inference(mode=req_mode)

