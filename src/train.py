"""
train.py — fine-tune a tiny classifier head on top of frozen CLIP image features

What this file does (in plain English):
- Loads the dataset splits from data_loader.py (we already filter out tiny classes etc.)
- Uses Hugging Face's CLIP (ViT-B/32) as a *frozen* feature extractor — we don't touch its weights
- Trains a single linear layer (the "head") on top of CLIP's image embeddings
- Uses mild data augmentations (if torchvision is installed) and class-balanced sampling
- Tracks Top-1 / Top-2 validation accuracy and does early stopping
- Saves the best classifier weights + labels + CLIP model + processor under ../results/fine_tuned_head
- Finally, evaluates on the test set and prints Top-1 / Top-2 accuracy

TL;DR: it's the "linear probe" training script for our blood cell classifier.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from data_loader import load_dataset
from PIL import Image

# ---- optional light augmentations ----
# If torchvision isn't around, we just skip augments (no drama).
try:
    import torchvision.transforms as T
except Exception:
    T = None
    print("Warning: torchvision not found. Augmentations will be disabled.")

# ---- new torch.amp API (fixes deprecation warnings) ----
# Using the modern AMP interface so PyTorch doesn't shout at us.
from torch.amp import GradScaler, autocast


# ----------------------------
# Custom Dataset
# ----------------------------
class BloodCellDataset(Dataset):
    """
    Super tiny Dataset wrapper:
    - opens the image
    - applies optional torchvision transforms (augmentations)
    - packs it through the CLIP processor to get pixel_values
    - returns pixel_values tensor + integer label
    """
    def __init__(self, df, processor, label2id, transforms=None):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.label2id = label2id
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)  # light augments (if available)
        label = self.label2id[row["Category"]]
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # remove batch dim (CLIP returns [1, C, H, W])
        return {"pixel_values": pixel_values, "label": torch.tensor(label, dtype=torch.long)}


# ----------------------------
# Metrics helpers
# ----------------------------
@torch.no_grad()
def eval_loader_topk(clip_model, classifier, loader, device, k=(1, 2)):
    """
    Evaluate Top-K accuracy (we use K = 1 and 2).
    - Runs the frozen CLIP image tower to get features
    - Runs the linear head to get logits
    - Checks whether the true label is in the top-K predictions
    """
    classifier.eval()
    topk_hits = {kk: 0 for kk in k}
    total = 0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        image_features = clip_model.get_image_features(pixel_values=pixel_values)
        logits = classifier(image_features)

        # Grab top-K predicted indices and see if the true label is in there
        for kk in k:
            _, pred = torch.topk(logits, kk, dim=1)
            match = (pred == labels.unsqueeze(1)).any(dim=1)
            topk_hits[kk] += match.sum().item()

        total += labels.size(0)

    return {kk: (topk_hits[kk] / total if total > 0 else 0.0) for kk in k}, total


def evaluate_on_test(clip_model, classifier, processor, label2id, batch_size=32, device="cpu"):
    """
    Quick held-out check:
    - Rebuilds the test loader with the same label mapping
    - Prints Top-1 / Top-2 accuracy on the test set
    """
    _, _, test_df = load_dataset()
    keep = set(label2id.keys())
    test_df = test_df[test_df["Category"].isin(keep)].reset_index(drop=True)
    test_dataset = BloodCellDataset(test_df, processor, label2id, transforms=None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    metrics, n = eval_loader_topk(clip_model, classifier, test_loader, device, k=(1, 2))
    print(f"Test Top-1 Accuracy: {metrics[1]*100:.2f}% | Top-2 Accuracy: {metrics[2]*100:.2f}%  (n={n})")


# ----------------------------
# Training (freeze CLIP; train classifier head only)
# ----------------------------
def train_model(epochs=12, batch_size=32, lr=5e-4, patience=3):
    """
    Main training loop:
    - Freezes CLIP, trains only the linear classifier head
    - Class-balanced sampler to not ignore minority classes
    - Early stopping on validation Top-1
    - Saves best classifier + assets for inference later
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset (data_loader already drops ultra-rare classes)
    train_df, val_df, _ = load_dataset()

    # Label mapping (sorted for stability)
    classes = sorted(train_df["Category"].unique())
    label2id = {label: i for i, label in enumerate(classes)}
    print("\nClass mapping:", label2id)

    # Models: processor (preprocess/normalize) + CLIP model (frozen feature extractor)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    # Freeze CLIP (feature extractor mode)
    for p in clip_model.parameters():
        p.requires_grad = False

    # A tiny linear head that maps CLIP features -> our 4 classes
    classifier = torch.nn.Linear(clip_model.config.projection_dim, len(classes)).to(device)

    # Optimizer & loss for the head
    optimizer = AdamW(classifier.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # AMP scaler (only actually active on CUDA; on CPU it's a no-op)
    scaler = GradScaler(device.type if device.type == "cuda" else "cpu", enabled=(device.type == "cuda"))

    # ----- Augmentations (train only) -----
    # Keep these light — CLIPProcessor already handles resize/normalize.
    train_tfms = None
    if T is not None:
        train_tfms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(degrees=10, expand=False),
            T.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02),
        ])
    else:
        print("Augmentations disabled (torchvision missing).")

    # Datasets for train/val
    train_dataset = BloodCellDataset(train_df, processor, label2id, transforms=train_tfms)
    val_dataset = BloodCellDataset(val_df, processor, label2id, transforms=None)

    # Class-balanced sampling:
    # Weight = 1 / class_count, so rarer classes get sampled more often.
    class_counts = train_df["Category"].value_counts()
    weight_per_class = {cls: 1.0 / cnt for cls, cnt in class_counts.items()}
    sample_weights = train_df["Category"].map(weight_per_class).values
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    # DataLoaders (Windows-safe: num_workers=0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Training loop with early stopping on val Top-1
    best_val_top1 = -1.0
    epochs_no_improve = 0

    save_dir = "../results/fine_tuned_head"
    os.makedirs(save_dir, exist_ok=True)
    best_classifier_path = os.path.join(save_dir, "classifier.pt")

    for epoch in range(epochs):
        classifier.train()
        total_loss = 0.0

        loop = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch in loop:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            # Clear grads (set_to_none is a small perf win)
            optimizer.zero_grad(set_to_none=True)

            # Get frozen CLIP features under autocast (only relevant on CUDA)
            with torch.no_grad():  # CLIP frozen
                with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    image_features = clip_model.get_image_features(pixel_values=pixel_values)

            # Forward head + CE loss under autocast
            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = classifier(image_features)
                loss = criterion(logits, labels)

            # Mixed-precision backward/update (or effectively FP32 on CPU)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.2f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Training Loss: {avg_loss:.4f}")

        # Validation Top-1 / Top-2 to track progress
        metrics, _ = eval_loader_topk(clip_model, classifier, val_loader, device, k=(1, 2))
        val_top1, val_top2 = metrics[1], metrics[2]
        print(f"Epoch {epoch+1} | Validation Top-1: {val_top1*100:.2f}% | Top-2: {val_top2*100:.2f}%")

        # Checkpoint the best head so far
        if val_top1 > best_val_top1 + 1e-6:
            best_val_top1 = val_top1
            epochs_no_improve = 0
            torch.save(classifier.state_dict(), best_classifier_path)
            print(f"✓ Saved new best classifier (val_acc={val_top1:.4f}) -> {best_classifier_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping – no improvement.")
                break

    # Save labels/model/processor for inference (clip_inference.py expects these)
    with open(os.path.join(save_dir, "labels.txt"), "w", encoding="utf-8") as f:
        for c in classes:
            f.write(c + "\n")
    clip_model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    print(f"Saved labels/model/processor to {os.path.abspath(save_dir)}")

    # Load the best head and run a final test set eval
    if os.path.exists(best_classifier_path):
        classifier.load_state_dict(torch.load(best_classifier_path, map_location=device))
        print(f"Loaded best classifier from {best_classifier_path}")
    else:
        print("Warning: best classifier checkpoint not found; using last epoch weights.")

    evaluate_on_test(clip_model, classifier, processor, label2id, batch_size=32, device=device)


if __name__ == "__main__":
    # Tweak epochs/batch_size/lr/patience here if you feel like experimenting
    train_model(epochs=12, batch_size=32, lr=5e-4, patience=3)
