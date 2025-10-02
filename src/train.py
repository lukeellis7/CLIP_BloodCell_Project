# src/train.py
import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
import torch.nn as nn
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from data_loader import load_dataset
from PIL import Image

# ---- optional light augmentations ----
try:
    import torchvision.transforms as T
except Exception:
    T = None
    print("Warning: torchvision not found. Augmentations will be disabled.")

# ----------------------------
# Custom Dataset: applies (optional) augmentations then CLIP processor; returns tensors
# ----------------------------
class BloodCellDataset(Dataset):
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
            image = self.transforms(image)
        label = self.label2id[row["Category"]]
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # [3, 224, 224]
        return {"pixel_values": pixel_values, "label": torch.tensor(label, dtype=torch.long)}

# ----------------------------
# Optional: quick test evaluation
# ----------------------------
def evaluate_on_test(clip_model, classifier, processor, label2id, batch_size=32, device="cpu"):
    _, _, test_df = load_dataset()
    keep = set(label2id.keys())
    test_df = test_df[test_df["Category"].isin(keep)].reset_index(drop=True)
    test_dataset = BloodCellDataset(test_df, processor, label2id, transforms=None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    classifier.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)
            image_features = clip_model.get_image_features(pixel_values=pixel_values)
            logits = classifier(image_features)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total if total > 0 else 0.0
    print(f"Test Accuracy: {acc:.2%}  (n={total})")

# ----------------------------
# Training Loop (freeze CLIP; train classifier head only)
# ----------------------------
def train_model(epochs=15, batch_size=32, lr=3e-4, sampler_gamma=1.0, early_stop_patience=5):
    """
    epochs: max epochs to train
    batch_size: batch size
    lr: AdamW learning rate for classifier head
    sampler_gamma: class sampling weight ~ 1 / (count ** gamma); try 1.0 (inverse), 0.75, 0.5 (inverse sqrt)
    early_stop_patience: stop if val acc doesn't improve for this many epochs
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset (data_loader handles dropping rare classes like Basophil)
    train_df, val_df, _ = load_dataset()

    # Label mapping (sorted for stability)
    classes = sorted(train_df["Category"].unique())
    label2id = {label: i for i, label in enumerate(classes)}
    print("\nClass mapping:", label2id)

    # Load CLIP (feature extractor mode)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    # Freeze CLIP
    for p in clip_model.parameters():
        p.requires_grad = False

    # Trainable classifier head
    classifier = nn.Linear(clip_model.config.projection_dim, len(classes)).to(device)

    # ----- Class-weighted loss (helps minority recall) -----
    class_counts = train_df["Category"].value_counts()
    # weights in label order
    cls_weights = torch.tensor([1.0 / class_counts[c] for c in classes], dtype=torch.float)
    # normalize weights to keep scale reasonable (optional)
    cls_weights = cls_weights / cls_weights.sum() * len(classes)
    cls_weights = cls_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=cls_weights)

    optimizer = AdamW(classifier.parameters(), lr=lr)

    # ----- Augmentations (train only) -----
    train_tfms = None
    if T is not None:
        train_tfms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.03),
        ])
    else:
        print("Augmentations disabled (torchvision missing).")

    # Datasets
    train_dataset = BloodCellDataset(train_df, processor, label2id, transforms=train_tfms)
    val_dataset = BloodCellDataset(val_df, processor, label2id, transforms=None)

    # ----- Class-balanced sampling (tunable) -----
    # weight per class = 1 / (count ** gamma); gamma in [0.5, 1.0]
    weight_per_class = {cls: (1.0 / (cnt ** sampler_gamma)) for cls, cnt in class_counts.items()}
    sample_weights = train_df["Category"].map(weight_per_class).values
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    # DataLoaders (Windows-friendly: num_workers=0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # ---- Early stopping tracking ----
    best_val = -1.0
    bad_epochs = 0
    save_dir = "../results/fine_tuned_head"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        classifier.train()
        total_loss = 0.0

        loop = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        for batch in loop:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            with torch.no_grad():  # CLIP is frozen
                image_features = clip_model.get_image_features(pixel_values=pixel_values)

            logits = classifier(image_features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch} | Training Loss: {avg_loss:.4f}")

        # ---- Validation ----
        classifier.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["label"].to(device)
                image_features = clip_model.get_image_features(pixel_values=pixel_values)
                logits = classifier(image_features)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch} | Validation Accuracy: {val_acc:.4f}")

        # ---- Checkpoint & Early stopping ----
        if val_acc > best_val:
            best_val = val_acc
            bad_epochs = 0
            # Save BEST classifier head only (keep CLIP frozen weights as-is)
            torch.save(classifier.state_dict(), os.path.join(save_dir, "classifier.pt"))
            print(f"✓ Saved new best classifier (val_acc={best_val:.4f}) -> {os.path.join(save_dir, 'classifier.pt')}")
        else:
            bad_epochs += 1
            if bad_epochs >= early_stop_patience:
                print("Early stopping – no improvement.")
                break

    # Always save labels/model/processor (used by clip_inference.py)
    with open(os.path.join(save_dir, "labels.txt"), "w", encoding="utf-8") as f:
        for c in classes:
            f.write(c + "\n")
    clip_model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    print(f"Saved labels/model/processor to {os.path.abspath(save_dir)}")

    # Load best head before test eval
    if os.path.exists(os.path.join(save_dir, "classifier.pt")):
        classifier.load_state_dict(torch.load(os.path.join(save_dir, "classifier.pt"), map_location=device))
        print(f"Loaded best classifier from {os.path.join(save_dir, 'classifier.pt')}")

    # Quick held-out evaluation
    evaluate_on_test(clip_model, classifier, processor, label2id, batch_size=batch_size, device=device)

if __name__ == "__main__":
    # You can tweak epochs/batch_size/gamma here if you like
    train_model(
        epochs=15,
        batch_size=32,
        lr=3e-4,
        sampler_gamma=1.0,        # try 0.75 or 0.5 for gentler rebalancing
        early_stop_patience=5
    )

