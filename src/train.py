import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
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

# ---- new torch.amp API (fixes deprecation warnings) ----
from torch.amp import GradScaler, autocast


# ----------------------------
# Custom Dataset
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
        pixel_values = inputs["pixel_values"].squeeze(0)  # remove batch dim
        return {"pixel_values": pixel_values, "label": torch.tensor(label, dtype=torch.long)}


# ----------------------------
# Metrics helpers
# ----------------------------
@torch.no_grad()
def eval_loader_topk(clip_model, classifier, loader, device, k=(1, 2)):
    classifier.eval()
    topk_hits = {kk: 0 for kk in k}
    total = 0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        image_features = clip_model.get_image_features(pixel_values=pixel_values)
        logits = classifier(image_features)

        for kk in k:
            _, pred = torch.topk(logits, kk, dim=1)
            match = (pred == labels.unsqueeze(1)).any(dim=1)
            topk_hits[kk] += match.sum().item()

        total += labels.size(0)

    return {kk: (topk_hits[kk] / total if total > 0 else 0.0) for kk in k}, total


def evaluate_on_test(clip_model, classifier, processor, label2id, batch_size=32, device="cpu"):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset (data_loader already drops ultra-rare classes)
    train_df, val_df, _ = load_dataset()

    # Label mapping
    classes = sorted(train_df["Category"].unique())
    label2id = {label: i for i, label in enumerate(classes)}
    print("\nClass mapping:", label2id)

    # Models
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    # Freeze CLIP (feature extractor)
    for p in clip_model.parameters():
        p.requires_grad = False

    classifier = torch.nn.Linear(clip_model.config.projection_dim, len(classes)).to(device)

    optimizer = AdamW(classifier.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # AMP scaler (enabled only for CUDA)
    scaler = GradScaler(device.type if device.type == "cuda" else "cpu", enabled=(device.type == "cuda"))

    # ----- Augmentations (train only) -----
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

    # Datasets
    train_dataset = BloodCellDataset(train_df, processor, label2id, transforms=train_tfms)
    val_dataset = BloodCellDataset(val_df, processor, label2id, transforms=None)

    # Class-balanced sampling
    class_counts = train_df["Category"].value_counts()
    weight_per_class = {cls: 1.0 / cnt for cls, cnt in class_counts.items()}
    sample_weights = train_df["Category"].map(weight_per_class).values
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    # Loaders (Windows-friendly workers)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Training loop with early stopping on val top-1
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

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():  # CLIP frozen
                with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    image_features = clip_model.get_image_features(pixel_values=pixel_values)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = classifier(image_features)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.2f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Training Loss: {avg_loss:.4f}")

        # Validation top-1/top-2
        metrics, _ = eval_loader_topk(clip_model, classifier, val_loader, device, k=(1, 2))
        val_top1, val_top2 = metrics[1], metrics[2]
        print(f"Epoch {epoch+1} | Validation Top-1: {val_top1*100:.2f}% | Top-2: {val_top2*100:.2f}%")

        # Save best classifier
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

    # Save labels/model/processor for inference
    with open(os.path.join(save_dir, "labels.txt"), "w", encoding="utf-8") as f:
        for c in classes:
            f.write(c + "\n")
    clip_model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    print(f"Saved labels/model/processor to {os.path.abspath(save_dir)}")

    # Load best classifier and evaluate on test
    if os.path.exists(best_classifier_path):
        classifier.load_state_dict(torch.load(best_classifier_path, map_location=device))
        print(f"Loaded best classifier from {best_classifier_path}")
    else:
        print("Warning: best classifier checkpoint not found; using last epoch weights.")

    evaluate_on_test(clip_model, classifier, processor, label2id, batch_size=32, device=device)


if __name__ == "__main__":
    # Tweak epochs/batch_size/lr if you like
    train_model(epochs=12, batch_size=32, lr=5e-4, patience=3)

