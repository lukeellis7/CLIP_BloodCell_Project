import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def _clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure labels are strings and no NaNs."""
    df = df.copy()
    for col in ["True_Label", "Predicted_Label"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
        else:
            raise ValueError(f"DataFrame must contain '{col}' column")
    return df

def plot_confusion_matrix(df: pd.DataFrame,
                          output_path: str = "../results/confusion_matrix.png",
                          normalize: bool = False,
                          cmap: str = "Blues") -> None:
    """
    Save a confusion matrix heatmap.
    - normalize=False -> raw counts (ints)
    - normalize=True  -> row-normalized percentages (floats)
    """
    df = _clean_labels(df)

    # Use union to be safe if preds contain a class not present in y_true
    labels = sorted(set(df["True_Label"].unique()) | set(df["Predicted_Label"].unique()))
    print(f"Labels being used for confusion matrix: {labels}")

    cm = confusion_matrix(
        df["True_Label"],
        df["Predicted_Label"],
        labels=labels,
        normalize="true" if normalize else None
    )

    # When normalized, show percentages (0-100); otherwise raw counts
    if normalize:
        cm_display = cm * 100.0
        fmt = ".1f"
        vmin, vmax = 0, 100
        title = "Confusion Matrix (row-normalized, %)"
    else:
        cm_display = cm
        fmt = "d"
        vmin, vmax = None, None
        title = "Confusion Matrix (counts)"

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        vmin=vmin,
        vmax=vmax,
        cbar=True
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Confusion matrix saved to {output_path}")
    plt.close()

def classification_report_to_file(df: pd.DataFrame,
                                  output_path: str = "../results/classification_report.txt",
                                  digits: int = 3) -> str:
    """
    Generate a sklearn classification_report and save it.
    Returns the report string.
    """
    df = _clean_labels(df)

    labels_sorted = sorted(set(df["True_Label"].unique()) | set(df["Predicted_Label"].unique()))

    report = classification_report(
        df["True_Label"],
        df["Predicted_Label"],
        labels=labels_sorted,
        zero_division=0,
        digits=digits
    )

    overall_acc = (df["True_Label"] == df["Predicted_Label"]).mean()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Classification Report\n")
        f.write("====================\n\n")
        f.write(report)
        f.write("\n")
        f.write(f"Overall accuracy: {overall_acc:.4f}\n")

    print(f"Classification report saved to {output_path}")
    return report


