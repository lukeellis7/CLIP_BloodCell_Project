"""
utils.py — small helper functions for plotting + reporting

what this file does:
- cleans up the DataFrame labels before plotting (avoids NaNs and dtype weirdness)
- plots a confusion matrix (counts or % normalized)
- saves a clean sklearn-style classification report to disk
used by: evaluation.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


# ----------------------------
# tiny helper — label cleaner
# ----------------------------
def _clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure labels exist and are strings (no NaNs or floats sneaking in)."""
    df = df.copy()
    for col in ["True_Label", "Predicted_Label"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
        else:
            raise ValueError(f"DataFrame must contain '{col}' column")
    return df


# ----------------------------
# confusion matrix plotting
# ----------------------------
def plot_confusion_matrix(df: pd.DataFrame,
                          output_path: str = "../results/confusion_matrix.png",
                          normalize: bool = False,
                          cmap: str = "Blues") -> None:
    """
    Plots and saves a confusion matrix heatmap.

    normalize=False -> raw counts (integers)
    normalize=True  -> row-normalized percentages (floats, 0–100%)

    we save directly to file because this script is used by evaluation.py
    """
    df = _clean_labels(df)

    # combine all possible classes from both columns just in case a label appears only in preds
    labels = sorted(set(df["True_Label"].unique()) | set(df["Predicted_Label"].unique()))
    print(f"Labels being used for confusion matrix: {labels}")

    cm = confusion_matrix(
        df["True_Label"],
        df["Predicted_Label"],
        labels=labels,
        normalize="true" if normalize else None
    )

    # tweak formatting if normalized (show as %)
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

    # draw the heatmap
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

    # make sure the folder exists then save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Confusion matrix saved to {output_path}")
    plt.close()


# ----------------------------
# classification report writer
# ----------------------------
def classification_report_to_file(df: pd.DataFrame,
                                  output_path: str = "../results/classification_report.txt",
                                  digits: int = 3) -> str:
    """
    Builds a sklearn classification_report and saves it as a plain text file.

    Returns the text string too, can print or reuse it later.
    """
    df = _clean_labels(df)

    # make sure both true/pred labels are represented
    labels_sorted = sorted(set(df["True_Label"].unique()) | set(df["Predicted_Label"].unique()))

    report = classification_report(
        df["True_Label"],
        df["Predicted_Label"],
        labels=labels_sorted,
        zero_division=0,  # avoids "nan" divisions if a class never shows up
        digits=digits
    )

    # handy overall accuracy number
    overall_acc = (df["True_Label"] == df["Predicted_Label"]).mean()

    # save to file (pretty text format)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Classification Report\n")
        f.write("====================\n\n")
        f.write(report)
        f.write("\n")
        f.write(f"Overall accuracy: {overall_acc:.4f}\n")

    print(f"Classification report saved to {output_path}")
    return report



