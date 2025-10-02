import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def load_predictions(predictions_path="../results/clip_predictions.csv"):
    try:
        df = pd.read_csv(predictions_path)
        print(f"Successfully loaded predictions from {predictions_path}")
        print(f"Total predictions: {len(df)}")
        return df
    except Exception as e:
        print(f"ERROR: Could not read predictions CSV. Reason: {e}")
        return None

def plot_confusion_matrix(df, output_path="../results/confusion_matrix.png"):
    if not all(col in df.columns for col in ["True_Label", "Predicted_Label"]):
        raise ValueError("DataFrame must contain 'True_Label' and 'Predicted_Label' columns")

    df["True_Label"] = df["True_Label"].fillna("Unknown").astype(str)
    df["Predicted_Label"] = df["Predicted_Label"].fillna("Unknown").astype(str)

    labels = sorted(df["True_Label"].unique())
    print(f"Labels being used for confusion matrix: {labels}")

    cm = confusion_matrix(df["True_Label"], df["Predicted_Label"], labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Confusion matrix saved to {output_path}")
    plt.close()

def classification_report_to_file(df, output_path="../results/classification_report.txt"):
    """
    Writes a full sklearn classification_report to a text file.
    Returns the string report too (if you want to print it).
    """
    df = df.copy()
    df["True_Label"] = df["True_Label"].fillna("Unknown").astype(str)
    df["Predicted_Label"] = df["Predicted_Label"].fillna("Unknown").astype(str)

    labels = sorted(df["True_Label"].unique())
    report = classification_report(
        df["True_Label"], df["Predicted_Label"], labels=labels, zero_division=0
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Classification Report\n")
        f.write("====================\n\n")
        f.write(report)

    print(f"Classification report saved to {output_path}")
    return report

