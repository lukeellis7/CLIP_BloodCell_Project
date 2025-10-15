"""
evaluation.py — crunches metrics and makes pretty pictures from your prediction CSV

what this file does:
- loads the per-image predictions CSV produced by clip_inference.py
- prints headline stats (Top-1 / Top-2)
- writes a precision/recall/F1 classification report to a txt file
- plots a confusion matrix (raw counts or normalized %)
- runs a "confidence sweep" to see the trade-off between auto-accept coverage and accuracy/F1
- saves everything into ../results so the report can pull figures straight from there
"""

# src/evaluation.py
import argparse
import os
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, accuracy_score

# reuse the helper that draws the heatmap (counts or normalized)
from utils import plot_confusion_matrix


def _coerce_str_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure labels are strings (and not NaN), so plotting/reporting doesn't choke."""
    for col in ["True_Label", "Predicted_Label"]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("Unknown")
    return df


def _maybe_add_top1_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill in a couple of convenience columns if they’re missing:
    - Top1_Correct: bool flag for whether Top-1 prediction was right
    - Confidence_Top1: if CSV used 'Confidence' earlier, map it over
    """
    # derive Top1_Correct if not provided
    if "Top1_Correct" not in df.columns and {"True_Label", "Predicted_Label"}.issubset(df.columns):
        df["Top1_Correct"] = (df["True_Label"] == df["Predicted_Label"])

    # standardize confidence column name if needed
    if "Confidence_Top1" not in df.columns:
        if "Confidence" in df.columns:  # earlier pipeline name
            df["Confidence_Top1"] = df["Confidence"]
    return df


def _compute_topk_accuracy(df: pd.DataFrame) -> dict:
    """
    Compute Top-1 (always) and Top-2 (if there is a second guess column) accuracies.


    Treat Top-2 as "correct if true label equals Predicted_Label OR the second label".
    Second label column can be 'Top2_Label' (preferred) or 'Second_Label' (older name).
    """
    out = {}
    # Top-1 accuracy is just exact match on the primary prediction
    if {"True_Label", "Predicted_Label"}.issubset(df.columns):
        top1 = accuracy_score(df["True_Label"], df["Predicted_Label"])
        out["top1"] = float(top1)
    else:
        out["top1"] = None

    # Try to find the second-prediction column
    second_label_col = None
    for cand in ["Top2_Label", "Second_Label"]:
        if cand in df.columns:
            second_label_col = cand
            break

    # Top-2 accuracy: credit if true label is in the top2 set for that row
    if second_label_col is not None and {"True_Label", "Predicted_Label", second_label_col}.issubset(df.columns):
        top2_correct = (
            (df["True_Label"] == df["Predicted_Label"]) |
            (df["True_Label"] == df[second_label_col])
        ).mean()
        out["top2"] = float(top2_correct)
    else:
        out["top2"] = None

    return out


def classification_report_to_file(df: pd.DataFrame, output_path: str) -> str:
    """
    Build the sklearn classification report (precision/recall/F1 per class)
    and save it to a .txt file for the appendix.
    """
    _coerce_str_labels(df)
    labels = sorted(df["True_Label"].unique())
    report = classification_report(
        df["True_Label"],
        df["Predicted_Label"],
        labels=labels,
        zero_division=0,  # don't explode on missing classes
        digits=4          # nice, consistent rounding
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Classification Report\n")
        f.write("====================\n\n")
        f.write(report)
        f.write("\n")
    return report


def confidence_threshold_sweep(
    df: pd.DataFrame,
    results_dir: str,
    start: float = 0.0,
    stop: float = 0.95,
    step: float = 0.05
):
    """
    The "how picky should we be?" experiment.
    Sweep a confidence threshold t from start..stop and, for each t:
      - coverage: fraction of samples we'd auto-accept (conf >= t)
      - accuracy_accepted: accuracy only on those accepted samples
      - precision/recall/F1 treating "correct" as the positive class

    This is great for proposing a human take over cutoff in the report.

    Needs: Confidence_Top1 (float) and Top1_Correct (bool) in the CSV.
    Saves: CSV + 2 PNGs into results_dir.
    """
    import matplotlib.pyplot as plt  # local import so this file doesn't hard-require matplotlib if unused

    if "Confidence_Top1" not in df.columns or "Top1_Correct" not in df.columns:
        print("Skipping threshold sweep (Confidence_Top1/Top1_Correct not found).")
        return

    # grab numpy arrays for speed
    conf = df["Confidence_Top1"].astype(float).values
    correct = df["Top1_Correct"].astype(bool).values
    n = len(df)
    total_correct = correct.sum()

    rows = []
    thresholds = np.arange(start, stop + 1e-9, step)  # include stop
    for t in thresholds:
        accepted = conf >= t
        cov = accepted.mean()  # how many are auto-accepted at this threshold

        if accepted.sum() > 0:
            # accuracy among accepted predictions (quality if we automate at this t)
            acc_acc = correct[accepted].mean()

            # micro P/R/F1 with "correct" treated as the positive class
            tp = (correct & accepted).sum()
            fp = (~correct & accepted).sum()
            fn = total_correct - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        else:
            acc_acc = 0.0
            precision = 0.0
            recall = 0.0
            f1 = 0.0

        rows.append({
            "threshold": round(float(t), 3),
            "coverage": float(cov),
            "accuracy_accepted": float(acc_acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "num_accepted": int(accepted.sum()),
            "num_total": int(n)
        })

    out_df = pd.DataFrame(rows)
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "confidence_sweep.csv")
    out_df.to_csv(csv_path, index=False)
    print("Saved confidence sweep to:", os.path.abspath(csv_path))

    # 1) coverage vs accuracy@accepted — nice for the automation trade-off story
    plt.figure()
    plt.plot(out_df["threshold"], out_df["coverage"], label="Coverage")
    plt.plot(out_df["threshold"], out_df["accuracy_accepted"], label="Accuracy@Accepted")
    plt.xlabel("Confidence threshold")
    plt.ylabel("Proportion")
    plt.title("Coverage vs. Accuracy@Accepted")
    plt.legend()
    plt.tight_layout()
    cov_plot = os.path.join(results_dir, "confidence_sweep_coverage_accuracy.png")
    plt.savefig(cov_plot, dpi=200)
    plt.close()
    print("Saved:", os.path.abspath(cov_plot))

    # 2) F1 vs threshold — shows the sweet spot if we treat “correct” as positive
    plt.figure()
    plt.plot(out_df["threshold"], out_df["f1"])
    plt.xlabel("Confidence threshold")
    plt.ylabel("F1 (correct-as-positive)")
    plt.title("F1 vs. Confidence Threshold")
    plt.tight_layout()
    f1_plot = os.path.join(results_dir, "confidence_sweep_f1.png")
    plt.savefig(f1_plot, dpi=200)
    plt.close()
    print("Saved:", os.path.abspath(f1_plot))


def run_evaluation(
    pred_csv: str,
    results_dir: str = "../results",
    normalize_cm: bool = False,
    sweep_start: float = 0.0,
    sweep_stop: float = 0.95,
    sweep_step: float = 0.05
):
    """
    Main entry point:
    - loads predictions
    - prints Top-k summary
    - saves confusion matrix + classification report
    - runs the confidence sweep (if the columns exist)
    """
    if not os.path.exists(pred_csv):
        raise FileNotFoundError(f"Predictions CSV not found at {pred_csv}")

    # load + clean
    df = pd.read_csv(pred_csv)
    _coerce_str_labels(df)
    _maybe_add_top1_fields(df)

    print(f"Successfully loaded predictions from {pred_csv}")
    print(f"Total predictions: {len(df)}\n")

    os.makedirs(results_dir, exist_ok=True)

    # --- headline numbers (Top-1 / Top-2) ---
    topk = _compute_topk_accuracy(df)
    if topk.get("top1") is not None:
        print(f"Top-1 Accuracy: {topk['top1']:.2%}")
    if topk.get("top2") is not None:
        print(f"Top-2 Accuracy: {topk['top2']:.2%}")

    # --- confusion matrix image (counts or normalized) ---
    print("\n--- Generating Confusion Matrix ---")
    cm_path = os.path.join(results_dir, "confusion_matrix.png" if not normalize_cm else "confusion_matrix_normalized.png")
    plot_confusion_matrix(df, output_path=cm_path, normalize=normalize_cm)
    print(f"Saved confusion matrix to: {os.path.abspath(cm_path)}")

    # --- full classification report to txt ---
    print("\n--- Generating Classification Report ---")
    report_path = os.path.join(results_dir, "classification_report.txt")
    _ = classification_report_to_file(df, report_path)
    print(f"Saved classification report to: {os.path.abspath(report_path)}")

    # --- sweep confidence thresholds and save plots/csv ---
    print("\n--- Confidence Threshold Sweep ---")
    confidence_threshold_sweep(df, results_dir, start=sweep_start, stop=sweep_stop, step=sweep_step)

# --- this whole next section is just to tweak arguments and file paths going in and out type beat ---
if __name__ == "__main__":
    # CLI sugar so we can tweak stuff without touching code
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_csv",
        type=str,
        default="../results/classifier_predictions.csv",
        help="Path to predictions CSV (columns: True_Label, Predicted_Label[, Confidence_Top1, Top1_Correct, Top2_Label])"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="../results",
        help="Directory to save evaluation artifacts"
    )
    parser.add_argument(
        "--normalize_cm",
        action="store_true",
        help="Plot normalized confusion matrix instead of raw counts"
    )
    parser.add_argument(
        "--sweep_start",
        type=float,
        default=0.0,
        help="Confidence sweep start (inclusive)"
    )
    parser.add_argument(
        "--sweep_stop",
        type=float,
        default=0.95,
        help="Confidence sweep stop (inclusive)"
    )
    parser.add_argument(
        "--sweep_step",
        type=float,
        default=0.05,
        help="Confidence sweep step"
    )
    args = parser.parse_args()

    run_evaluation(
        pred_csv=args.pred_csv,
        results_dir=args.results_dir,
        normalize_cm=args.normalize_cm,
        sweep_start=args.sweep_start,
        sweep_stop=args.sweep_stop,
        sweep_step=args.sweep_step
    )

