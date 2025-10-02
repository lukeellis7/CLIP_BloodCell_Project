import argparse, os
import pandas as pd
from utils import plot_confusion_matrix, classification_report_to_file

def run_evaluation(pred_csv, results_dir="../results"):
    if not os.path.exists(pred_csv):
        raise FileNotFoundError(f"Predictions CSV not found at {pred_csv}")

    df = pd.read_csv(pred_csv)

    for col in ["True_Label", "Predicted_Label"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    print(f"Successfully loaded predictions from {pred_csv}")
    print(f"Total predictions: {len(df)}\n")

    os.makedirs(results_dir, exist_ok=True)

    print("--- Generating Confusion Matrix ---")
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plot_confusion_matrix(df, output_path=cm_path)
    print(f"Saved confusion matrix to: {os.path.abspath(cm_path)}")

    print("\n--- Generating Classification Report ---")
    report_path = os.path.join(results_dir, "classification_report.txt")
    classification_report_to_file(df, output_path=report_path)
    print(f"Saved classification report to: {os.path.abspath(report_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", type=str, default="../results/classifier_predictions.csv",
                        help="Path to predictions CSV (columns: True_Label, Predicted_Label)")
    parser.add_argument("--results_dir", type=str, default="../results",
                        help="Directory to save evaluation artifacts")
    args = parser.parse_args()
    run_evaluation(args.pred_csv, args.results_dir)


