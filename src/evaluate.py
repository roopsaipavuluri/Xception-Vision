"""
Xceptional Vision - Model Evaluation & Diagnostic Performance Module
===================================================================
Evaluates the trained Xception model on the held-out test dataset:
  - Computes global Accuracy, Precision, Recall, and F1-Score (Macro, Weighted, Micro)
  - Computes per-class Precision, Recall, F1-Score, and Support
  - Generates and plots raw and normalized Confusion Matrix heatmaps (Seaborn/Matplotlib)
  - Exports comprehensive metrics reports to results/metrics/
  - Exports visual confusion matrices and tables to results/confusion_matrix/
  - Generates sample test CT scan prediction grids with confidence scores

DISCLAIMER:
-----------
This evaluation pipeline and model are developed strictly for educational and research
purposes in computer vision and biomedical image analysis. This software is NOT a certified
medical diagnostic tool and must NOT be used for clinical decision-making or diagnosis.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import tensorflow as tf

# Ensure src root is in module path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import (
    VALID_IMAGE_EXTENSIONS,
    XCEPTION_INPUT_SHAPE,
    load_image,
    preprocess_image
)
from src.train import (
    CTScanDataGenerator,
    scan_dataset_split,
    plot_training_history_curves
)


def load_evaluation_model(model_path: str = "models/xception_model.keras") -> tf.keras.Model:
    """
    Loads saved Keras model checkpoint (.keras or .h5) from disk.

    Args:
        model_path (str): Filepath to the model checkpoint.

    Returns:
        tf.keras.Model: Loaded Keras model.

    Raises:
        FileNotFoundError: If the model file cannot be found.
    """
    if not os.path.exists(model_path):
        # Fallback check for alternative model paths
        alternative_paths = [
            "models/xception_model.keras",
            "models/best_xceptional_model.keras",
            "models/final_xceptional_model.keras",
            "models/best_xceptional_model.h5",
            "models/final_xceptional_model.h5"
        ]
        found_path = None
        for p in alternative_paths:
            if os.path.exists(p):
                found_path = p
                break

        if found_path is None:
            raise FileNotFoundError(
                f"Model checkpoint not found at: '{model_path}'. "
                f"Please ensure you have executed the training pipeline (python src/train.py) "
                f"before running evaluation."
            )
        model_path = found_path

    print(f"[Model Loader] Loading model checkpoint from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    y_probs: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes comprehensive multi-class classification metrics using scikit-learn:
      - Overall Accuracy
      - Macro, Weighted, and Micro Precision
      - Macro, Weighted, and Micro Recall
      - Macro, Weighted, and Micro F1-score
      - Per-class Precision, Recall, F1-score, and Support
      - Full scikit-learn classification report

    Args:
        y_true (np.ndarray): Ground truth integer class labels (shape: [N]).
        y_pred (np.ndarray): Predicted integer class labels (shape: [N]).
        class_names (List[str]): List of class label strings.
        y_probs (Optional[np.ndarray]): Predicted probability distribution (shape: [N, num_classes]).

    Returns:
        Dict[str, Any]: Structured dictionary containing all computed metrics.
    """
    # 1. Global Aggregate Metrics
    acc = float(accuracy_score(y_true, y_pred))

    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    prec_micro = float(precision_score(y_true, y_pred, average="micro", zero_division=0))

    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_micro = float(recall_score(y_true, y_pred, average="micro", zero_division=0))

    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))

    # 2. Per-Class Metrics
    per_class_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Class support counts
    unique_labels, counts = np.unique(y_true, return_counts=True)
    support_dict = dict(zip(unique_labels, counts))

    per_class_metrics = {}
    for i, cls_name in enumerate(class_names):
        supp = int(support_dict.get(i, 0))
        per_class_metrics[cls_name] = {
            "precision": float(per_class_prec[i]) if i < len(per_class_prec) else 0.0,
            "recall": float(per_class_rec[i]) if i < len(per_class_rec) else 0.0,
            "f1_score": float(per_class_f1[i]) if i < len(per_class_f1) else 0.0,
            "support": supp
        }

    # 3. Scikit-learn Text & Dict Classification Reports
    clf_report_text = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    clf_report_dict = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # 4. Confusion Matrix Raw Array
    cm_array = confusion_matrix(y_true, y_pred)

    metrics_result = {
        "overall": {
            "accuracy": acc,
            "precision_macro": prec_macro,
            "precision_weighted": prec_weighted,
            "precision_micro": prec_micro,
            "recall_macro": rec_macro,
            "recall_weighted": rec_weighted,
            "recall_micro": rec_micro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "f1_micro": f1_micro,
            "total_samples": int(len(y_true))
        },
        "per_class": per_class_metrics,
        "classification_report_text": clf_report_text,
        "classification_report_dict": clf_report_dict,
        "confusion_matrix": cm_array.tolist()
    }

    return metrics_result


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_dir: str = "results/confusion_matrix"
) -> Dict[str, str]:
    """
    Generates and saves publication-quality confusion matrix heatmaps:
      1. Raw Sample Counts Heatmap -> results/confusion_matrix/confusion_matrix_raw.png
      2. Normalized (Recall / True Class %) Heatmap -> results/confusion_matrix/confusion_matrix_normalized.png
      3. Dual-Panel Combined Heatmap -> results/confusion_matrix/confusion_matrix_combined.png
      4. Exports CSV data tables of confusion matrices.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        class_names (List[str]): List of class names.
        save_dir (str): Directory where plots and tables will be saved.

    Returns:
        Dict[str, str]: Filepaths of generated plots and tables.
    """
    os.makedirs(save_dir, exist_ok=True)

    cm_raw = confusion_matrix(y_true, y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    clean_class_names = [c.replace("_", " ").title() for c in class_names]

    # Export CSV tables
    df_raw = pd.DataFrame(cm_raw, index=clean_class_names, columns=clean_class_names)
    raw_csv_path = os.path.join(save_dir, "confusion_matrix_raw.csv")
    df_raw.to_csv(raw_csv_path)

    df_norm = pd.DataFrame(cm_norm, index=clean_class_names, columns=clean_class_names)
    norm_csv_path = os.path.join(save_dir, "confusion_matrix_normalized.csv")
    df_norm.to_csv(norm_csv_path)

    # -------------------------------------------------------------
    # 1. Raw Counts Confusion Matrix Heatmap
    # -------------------------------------------------------------
    raw_plot_path = os.path.join(save_dir, "confusion_matrix_raw.png")
    plt.figure(figsize=(8.5, 7.0), dpi=300)
    sns.heatmap(
        cm_raw,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=clean_class_names,
        yticklabels=clean_class_names,
        cbar=True,
        linewidths=1.0,
        linecolor="#f0f0f0",
        annot_kws={"size": 12, "weight": "bold"}
    )
    plt.title("Xception Model: Test Confusion Matrix (Raw Sample Counts)", fontsize=13, fontweight="bold", pad=14)
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold", labelpad=10)
    plt.ylabel("Ground Truth Class", fontsize=11, fontweight="bold", labelpad=10)
    plt.xticks(rotation=20, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(raw_plot_path, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # 2. Normalized Percentage Confusion Matrix Heatmap
    # -------------------------------------------------------------
    norm_plot_path = os.path.join(save_dir, "confusion_matrix_normalized.png")
    plt.figure(figsize=(8.5, 7.0), dpi=300)
    sns.heatmap(
        cm_norm * 100.0,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        xticklabels=clean_class_names,
        yticklabels=clean_class_names,
        cbar=True,
        linewidths=1.0,
        linecolor="#f0f0f0",
        annot_kws={"size": 12, "weight": "bold"}
    )
    plt.title("Xception Model: Test Confusion Matrix (Normalized Percentages %)", fontsize=13, fontweight="bold", pad=14)
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold", labelpad=10)
    plt.ylabel("Ground Truth Class", fontsize=11, fontweight="bold", labelpad=10)
    plt.xticks(rotation=20, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(norm_plot_path, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # 3. Dual-Panel Combined Confusion Matrix Heatmap
    # -------------------------------------------------------------
    comb_plot_path = os.path.join(save_dir, "confusion_matrix_combined.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7.0), dpi=300)

    # Panel 1: Raw
    sns.heatmap(
        cm_raw,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=clean_class_names,
        yticklabels=clean_class_names,
        cbar=True,
        ax=ax1,
        linewidths=1.0,
        linecolor="#e8e8e8",
        annot_kws={"size": 11, "weight": "bold"}
    )
    ax1.set_title("A. Absolute Sample Counts", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Predicted Class", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Ground Truth Class", fontsize=10, fontweight="bold")
    ax1.set_xticklabels(clean_class_names, rotation=25, ha="right")
    ax1.set_yticklabels(clean_class_names, rotation=0)

    # Panel 2: Normalized
    sns.heatmap(
        cm_norm * 100.0,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        xticklabels=clean_class_names,
        yticklabels=clean_class_names,
        cbar=True,
        ax=ax2,
        linewidths=1.0,
        linecolor="#e8e8e8",
        annot_kws={"size": 11, "weight": "bold"}
    )
    ax2.set_title("B. Normalized Per-Class Proportions (%)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Predicted Class", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Ground Truth Class", fontsize=10, fontweight="bold")
    ax2.set_xticklabels(clean_class_names, rotation=25, ha="right")
    ax2.set_yticklabels(clean_class_names, rotation=0)

    plt.suptitle("Xceptional Vision: Test Set Confusion Matrix Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    std_cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(comb_plot_path, bbox_inches="tight")
    plt.savefig(std_cm_path, bbox_inches="tight")
    plt.close()

    print(f"✓ Confusion Matrix plots saved:")
    print(f"  • Standard:     {std_cm_path}")
    print(f"  • Raw Counts:   {raw_plot_path}")
    print(f"  • Normalized:   {norm_plot_path}")
    print(f"  • Dual-Panel:   {comb_plot_path}")

    return {
        "standard_plot": std_cm_path,
        "raw_plot": raw_plot_path,
        "norm_plot": norm_plot_path,
        "combined_plot": comb_plot_path,
        "raw_csv": raw_csv_path,
        "norm_csv": norm_csv_path
    }


def save_metrics_reports(
    metrics_result: Dict[str, Any],
    class_names: List[str],
    save_dir: str = "results/metrics"
) -> Dict[str, str]:
    """
    Saves comprehensive metrics reports to text, JSON, and CSV formats.

    Args:
        metrics_result (Dict[str, Any]): Dictionary of computed metrics.
        class_names (List[str]): List of class names.
        save_dir (str): Directory where reports will be saved.

    Returns:
        Dict[str, str]: Map of saved filepaths.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Save Full Metrics Summary JSON
    json_path = os.path.join(save_dir, "test_metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump(metrics_result, f, indent=4)

    # 2. Save Per-Class Metrics Table CSV
    per_class_rows = []
    for cls_name, vals in metrics_result["per_class"].items():
        per_class_rows.append({
            "Class Name": cls_name.replace("_", " ").title(),
            "Precision": f"{vals['precision']:.4f}",
            "Recall": f"{vals['recall']:.4f}",
            "F1-Score": f"{vals['f1_score']:.4f}",
            "Support (Samples)": vals["support"]
        })
    df_per_class = pd.DataFrame(per_class_rows)
    per_class_csv_path = os.path.join(save_dir, "per_class_metrics.csv")
    df_per_class.to_csv(per_class_csv_path, index=False)

    # 3. Save Overall Summary CSV
    overall = metrics_result["overall"]
    df_overall = pd.DataFrame([{
        "Overall Accuracy": f"{overall['accuracy'] * 100:.2f}%",
        "Macro Precision": f"{overall['precision_macro'] * 100:.2f}%",
        "Weighted Precision": f"{overall['precision_weighted'] * 100:.2f}%",
        "Macro Recall": f"{overall['recall_macro'] * 100:.2f}%",
        "Weighted Recall": f"{overall['recall_weighted'] * 100:.2f}%",
        "Macro F1-Score": f"{overall['f1_macro'] * 100:.2f}%",
        "Weighted F1-Score": f"{overall['f1_weighted'] * 100:.2f}%",
        "Total Test Samples": overall["total_samples"]
    }])
    overall_csv_path = os.path.join(save_dir, "overall_test_metrics.csv")
    df_overall.to_csv(overall_csv_path, index=False)

    # 4. Save Human-Readable Text Report
    txt_path = os.path.join(save_dir, "evaluation_report.txt")
    std_txt_path = os.path.join(save_dir, "evaluation.txt")
    
    report_content = (
        "=" * 80 + "\n" +
        "XCEPTIONAL VISION - TEST DATASET EVALUATION REPORT\n" +
        "=" * 80 + "\n\n" +
        "DISCLAIMER: Educational/Research biomedical imaging project.\n" +
        "Not for medical diagnosis or clinical use.\n\n" +
        "-" * 80 + "\n" +
        "1. GLOBAL TEST PERFORMANCE SUMMARY\n" +
        "-" * 80 + "\n" +
        f"  • Overall Test Accuracy:     {overall['accuracy'] * 100:6.2f}%\n" +
        f"  • Macro-Average Precision:   {overall['precision_macro'] * 100:6.2f}%\n" +
        f"  • Weighted-Average Precision:{overall['precision_weighted'] * 100:6.2f}%\n" +
        f"  • Macro-Average Recall:      {overall['recall_macro'] * 100:6.2f}%\n" +
        f"  • Weighted-Average Recall:   {overall['recall_weighted'] * 100:6.2f}%\n" +
        f"  • Macro-Average F1-Score:    {overall['f1_macro'] * 100:6.2f}%\n" +
        f"  • Weighted-Average F1-Score: {overall['f1_weighted'] * 100:6.2f}%\n" +
        f"  • Total Evaluated Samples:   {overall['total_samples']:6d}\n\n" +
        "-" * 80 + "\n" +
        "2. PER-CLASS CLASSIFICATION BREAKDOWN\n" +
        "-" * 80 + "\n" +
        f"{'Class Name':28s} {'Precision':10s} {'Recall':10s} {'F1-Score':10s} {'Support':8s}\n" +
        "-" * 80 + "\n"
    )
    for row in per_class_rows:
        report_content += f"{row['Class Name']:28s} {row['Precision']:10s} {row['Recall']:10s} {row['F1-Score']:10s} {str(row['Support (Samples)']):8s}\n"
    report_content += (
        "\n" +
        "-" * 80 + "\n" +
        "3. SCIKIT-LEARN DETAILED CLASSIFICATION REPORT\n" +
        "-" * 80 + "\n" +
        metrics_result["classification_report_text"] + "\n" +
        "=" * 80 + "\n"
    )

    with open(txt_path, "w") as f:
        f.write(report_content)
    with open(std_txt_path, "w") as f:
        f.write(report_content)

    print(f"✓ Metrics reports saved:")
    print(f"  • Text Report:   {std_txt_path}")
    print(f"  • Summary JSON:  {json_path}")
    print(f"  • Per-Class CSV: {per_class_csv_path}")
    print(f"  • Overall CSV:   {overall_csv_path}")

    return {
        "text_report": std_txt_path,
        "full_report": txt_path,
        "summary_json": json_path,
        "per_class_csv": per_class_csv_path,
        "overall_csv": overall_csv_path
    }


def visualize_sample_predictions(
    model: tf.keras.Model,
    test_paths: List[str],
    test_labels: List[int],
    class_names: List[str],
    save_path: str = "results/graphs/sample_predictions.png",
    num_samples: int = 9
) -> str:
    """
    Visualizes a grid of sample test CT images with ground truth, predicted class,
    and confidence scores. Saves to results/graphs/sample_predictions.png.

    Args:
        model (tf.keras.Model): Trained Keras model.
        test_paths (List[str]): List of test image filepaths.
        test_labels (List[int]): Ground truth class indices.
        class_names (List[str]): Class name strings.
        save_path (str): Filepath for output image.
        num_samples (int): Number of sample images to display (default: 9).

    Returns:
        str: Filepath to saved image grid.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Pick representative samples evenly across classes
    selected_indices: List[int] = []
    classes_present = np.unique(test_labels)
    samples_per_class = max(1, num_samples // len(classes_present))

    for cls_idx in classes_present:
        cls_indices = [i for i, lbl in enumerate(test_labels) if lbl == cls_idx]
        selected_indices.extend(cls_indices[:samples_per_class])

    # If we need more samples to fill the grid, take from start
    if len(selected_indices) < num_samples:
        for idx in range(len(test_paths)):
            if idx not in selected_indices:
                selected_indices.append(idx)
            if len(selected_indices) >= num_samples:
                break

    selected_indices = selected_indices[:num_samples]

    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.4 * rows), dpi=300)
    axes = axes.flatten()

    for i, idx in enumerate(selected_indices):
        img_path = test_paths[idx]
        true_label = test_labels[idx]
        true_name = class_names[true_label].replace("_", " ").title()

        # Load and preprocess image
        processed_img = preprocess_image(img_path, target_size=XCEPTION_INPUT_SHAPE, return_normalized=False)
        norm_img = (processed_img.astype(np.float32) / 127.5) - 1.0
        batch_input = np.expand_dims(norm_img, axis=0)

        # Predict
        probs = model.predict(batch_input, verbose=0)[0]
        pred_label = int(np.argmax(probs))
        pred_name = class_names[pred_label].replace("_", " ").title()
        confidence = float(probs[pred_label])

        is_correct = (pred_label == true_label)
        color = "#1b5e20" if is_correct else "#b71c1c"
        status_symbol = "✓" if is_correct else "✗"

        axes[i].imshow(processed_img)
        title_text = (
            f"True: {true_name}\n"
            f"Pred: {pred_name} ({confidence * 100:.1f}%) {status_symbol}"
        )
        axes[i].set_title(title_text, fontsize=10, fontweight="bold", color=color, pad=8)
        axes[i].axis("off")

    # Hide extra unused subplots if any
    for j in range(len(selected_indices), len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        "Xceptional Vision: Sample Test CT Scan Predictions & Confidence Scores",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"✓ Sample predictions grid saved to: {save_path}")
    return save_path


def evaluate_pipeline(
    dataset_root: str = "dataset",
    model_path: str = "models/best_xceptional_model.keras",
    batch_size: int = 16
) -> Dict[str, Any]:
    """
    Executes the complete test evaluation pipeline on held-out test data:
      1. Loads test images and labels.
      2. Loads trained model checkpoint.
      3. Performs batch inference.
      4. Computes all scikit-learn metrics.
      5. Prints formatted results to terminal.
      6. Generates confusion matrices and visualization charts.
      7. Saves all reports to disk.

    Args:
        dataset_root (str): Root dataset folder.
        model_path (str): Filepath to trained model.
        batch_size (int): Inference batch size.

    Returns:
        Dict[str, Any]: Complete evaluation metrics dictionary.
    """
    print("=" * 80)
    print("XCEPTIONAL VISION: MODEL TEST EVALUATION PIPELINE")
    print("=" * 80)

    # 1. Scan Test Split
    test_dir = os.path.join(dataset_root, "test")
    test_paths, test_labels, class_names = scan_dataset_split(test_dir)
    print(f"[Dataset Audit] Test images found: {len(test_paths)} across {len(class_names)} classes.")

    # 2. Build Test Generator
    test_gen = CTScanDataGenerator(
        image_paths=test_paths,
        labels=test_labels,
        num_classes=len(class_names),
        batch_size=batch_size,
        target_size=XCEPTION_INPUT_SHAPE,
        augment=False,
        shuffle=False
    )

    # 3. Load Trained Model
    model = load_evaluation_model(model_path)

    # 4. Run Model Predictions
    print("\n[Inference] Executing batch predictions across test dataset...")
    y_probs = model.predict(test_gen, verbose=1)

    if len(class_names) == 1:
        y_pred = (y_probs > 0.5).astype(int).flatten()
    else:
        y_pred = np.argmax(y_probs, axis=1)

    y_true = np.array(test_labels[:len(y_pred)])

    # 5. Compute Metrics
    print("\n[Metrics] Computing scikit-learn classification performance metrics...")
    metrics_result = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        y_probs=y_probs
    )

    # 6. Display Terminal Output
    overall = metrics_result["overall"]
    print("\n" + "=" * 80)
    print("ACTUAL TEST EVALUATION METRICS")
    print("=" * 80)
    print(f"  • Total Test Samples:       {overall['total_samples']:4d}")
    print(f"  • Overall Test Accuracy:    {overall['accuracy'] * 100:6.2f}%")
    print(f"  • Macro Precision:          {overall['precision_macro'] * 100:6.2f}%")
    print(f"  • Weighted Precision:       {overall['precision_weighted'] * 100:6.2f}%")
    print(f"  • Macro Recall:             {overall['recall_macro'] * 100:6.2f}%")
    print(f"  • Weighted Recall:          {overall['recall_weighted'] * 100:6.2f}%")
    print(f"  • Macro F1-Score:           {overall['f1_macro'] * 100:6.2f}%")
    print(f"  • Weighted F1-Score:        {overall['f1_weighted'] * 100:6.2f}%")
    print("-" * 80)
    print("PER-CLASS METRICS BREAKDOWN:")
    print(f"  {'Class':28s} {'Precision':10s} {'Recall':10s} {'F1-Score':10s} {'Support':8s}")
    print("  " + "-" * 70)
    for cls_name, vals in metrics_result["per_class"].items():
        clean_name = cls_name.replace("_", " ").title()
        print(f"  {clean_name:28s} {vals['precision'] * 100:6.2f}%    {vals['recall'] * 100:6.2f}%    {vals['f1_score'] * 100:6.2f}%    {vals['support']:4d}")
    print("=" * 80)
    print("\nSCIKIT-LEARN CLASSIFICATION REPORT:\n")
    print(metrics_result["classification_report_text"])
    print("=" * 80)

    # 7. Generate Visualizations & Save Reports
    print("\n[Artifact Generation] Generating confusion matrix plots and metric reports...")
    plot_confusion_matrix(y_true, y_pred, class_names, save_dir="results/confusion_matrix")
    save_metrics_reports(metrics_result, class_names, save_dir="results/metrics")
    visualize_sample_predictions(
        model=model,
        test_paths=test_paths,
        test_labels=test_labels,
        class_names=class_names,
        save_path="results/graphs/sample_predictions.png",
        num_samples=9
    )

    print("\n[Success] Test evaluation completed and all artifacts successfully saved!")
    return metrics_result


def main():
    """Main execution point for model evaluation."""
    evaluate_pipeline(
        dataset_root="dataset",
        model_path="models/best_xceptional_model.keras",
        batch_size=16
    )


if __name__ == "__main__":
    main()
