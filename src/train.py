"""
Xceptional Vision - Complete Model Training Pipeline
====================================================
Orchestrates end-to-end model training for exactly 105 epochs using:
  - Clinical image preprocessing (Bilateral Denoising + CLAHE Contrast + [-1, 1] Normalization)
  - Medically valid data augmentation (rotations, zoom, shifts, horizontal flip)
  - Pre-trained Xception transfer learning model
  - Callbacks: ModelCheckpoint (best model), ReduceLROnPlateau, CSVLogger
  - Audit logging of honest metrics (accuracy, loss, precision, recall, AUC)
  - Publication-quality visualization of loss and accuracy curves saved to results/graphs/

DISCLAIMER:
-----------
This model and training pipeline are developed strictly for educational and research purposes
in computer vision and biomedical image analysis. This software is NOT a certified medical
diagnostic tool and must NOT be used for clinical decision-making or diagnosis.
"""

import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

# Ensure src root is in module path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.augmentation import MedicallyValidAugmentor
from src.model import (
    DEFAULT_INPUT_SHAPE,
    DEFAULT_NUM_CLASSES,
    DEFAULT_LEARNING_RATE,
    DEFAULT_DROPOUT_RATE,
    DEFAULT_DENSE_UNITS,
    build_xceptional_model,
    compile_model,
    display_model_summary
)
from src.preprocessing import (
    VALID_IMAGE_EXTENSIONS,
    XCEPTION_INPUT_SHAPE,
    load_image,
    preprocess_image
)


# Default Training Hyperparameters
RANDOM_SEED: int = 42
TOTAL_EPOCHS: int = 105
BATCH_SIZE: int = 16
DEFAULT_LR_PATIENCE: int = 6
DEFAULT_LR_FACTOR: float = 0.5
DEFAULT_MIN_LR: float = 1e-7


def set_reproducibility_seeds(seed: int = RANDOM_SEED) -> None:
    """
    Sets global deterministic random seeds across Python, NumPy, and TensorFlow
    for experimental reproducibility.

    Args:
        seed (int): The integer random seed. Default: 42.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"[Reproducibility] Global random seeds locked to: {seed}")


class CTScanDataGenerator(tf.keras.utils.Sequence):
    """
    Custom Keras Sequence generator for batchwise loading, preprocessing,
    and medically-constrained augmentation of CT scan slices.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        num_classes: int = DEFAULT_NUM_CLASSES,
        batch_size: int = BATCH_SIZE,
        target_size: Tuple[int, int] = XCEPTION_INPUT_SHAPE,
        augment: bool = False,
        shuffle: bool = True,
        seed: int = RANDOM_SEED
    ) -> None:
        self.image_paths = np.array(image_paths)
        self.labels = np.array(labels)
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augment
        self.shuffle = shuffle
        self.augmentor = MedicallyValidAugmentor(seed=seed) if augment else None
        self.indexes = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self) -> int:
        """Denotes the number of batches per epoch."""
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate one batch of data."""
        # Generate indexes of the batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_paths = self.image_paths[batch_indexes]
        batch_labels = self.labels[batch_indexes]

        # Allocate batch memory
        x_batch = np.empty((len(batch_paths), *self.target_size, 3), dtype=np.float32)
        y_batch = np.zeros((len(batch_paths), self.num_classes), dtype=np.float32)

        for i, (img_path, label) in enumerate(zip(batch_paths, batch_labels)):
            try:
                # 1. Preprocess: Denoise + CLAHE + Resize (returns uint8 in [0, 255])
                processed_img = preprocess_image(
                    image_path=img_path,
                    target_size=self.target_size,
                    return_normalized=False
                )

                # 2. Medically-constrained data augmentation (training only)
                if self.augment and self.augmentor is not None:
                    processed_img = self.augmentor.augment(processed_img)

                # 3. Standard Xception Normalization to [-1.0, 1.0]
                norm_img = (processed_img.astype(np.float32) / 127.5) - 1.0
                x_batch[i] = norm_img

                # 4. One-hot encode label
                if self.num_classes == 1:
                    y_batch[i] = float(label)
                else:
                    y_batch[i, label] = 1.0

            except Exception as err:
                print(f"⚠️ Error loading image {img_path}: {err}. Using zero tensor.")
                x_batch[i] = np.zeros((*self.target_size, 3), dtype=np.float32)
                if self.num_classes > 1:
                    y_batch[i, label] = 1.0

        return x_batch, y_batch

    def on_epoch_end(self) -> None:
        """Updates indexes after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indexes)


def scan_dataset_split(split_dir: str) -> Tuple[List[str], List[int], List[str]]:
    """
    Collects image filepaths and corresponding class indices for a split folder.

    Args:
        split_dir (str): Path to split directory (e.g., 'dataset/train').

    Returns:
        Tuple[List[str], List[int], List[str]]: (image_paths, labels, class_names)
    """
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    class_names = sorted([
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d)) and not d.startswith(".")
    ])

    image_paths: List[str] = []
    labels: List[int] = []

    for class_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(split_dir, class_name)
        for fname in sorted(os.listdir(class_folder)):
            if fname.startswith("."):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in VALID_IMAGE_EXTENSIONS:
                image_paths.append(os.path.join(class_folder, fname))
                labels.append(class_idx)

    return image_paths, labels, class_names


def create_datasets(
    dataset_root: str = "dataset",
    batch_size: int = BATCH_SIZE,
    target_size: Tuple[int, int] = XCEPTION_INPUT_SHAPE,
    seed: int = RANDOM_SEED
) -> Tuple[CTScanDataGenerator, CTScanDataGenerator, CTScanDataGenerator, List[str], Optional[Dict[int, float]]]:
    """
    Builds training, validation, and test data generators from dataset splits.

    Args:
        dataset_root (str): Root dataset folder containing train/, validation/, and test/.
        batch_size (int): Mini-batch size. Default: 16.
        target_size (Tuple[int, int]): Image resolution (width, height). Default: (299, 299).
        seed (int): Seed for augmentation and shuffling.

    Returns:
        Tuple: (train_generator, val_generator, test_generator, class_names, class_weights)
    """
    train_dir = os.path.join(dataset_root, "train")
    val_dir = os.path.join(dataset_root, "validation")
    test_dir = os.path.join(dataset_root, "test")

    train_paths, train_labels, class_names = scan_dataset_split(train_dir)
    val_paths, val_labels, _ = scan_dataset_split(val_dir)
    test_paths, test_labels, _ = scan_dataset_split(test_dir)

    num_classes = len(class_names)
    print(f"\n[Dataset Summary]")
    print(f"  • Target Classes ({num_classes}): {class_names}")
    print(f"  • Training Samples:   {len(train_paths):4d} images")
    print(f"  • Validation Samples: {len(val_paths):4d} images")
    print(f"  • Test Samples:       {len(test_paths):4d} images")

    # Compute balanced class weights to handle any minor class distribution skew
    try:
        raw_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_labels),
            y=np.array(train_labels)
        )
        class_weight_dict = {i: float(w) for i, w in enumerate(raw_weights)}
        print(f"  • Class Weights:      {class_weight_dict}")
    except Exception:
        class_weight_dict = None

    train_gen = CTScanDataGenerator(
        image_paths=train_paths,
        labels=train_labels,
        num_classes=num_classes,
        batch_size=batch_size,
        target_size=target_size,
        augment=True,
        shuffle=True,
        seed=seed
    )

    val_gen = CTScanDataGenerator(
        image_paths=val_paths,
        labels=val_labels,
        num_classes=num_classes,
        batch_size=batch_size,
        target_size=target_size,
        augment=False,
        shuffle=False,
        seed=seed
    )

    test_gen = CTScanDataGenerator(
        image_paths=test_paths,
        labels=test_labels,
        num_classes=num_classes,
        batch_size=batch_size,
        target_size=target_size,
        augment=False,
        shuffle=False,
        seed=seed
    )

    return train_gen, val_gen, test_gen, class_names, class_weight_dict


def get_callbacks(
    checkpoint_path: str = "models/best_xceptional_model.keras",
    log_csv_path: str = "results/metrics/training_history.csv",
    monitor: str = "val_loss",
    lr_patience: int = DEFAULT_LR_PATIENCE,
    lr_factor: float = DEFAULT_LR_FACTOR,
    min_lr: float = DEFAULT_MIN_LR
) -> List[tf.keras.callbacks.Callback]:
    """
    Instantiates training callbacks:
      1. ModelCheckpoint: Saves the best model checkpoint based on validation loss.
      2. ReduceLROnPlateau: Dynamically decays learning rate when validation loss plateaus.
      3. CSVLogger: Streams epoch metrics to a persistent CSV audit log.

    Note on EarlyStopping:
      EarlyStopping is intentionally omitted here so that the full 105-epoch training
      experiment runs completely and documents the comprehensive convergence behavior.

    Args:
        checkpoint_path (str): Filepath for best model checkpoint.
        log_csv_path (str): Filepath for training log CSV.
        monitor (str): Metric to monitor. Default: 'val_loss'.
        lr_patience (int): Epochs to wait before reducing LR. Default: 6.
        lr_factor (float): Factor by which LR is multiplied on plateau. Default: 0.5.
        min_lr (float): Lower bound for learning rate. Default: 1e-7.

    Returns:
        List[tf.keras.callbacks.Callback]: Configured Keras callbacks.
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)

    mode = "min" if "loss" in monitor else "max"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor,
            save_best_only=True,
            mode=mode,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=lr_factor,
            patience=lr_patience,
            min_lr=min_lr,
            mode=mode,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            filename=log_csv_path,
            separator=",",
            append=False
        )
    ]
    return callbacks


def plot_training_history_curves(
    history_source: Union[Dict[str, List[float]], str, pd.DataFrame],
    save_dir: str = "results/graphs"
) -> Tuple[str, str, str]:
    """
    Generates publication-quality graphs for:
      1. Training vs Validation Accuracy (105 Epochs) -> accuracy_curve.png
      2. Training vs Validation Loss (105 Epochs) -> loss_curve.png
      3. Dual-Panel Combined Training History -> training_history_curves.png

    Args:
        history_source: History dict, CSV path, or DataFrame.
        save_dir (str): Directory where graphs will be saved.

    Returns:
        Tuple[str, str, str]: Paths to (accuracy_curve_path, loss_curve_path, combined_curve_path).
    """
    os.makedirs(save_dir, exist_ok=True)

    # Convert history_source to standard dictionary
    if isinstance(history_source, str):
        df = pd.read_csv(history_source)
        hist = df.to_dict(orient="list")
    elif isinstance(history_source, pd.DataFrame):
        hist = history_source.to_dict(orient="list")
    elif hasattr(history_source, "history"):
        hist = history_source.history
    else:
        hist = dict(history_source)

    # Standardize metric key lookups
    acc_key = "accuracy" if "accuracy" in hist else "categorical_accuracy"
    val_acc_key = "val_accuracy" if "val_accuracy" in hist else "val_categorical_accuracy"
    loss_key = "loss"
    val_loss_key = "val_loss"

    train_acc = hist.get(acc_key, [])
    val_acc = hist.get(val_acc_key, [])
    train_loss = hist.get(loss_key, [])
    val_loss = hist.get(val_loss_key, [])

    total_epochs = len(train_acc)
    epochs_range = np.arange(1, total_epochs + 1)

    # Global chart styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # -------------------------------------------------------------
    # 1. Training vs Validation Accuracy Graph
    # -------------------------------------------------------------
    acc_path = os.path.join(save_dir, "accuracy_curve.png")
    acc_std_path = os.path.join(save_dir, "accuracy.png")
    plt.figure(figsize=(9, 6), dpi=300)
    plt.plot(epochs_range, train_acc, color="#1f77b4", linewidth=2.2, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, color="#ff7f0e", linewidth=2.2, linestyle="--", label="Validation Accuracy")

    if len(val_acc) > 0:
        best_val_acc_idx = int(np.argmax(val_acc))
        best_val_acc = val_acc[best_val_acc_idx]
        plt.scatter([best_val_acc_idx + 1], [best_val_acc], color="#2ca02c", s=100, zorder=5)
        plt.annotate(
            f"Best Val Acc: {best_val_acc:.4f}\n(Epoch {best_val_acc_idx + 1})",
            xy=(best_val_acc_idx + 1, best_val_acc),
            xytext=(best_val_acc_idx + 1 - 15, best_val_acc - 0.08),
            arrowprops=dict(facecolor="#2ca02c", shrink=0.05, width=1.5, headwidth=6),
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#eafaf1", ec="#2ca02c", lw=1)
        )

    plt.title(f"Xception Model: Training vs. Validation Accuracy ({total_epochs} Epochs)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=11, fontweight="bold")
    plt.ylabel("Accuracy Score", fontsize=11, fontweight="bold")
    plt.ylim(0.0, 1.02)
    plt.xlim(1, total_epochs)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(acc_path, bbox_inches="tight")
    plt.savefig(acc_std_path, bbox_inches="tight")
    plt.close()
    print(f"✓ Accuracy curve saved to: {acc_std_path}")

    # -------------------------------------------------------------
    # 2. Training vs Validation Loss Graph
    # -------------------------------------------------------------
    loss_path = os.path.join(save_dir, "loss_curve.png")
    loss_std_path = os.path.join(save_dir, "loss.png")
    plt.figure(figsize=(9, 6), dpi=300)
    plt.plot(epochs_range, train_loss, color="#d62728", linewidth=2.2, label="Training Loss")
    plt.plot(epochs_range, val_loss, color="#9467bd", linewidth=2.2, linestyle="--", label="Validation Loss")

    if len(val_loss) > 0:
        min_val_loss_idx = int(np.argmin(val_loss))
        min_val_loss = val_loss[min_val_loss_idx]
        plt.scatter([min_val_loss_idx + 1], [min_val_loss], color="#17becf", s=100, zorder=5)
        plt.annotate(
            f"Min Val Loss: {min_val_loss:.4f}\n(Epoch {min_val_loss_idx + 1})",
            xy=(min_val_loss_idx + 1, min_val_loss),
            xytext=(min_val_loss_idx + 1 + 5, min_val_loss + 0.15),
            arrowprops=dict(facecolor="#17becf", shrink=0.05, width=1.5, headwidth=6),
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#e8f8f5", ec="#17becf", lw=1)
        )

    plt.title(f"Xception Model: Training vs. Validation Loss ({total_epochs} Epochs)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=11, fontweight="bold")
    plt.ylabel("Loss (Categorical Crossentropy)", fontsize=11, fontweight="bold")
    plt.xlim(1, total_epochs)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(loss_path, bbox_inches="tight")
    plt.savefig(loss_std_path, bbox_inches="tight")
    plt.close()
    print(f"✓ Loss curve saved to: {loss_std_path}")

    # -------------------------------------------------------------
    # 3. Dual-Panel Combined History Overview
    # -------------------------------------------------------------
    comb_path = os.path.join(save_dir, "training_history_curves.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # Left: Accuracy
    ax1.plot(epochs_range, train_acc, color="#1f77b4", linewidth=2.0, label="Training Accuracy")
    ax1.plot(epochs_range, val_acc, color="#ff7f0e", linewidth=2.0, linestyle="--", label="Validation Accuracy")
    ax1.set_title("Classification Accuracy per Epoch", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Accuracy", fontsize=10, fontweight="bold")
    ax1.set_ylim(0.0, 1.02)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", frameon=True)

    # Right: Loss
    ax2.plot(epochs_range, train_loss, color="#d62728", linewidth=2.0, label="Training Loss")
    ax2.plot(epochs_range, val_loss, color="#9467bd", linewidth=2.0, linestyle="--", label="Validation Loss")
    ax2.set_title("Categorical Crossentropy Loss per Epoch", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Loss", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True)

    plt.suptitle(
        f"Xceptional Vision: Comprehensive 105-Epoch Training & Validation Trajectory",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(comb_path, bbox_inches="tight")
    plt.close()
    print(f"✓ Combined curves saved to: {comb_path}")

    return acc_path, loss_path, comb_path


def train_model(
    model: tf.keras.Model,
    train_data: CTScanDataGenerator,
    val_data: CTScanDataGenerator,
    epochs: int = TOTAL_EPOCHS,
    callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    class_weights: Optional[Dict[int, float]] = None,
    best_model_path: str = "models/best_xceptional_model.keras",
    final_model_path: str = "models/final_xceptional_model.keras",
    log_json_path: str = "results/metrics/training_history.json"
) -> Dict[str, List[float]]:
    """
    Executes model training for exactly 105 epochs, honestly recording and reporting
    metrics without artificial manipulation. Saves the best model, final model, and curves.

    Args:
        model (tf.keras.Model): The compiled Xception model.
        train_data (CTScanDataGenerator): Training generator.
        val_data (CTScanDataGenerator): Validation generator.
        epochs (int): Number of epochs (default: 105).
        callbacks (Optional[List]): Training callbacks.
        class_weights (Optional[Dict]): Class weighting dictionary.
        best_model_path (str): Destination for best checkpoint.
        final_model_path (str): Destination for final epoch model.
        log_json_path (str): Destination for JSON history summary.

    Returns:
        Dict[str, List[float]]: Training history metrics dictionary.
    """
    print("=" * 80)
    print(f"INITIATING XCEPTION TRAINING FOR EXACTLY {epochs} EPOCHS")
    print("=" * 80)
    print(f"• Training Batches per Epoch:   {len(train_data)}")
    print(f"• Validation Batches per Epoch: {len(val_data)}")
    print(f"• Checkpoint Best Target:       {best_model_path}")
    print(f"• Final Model Target:           {final_model_path}")
    print("=" * 80)

    # Execute training
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Save final model state
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    model.save(final_model_path)
    primary_model_path = os.path.join(os.path.dirname(final_model_path), "xception_model.keras")
    model.save(primary_model_path)
    print(f"\n✓ Saved model checkpoint to: {primary_model_path}")
    print(f"✓ Saved final model to: {final_model_path}")

    # Export history to JSON
    os.makedirs(os.path.dirname(log_json_path), exist_ok=True)
    serializable_history = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(log_json_path, "w") as f:
        json.dump(serializable_history, f, indent=4)
    print(f"✓ Saved training history JSON to: {log_json_path}")

    # Generate and save curves
    plot_training_history_curves(serializable_history, save_dir="results/graphs")

    # Honest metric reporting
    print("\n" + "=" * 80)
    print("HONEST EXPERIMENTAL RESULTS SUMMARY (105 EPOCHS)")
    print("=" * 80)
    final_train_acc = serializable_history["accuracy"][-1]
    final_val_acc = serializable_history["val_accuracy"][-1]
    final_train_loss = serializable_history["loss"][-1]
    final_val_loss = serializable_history["val_loss"][-1]

    best_val_acc = max(serializable_history["val_accuracy"])
    best_val_acc_epoch = serializable_history["val_accuracy"].index(best_val_acc) + 1

    min_val_loss = min(serializable_history["val_loss"])
    min_val_loss_epoch = serializable_history["val_loss"].index(min_val_loss) + 1

    print(f"  • Final Epoch {epochs} Training Accuracy:   {final_train_acc * 100:.2f}%")
    print(f"  • Final Epoch {epochs} Validation Accuracy: {final_val_acc * 100:.2f}%")
    print(f"  • Final Epoch {epochs} Training Loss:       {final_train_loss:.4f}")
    print(f"  • Final Epoch {epochs} Validation Loss:     {final_val_loss:.4f}")
    print("-" * 80)
    print(f"  ★ Peak Validation Accuracy:   {best_val_acc * 100:.2f}% (Epoch {best_val_acc_epoch})")
    print(f"  ★ Minimum Validation Loss:    {min_val_loss:.4f} (Epoch {min_val_loss_epoch})")
    print("=" * 80)

    return serializable_history


def main():
    """
    Main execution routine for the complete training pipeline.
    """
    print("=" * 80)
    print("XCEPTIONAL VISION: DEEP LEARNING TRAINING PIPELINE")
    print("=" * 80)

    # 1. Set reproducibility seeds
    set_reproducibility_seeds(seed=RANDOM_SEED)

    # 2. Build data generators
    print("\n[Step 1/5] Building dataset generators with clinical preprocessing...")
    train_gen, val_gen, test_gen, class_names, class_weights = create_datasets(
        dataset_root="dataset",
        batch_size=BATCH_SIZE,
        target_size=XCEPTION_INPUT_SHAPE,
        seed=RANDOM_SEED
    )

    # 3. Construct and compile model
    print("\n[Step 2/5] Building and compiling Xception transfer learning model...")
    model = build_xceptional_model(
        input_shape=(*XCEPTION_INPUT_SHAPE, 3),
        num_classes=len(class_names),
        freeze_backbone=True,
        dropout_rate=DEFAULT_DROPOUT_RATE,
        dense_units=DEFAULT_DENSE_UNITS
    )

    model = compile_model(
        model=model,
        learning_rate=DEFAULT_LEARNING_RATE,
        num_classes=len(class_names)
    )

    display_model_summary(model, show_explanations=False)

    # 4. Prepare callbacks
    print("\n[Step 3/5] Configuring training callbacks...")
    checkpoint_file = "models/best_xceptional_model.keras"
    log_csv_file = "results/metrics/training_history.csv"
    callbacks = get_callbacks(
        checkpoint_path=checkpoint_file,
        log_csv_path=log_csv_file,
        monitor="val_loss",
        lr_patience=DEFAULT_LR_PATIENCE,
        lr_factor=DEFAULT_LR_FACTOR,
        min_lr=DEFAULT_MIN_LR
    )

    # 5. Execute 105-epoch training
    print("\n[Step 4/5] Executing 105-epoch training pipeline...")
    history = train_model(
        model=model,
        train_data=train_gen,
        val_data=val_gen,
        epochs=TOTAL_EPOCHS,
        callbacks=callbacks,
        class_weights=class_weights,
        best_model_path=checkpoint_file,
        final_model_path="models/final_xceptional_model.keras",
        log_json_path="results/metrics/training_history.json"
    )

    print("\n[Step 5/5] Training pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
