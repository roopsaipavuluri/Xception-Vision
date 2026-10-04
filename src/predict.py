"""
Run single-image inference with the trained Xceptional Vision model.
"""

import argparse
import os
import sys
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Support both `python src/predict.py` and importing this module.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import XCEPTION_INPUT_SHAPE, load_image, preprocess_image


def display_prediction_visualization(
    result: Dict[str, Any],
    save_path: Optional[str] = "results/graphs/single_prediction.png",
    show_plot: bool = True
) -> None:
    """
    Displays and saves a clean, professional single-image prediction
    visualization.

    Layout:
        1. Original CT Scan
        2. Preprocessed CT Scan
        3. Class Probability Distribution

    Only visualization formatting is handled here.
    Prediction logic, preprocessing, model loading, and class names
    are not modified.
    """

    # ------------------------------------------------------------------
    # Extract prediction data
    # ------------------------------------------------------------------
    raw_img = result["raw_image"]
    prep_img = result["preprocessed_image"]

    pred_class = result["predicted_class_clean"]
    conf_pct = result["confidence_percent"]
    probs = result["all_probabilities"]

    # ------------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10, 5),
        dpi=100,
        gridspec_kw={"width_ratios": (1, 1, 1.3)},
        constrained_layout=True
    )

    # ==============================================================
    # PANEL 1 — ORIGINAL CT SCAN
    # ==============================================================

    axes[0].imshow(
        raw_img,
        aspect="equal"
    )

    axes[0].set_title(
        "Original CT Scan",
        fontsize=9,
        fontweight="bold",
        pad=6
    )

    axes[0].axis("off")
    axes[0].set_aspect("equal")

    # Small filename below the image
    filename = os.path.basename(result["image_path"])

    axes[0].text(
        0.5,
        -0.07,
        f"File: {filename}",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=8,
        wrap=True
    )

    # ==============================================================
    # PANEL 2 — PREPROCESSED CT SCAN
    # ==============================================================

    axes[1].imshow(
        prep_img,
        aspect="equal"
    )

    axes[1].set_title(
        "Preprocessed CT Scan\n"
        "Bilateral Filter + CLAHE",
        fontsize=9,
        fontweight="bold",
        pad=6
    )

    axes[1].axis("off")
    axes[1].set_aspect("equal")

    # Small preprocessing description
    axes[1].text(
        0.5,
        -0.07,
        "Normalized for Xception",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=8
    )

    # ==============================================================
    # PANEL 3 — CLASS PROBABILITY DISTRIBUTION
    # ==============================================================

    classes = [
        c.replace("_", " ").title()
        for c in probs.keys()
    ]

    scores = [
        float(v) * 100.0
        for v in probs.values()
    ]

    # Highlight predicted class
    bar_colors = [
        "#2ca02c" if c == pred_class else "#1f77b4"
        for c in classes
    ]

    y_pos = np.arange(len(classes))

    bars = axes[2].barh(
        y_pos,
        scores,
        color=bar_colors,
        height=0.52,
        edgecolor="#333333",
        linewidth=0.6
    )

    # Class names
    axes[2].set_yticks(y_pos)

    axes[2].set_yticklabels(
        classes,
        fontsize=8,
        fontweight="bold"
    )

    # X-axis
    axes[2].set_xlabel(
        "Confidence Score (%)",
        fontsize=8,
        fontweight="bold",
        labelpad=6
    )

    # Give percentage labels enough space
    axes[2].set_xlim(
        0,
        max(105, max(scores) + 12)
    )

    # Grid
    axes[2].grid(
        True,
        linestyle=":",
        alpha=0.5,
        axis="x"
    )

    axes[2].set_axisbelow(True)

    # Put highest class at top
    axes[2].invert_yaxis()

    # ==============================================================
    # PERCENTAGE LABELS ON BARS
    # ==============================================================

    for bar, score in zip(bars, scores):

        axes[2].text(
            score + 1.0,
            bar.get_y() + bar.get_height() / 2.0,
            f"{score:.1f}%",
            va="center",
            ha="left",
            fontsize=8,
            fontweight="bold"
        )

    # ==============================================================
    # PROBABILITY CHART TITLE
    # ==============================================================

    axes[2].set_title(
        "Class Probabilities",
        fontsize=9,
        fontweight="bold",
        pad=6
    )

    # ==============================================================
    # REMOVE UNNECESSARY SPINES
    # ==============================================================

    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    # ==============================================================
    # MAIN FIGURE TITLE
    # ==============================================================

    fig.suptitle(
        f"CT Scan Prediction: {pred_class} ({conf_pct:.1f}%)",
        fontsize=11,
        fontweight="bold",
        wrap=True
    )

    # ==============================================================
    # FOOTER / DISCLAIMER
    # ==============================================================

    fig.text(
        0.5,
        0.035,
        "For educational and research use only; not for clinical diagnosis.",
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.35",
            fc="#fff3f3",
            ec="#cc0000",
            lw=0.8
        )
    )

    # ==============================================================
    # SAVE FIGURE
    # ==============================================================

    if save_path:

        save_directory = os.path.dirname(save_path)

        if save_directory:
            os.makedirs(
                save_directory,
                exist_ok=True
            )

        fig.savefig(
            save_path,
            dpi=100,
            pad_inches=0.1
        )

        print(f"Prediction visualization saved to: {save_path}")

    # ==============================================================
    # DISPLAY FIGURE
    # ==============================================================

    if show_plot:
        plt.show()

    # Close figure after display/save
    plt.close(fig)


def predict_image(image_path: str, model_path: str) -> Dict[str, Any]:
    """Load the model and return the prediction and visualization inputs."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    train_dir = os.path.join(os.path.dirname(__file__), "..", "dataset", "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Training class folders not found: {os.path.abspath(train_dir)}"
        )

    class_names = sorted(
        name
        for name in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, name)) and not name.startswith(".")
    )
    if not class_names:
        raise ValueError(f"No class folders found in training directory: {train_dir}")

    model = tf.keras.models.load_model(model_path)
    raw_img = load_image(image_path)
    preprocessed_img = preprocess_image(
        image_path,
        target_size=XCEPTION_INPUT_SHAPE,
        return_normalized=False
    )
    model_input = np.expand_dims(
        (preprocessed_img.astype(np.float32) / 127.5) - 1.0,
        axis=0
    )

    probabilities = np.asarray(model.predict(model_input, verbose=0))[0]
    if probabilities.ndim != 1 or probabilities.size != len(class_names):
        raise ValueError(
            "Model output does not match the training classes: "
            f"received {probabilities.size} scores for {len(class_names)} classes."
        )

    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence_percent = float(probabilities[predicted_index]) * 100.0

    return {
        "image_path": image_path,
        "raw_image": raw_img,
        "preprocessed_image": preprocessed_img,
        "predicted_class_clean": predicted_class.replace("_", " ").title(),
        "confidence_percent": confidence_percent,
        "all_probabilities": dict(zip(class_names, probabilities.tolist())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict the class of a CT scan image."
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--model",
        default=os.path.join("models", "best_xceptional_model.keras"),
        help="Path to a trained Keras model checkpoint.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Save the visualization without opening a figure window.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("results", "graphs", "single_prediction.png"),
        help="Path for the prediction visualization.",
    )
    args = parser.parse_args()

    result = predict_image(args.image, args.model)
    print(
        f"Prediction: {result['predicted_class_clean']} "
        f"({result['confidence_percent']:.2f}%)"
    )
    display_prediction_visualization(
        result,
        save_path=args.output,
        show_plot=not args.no_display
    )


if __name__ == "__main__":
    main()