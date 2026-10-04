"""
Xceptional Vision - Preprocessing Module
=======================================
Implements the clinical image preprocessing pipeline for CT scan images:
1. Dataset format verification and integrity auditing (corrupted file detection).
2. Noise reduction using edge-preserving Bilateral Filtering.
3. Contrast enhancement using Contrast Limited Adaptive Histogram Equalization (CLAHE) on Luminance channel.
4. Bilinear/Bicubic resizing to 299x299 px (Xception input requirement).
5. Normalization matching the Xception architecture expectation ([-1.0, 1.0]).
6. Before/After visual comparison generation.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np


# Standard input dimensions for Xception
XCEPTION_INPUT_SHAPE: Tuple[int, int] = (299, 299)
VALID_IMAGE_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk using OpenCV in RGB format.

    Args:
        image_path (str): Filepath to the CT scan image.

    Returns:
        np.ndarray: Image array in RGB color format with uint8 dtype.

    Raises:
        FileNotFoundError: If the specified path does not exist.
        ValueError: If the file is unreadable or corrupted.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Read image using OpenCV (loads in BGR format)
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Corrupted or invalid image file: {image_path}")

    # Convert BGR to standard RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


def verify_dataset(dataset_root: str = "dataset") -> Dict[str, Union[Dict[str, int], List[str]]]:
    """
    Audits the entire dataset across all splits (train, validation, test) to:
      - Confirm supported image file extensions.
      - Detect any corrupted, unreadable, or truncated image files.
      - Count valid images per class and split.

    Args:
        dataset_root (str): Root path to the dataset folder.

    Returns:
        Dict: Audit summary containing per-split class counts and list of corrupted files.
    """
    report = {
        "counts": {},
        "corrupted_files": [],
        "unsupported_files": [],
        "total_valid_images": 0,
    }

    splits = ["train", "validation", "test"]

    for split in splits:
        split_dir = os.path.join(dataset_root, split)
        report["counts"][split] = {}

        if not os.path.isdir(split_dir):
            continue

        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            valid_count = 0
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)

                # Skip hidden files like .gitkeep
                if file_name.startswith("."):
                    continue

                ext = os.path.splitext(file_name)[1].lower()
                if ext not in VALID_IMAGE_EXTENSIONS:
                    report["unsupported_files"].append(file_path)
                    continue

                try:
                    # Validate image header and pixel buffer readability
                    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                    if img is None or img.size == 0 or img.ndim != 3:
                        report["corrupted_files"].append(file_path)
                    else:
                        valid_count += 1
                        report["total_valid_images"] += 1
                except Exception:
                    report["corrupted_files"].append(file_path)

            report["counts"][split][class_name] = valid_count

    return report


def remove_noise(
    image: np.ndarray,
    method: str = "bilateral",
    diameter: int = 5,
    sigma_color: float = 50.0,
    sigma_space: float = 50.0
) -> np.ndarray:
    """
    Reduces CT scanner quantum mottle and sensor noise while preserving tissue boundaries.

    Clinical Rationale:
        Bilateral filtering smooths flat homogeneous regions (parenchyma, fluid)
        while strictly preserving high-frequency edges of nodules, calcifications,
        and pleural walls, unlike Gaussian blur which blurs diagnostic margins.

    Args:
        image (np.ndarray): Input RGB image (uint8).
        method (str): Filtering method ('bilateral' or 'median').
        diameter (int): Diameter of pixel neighborhood.
        sigma_color (float): Filter sigma in the color space.
        sigma_space (float): Filter sigma in coordinate space.

    Returns:
        np.ndarray: Denoised RGB image.
    """
    if method == "bilateral":
        return cv2.bilateralFilter(
            src=image,
            d=diameter,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_space
        )
    elif method == "median":
        return cv2.medianBlur(src=image, ksize=3)
    else:
        raise ValueError(f"Unsupported noise removal method: {method}")


def enhance_contrast(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Enhances local tissue contrast using CLAHE applied to the Luminance (L) channel.

    Clinical Rationale:
        Global histogram equalization produces unnatural brightness blowouts in CT scans.
        Applying CLAHE to the L* channel in CIE LAB color space amplifies subtle density
        variations within tumors and soft tissues without introducing chromatic distortion.

    Args:
        image (np.ndarray): Input RGB image (uint8).
        clip_limit (float): Threshold for contrast limiting.
        tile_grid_size (Tuple[int, int]): Size of grid for histogram equalization.

    Returns:
        np.ndarray: Contrast-enhanced RGB image.
    """
    # Convert RGB to CIE LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Apply CLAHE to the Luminance channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_l = clahe.apply(l_channel)

    # Re-merge channels and convert back to RGB
    enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return enhanced_rgb


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = XCEPTION_INPUT_SHAPE,
    interpolation: int = cv2.INTER_CUBIC
) -> np.ndarray:
    """
    Resizes CT scan image to the input dimensions required by the Xception architecture.

    Args:
        image (np.ndarray): Input image array.
        target_size (Tuple[int, int]): (width, height), default (299, 299).
        interpolation (int): OpenCV interpolation method (default: cv2.INTER_CUBIC).

    Returns:
        np.ndarray: Resized image with shape (target_size[1], target_size[0], 3).
    """
    return cv2.resize(image, target_size, interpolation=interpolation)


def normalize_for_xception(image: np.ndarray) -> np.ndarray:
    """
    Normalizes pixel values to the range [-1.0, 1.0] matching Xception pre-training.

    Formula:
        x_norm = (x / 127.5) - 1.0

    Args:
        image (np.ndarray): Input image array with pixel values in [0, 255].

    Returns:
        np.ndarray: Float32 normalized array bounded in [-1.0, 1.0].
    """
    image_float = image.astype(np.float32)
    normalized = (image_float / 127.5) - 1.0
    return normalized


def denormalize_from_xception(normalized_image: np.ndarray) -> np.ndarray:
    """
    Reverses Xception normalization from [-1.0, 1.0] back to [0, 255] for visual inspection.

    Formula:
        x = ((x_norm + 1.0) * 127.5).astype(uint8)

    Args:
        normalized_image (np.ndarray): Normalized image array in [-1.0, 1.0].

    Returns:
        np.ndarray: Uint8 array in [0, 255].
    """
    denorm = (normalized_image + 1.0) * 127.5
    denorm = np.clip(denorm, 0, 255).astype(np.uint8)
    return denorm


def preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = XCEPTION_INPUT_SHAPE,
    return_normalized: bool = True
) -> np.ndarray:
    """
    Complete end-to-end preprocessing pipeline for a single CT scan image:
      1. Load image (RGB)
      2. Denoise (Bilateral Filter)
      3. Enhance contrast (CLAHE on L-channel)
      4. Resize to target size (299x299)
      5. Normalize for Xception [-1.0, 1.0] (optional)

    Args:
        image_path (str): Filepath to the raw CT image.
        target_size (Tuple[int, int]): Target dimensions (default 299x299).
        return_normalized (bool): If True, returns float32 in [-1, 1]; else uint8 in [0, 255].

    Returns:
        np.ndarray: Preprocessed CT scan image array.
    """
    raw_img = load_image(image_path)
    denoised_img = remove_noise(raw_img, method="bilateral")
    enhanced_img = enhance_contrast(denoised_img, clip_limit=2.0)
    resized_img = resize_image(enhanced_img, target_size=target_size)

    if return_normalized:
        return normalize_for_xception(resized_img)
    return resized_img


def display_raw_sample_grid(
    samples_by_class: Dict[str, str],
    save_path: Optional[str] = "results/graphs/raw_samples_before_preprocessing.png"
) -> None:
    """
    Visualizes raw sample CT scans before any preprocessing across all classes.

    Args:
        samples_by_class (Dict[str, str]): Mapping of class name to sample image path.
        save_path (Optional[str]): Path to save the resulting figure.
    """
    num_classes = len(samples_by_class)
    fig, axes = plt.subplots(1, num_classes, figsize=(4 * num_classes, 4.5))

    if num_classes == 1:
        axes = [axes]

    for idx, (class_name, image_path) in enumerate(samples_by_class.items()):
        img = load_image(image_path)
        h, w, c = img.shape
        clean_title = class_name.replace("_", " ").title()

        axes[idx].imshow(img)
        axes[idx].set_title(f"{clean_title}\nOriginal Shape: {w}x{h} px", fontsize=11, fontweight="bold", pad=8)
        axes[idx].axis("off")

    plt.suptitle("Raw CT Scan Slices Before Preprocessing (Ground Truth Unaltered)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved raw samples visualization to: {save_path}")

    plt.close()


def display_preprocessing_comparison(
    samples_by_class: Dict[str, str],
    save_path: Optional[str] = "results/graphs/preprocessing_comparison.png"
) -> None:
    """
    Generates a step-by-step comparative visualization showing:
      Column 1: Raw CT Scan (Original dimensions)
      Column 2: Denoised (Bilateral Filter preserving edges)
      Column 3: Contrast Enhanced (CLAHE on L-channel)
      Column 4: Final Preprocessed (Resized 299x299 & Normalized for Xception)

    Args:
        samples_by_class (Dict[str, str]): Mapping of class name to sample image path.
        save_path (Optional[str]): Path to save the figure.
    """
    num_classes = len(samples_by_class)
    fig, axes = plt.subplots(num_classes, 4, figsize=(16, 3.8 * num_classes))

    if num_classes == 1:
        axes = np.expand_dims(axes, axis=0)

    col_headers = [
        "1. Raw CT Scan\n(Original)",
        "2. Denoised\n(Bilateral Filter)",
        "3. Contrast Enhanced\n(CLAHE L-Channel)",
        "4. Preprocessed for Xception\n(299x299, Normalized)"
    ]

    for row_idx, (class_name, image_path) in enumerate(samples_by_class.items()):
        raw = load_image(image_path)
        denoised = remove_noise(raw, method="bilateral")
        enhanced = enhance_contrast(denoised, clip_limit=2.0)
        final_norm = preprocess_image(image_path, target_size=XCEPTION_INPUT_SHAPE, return_normalized=True)
        # Denormalize to uint8 for visual rendering
        final_display = denormalize_from_xception(final_norm)

        images = [raw, denoised, enhanced, final_display]
        clean_class_name = class_name.replace("_", " ").title()

        for col_idx, (img, header) in enumerate(zip(images, col_headers)):
            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            ax.axis("off")

            # Column headers on top row
            if row_idx == 0:
                ax.set_title(header, fontsize=12, fontweight="bold", pad=10)

            # Class label on left column
            if col_idx == 0:
                ax.text(
                    -0.1, 0.5, clean_class_name,
                    transform=ax.transAxes,
                    fontsize=12, fontweight="bold",
                    va="center", ha="right", rotation=90
                )

    plt.suptitle("Xceptional Vision – CT Image Preprocessing Pipeline Stages", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved preprocessing comparison visualization to: {save_path}")

    plt.close()
