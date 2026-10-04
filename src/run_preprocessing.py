"""
Xceptional Vision - Preprocessing Pipeline Runner
================================================
Orchestrates dataset verification, preprocessing, augmentation, and visualization.
Generates publication-quality charts saved to results/graphs/.
"""

import os
import sys
from typing import Dict

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure src module is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import (
    verify_dataset,
    load_image,
    preprocess_image,
    display_raw_sample_grid,
    display_preprocessing_comparison,
    XCEPTION_INPUT_SHAPE
)
from src.augmentation import visualize_augmentation_variations


def get_sample_images_by_class(dataset_dir: str = "dataset/train") -> Dict[str, str]:
    """
    Finds one representative CT scan image for each class.
    """
    samples = {}
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Directory not found: {dataset_dir}")

    for class_name in sorted(os.listdir(dataset_dir)):
        class_path = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_path):
            files = [f for f in sorted(os.listdir(class_path)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if files:
                samples[class_name] = os.path.join(class_path, files[0])

    return samples


def main():
    print("=" * 75)
    print("XCEPTIONAL VISION: CT IMAGE PREPROCESSING & VERIFICATION PIPELINE")
    print("=" * 75)

    # 1. Dataset Integrity and Format Verification
    print("\n[Step 1/5] Auditing dataset integrity and image formats...")
    audit_report = verify_dataset(dataset_root="dataset")

    print("\nDataset Verification Summary:")
    print(f"Total Valid Images Found: {audit_report['total_valid_images']}")
    print("-" * 50)
    for split, counts in audit_report["counts"].items():
        print(f"Split: [{split.upper()}] (Total: {sum(counts.values())} images)")
        for cls_name, count in counts.items():
            print(f"  • {cls_name:25s}: {count:4d} images")
    print("-" * 50)

    if audit_report["corrupted_files"]:
        print(f"⚠️ Warning: Found {len(audit_report['corrupted_files'])} corrupted files:")
        for f in audit_report["corrupted_files"]:
            print(f"  - {f}")
    else:
        print("✓ Zero corrupted or unreadable images detected. 100% data integrity verified.")

    if audit_report["unsupported_files"]:
        print(f"⚠️ Warning: Found {len(audit_report['unsupported_files'])} unsupported files.")
    else:
        print("✓ All image formats valid (.png, .jpg).")

    # 2. Locate Sample CT Slices Across Classes
    print("\n[Step 2/5] Locating representative sample CT slices for each class...")
    samples = get_sample_images_by_class(dataset_dir="dataset/train")
    for cls_name, path in samples.items():
        print(f"  • {cls_name}: {path}")

    # 3. Display and Save Raw Images Before Preprocessing
    print("\n[Step 3/5] Visualizing raw CT images before preprocessing...")
    raw_output_path = "results/graphs/raw_samples_before_preprocessing.png"
    display_raw_sample_grid(samples, save_path=raw_output_path)

    # 4. Display and Save Multi-Stage Preprocessing Comparison
    print("\n[Step 4/5] Executing multi-stage preprocessing comparison...")
    print("  • Bilateral Denoising: Preserving tumor margins while smoothing quantum mottle")
    print("  • CLAHE Contrast: Adaptive equalization on Luminance (L*) channel in LAB space")
    print(f"  • Resizing: Standardizing to Xception input dimensions {XCEPTION_INPUT_SHAPE}")
    print("  • Normalization: Scaling pixel range to [-1.0, 1.0]")
    comparison_output_path = "results/graphs/preprocessing_comparison.png"
    display_preprocessing_comparison(samples, save_path=comparison_output_path)

    # 5. Medically Valid Data Augmentation
    print("\n[Step 5/5] Visualizing medically valid data augmentations...")
    # Use adenocarcinoma as demonstration base
    demo_class = "adenocarcinoma"
    if demo_class in samples:
        demo_img_norm = preprocess_image(samples[demo_class], return_normalized=False)
        aug_output_path = "results/graphs/data_augmentation_samples.png"
        visualize_augmentation_variations(
            base_image=demo_img_norm,
            class_name=demo_class,
            save_path=aug_output_path,
            num_variations=5
        )

    print("\n" + "=" * 75)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Generated Visualizations:")
    print(f"  1. {raw_output_path}")
    print(f"  2. {comparison_output_path}")
    print(f"  3. {aug_output_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
