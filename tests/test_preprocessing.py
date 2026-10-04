"""
Unit Tests for CT Image Preprocessing & Data Augmentation Pipelines
===================================================================
Tests core functionality, shape constraints, normalization bounds,
and clinical validity of transformations.
"""

import os
import sys
import unittest
import numpy as np

# Ensure src module is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import (
    load_image,
    remove_noise,
    enhance_contrast,
    resize_image,
    normalize_for_xception,
    denormalize_from_xception,
    preprocess_image,
    verify_dataset,
    XCEPTION_INPUT_SHAPE
)
from src.augmentation import MedicallyValidAugmentor


class TestPreprocessingPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Locate a sample image from the test set
        cls.sample_path = "sample_images/sample_adenocarcinoma.png"
        if not os.path.exists(cls.sample_path):
            cls.sample_path = "dataset/train/adenocarcinoma/000000 (6).png"

    def test_load_image(self):
        """Test that images load correctly in RGB format with 3 dimensions."""
        img = load_image(self.sample_path)
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.ndim, 3)
        self.assertEqual(img.shape[2], 3)
        self.assertEqual(img.dtype, np.uint8)

    def test_noise_removal(self):
        """Test that bilateral noise filter preserves image shape and type."""
        img = load_image(self.sample_path)
        denoised = remove_noise(img, method="bilateral")
        self.assertEqual(denoised.shape, img.shape)
        self.assertEqual(denoised.dtype, np.uint8)

    def test_contrast_enhancement(self):
        """Test CLAHE contrast enhancement on LAB luminance channel."""
        img = load_image(self.sample_path)
        enhanced = enhance_contrast(img, clip_limit=2.0)
        self.assertEqual(enhanced.shape, img.shape)
        self.assertEqual(enhanced.dtype, np.uint8)

    def test_resize_image(self):
        """Test that images resize exactly to Xception dimensions (299, 299)."""
        img = load_image(self.sample_path)
        resized = resize_image(img, target_size=XCEPTION_INPUT_SHAPE)
        self.assertEqual(resized.shape, (299, 299, 3))

    def test_normalization_bounds(self):
        """Test that Xception normalization bounds pixels strictly into [-1.0, 1.0]."""
        test_pixels = np.array([[[0, 127, 255]]], dtype=np.uint8)
        norm = normalize_for_xception(test_pixels)
        self.assertAlmostEqual(norm[0, 0, 0], -1.0, places=4)
        self.assertAlmostEqual(norm[0, 0, 2], 1.0, places=4)
        self.assertTrue(np.all(norm >= -1.0) and np.all(norm <= 1.0))

    def test_denormalization(self):
        """Test that denormalization recovers uint8 values in [0, 255]."""
        norm = np.array([[[-1.0, 0.0, 1.0]]], dtype=np.float32)
        recovered = denormalize_from_xception(norm)
        self.assertEqual(recovered.dtype, np.uint8)
        self.assertEqual(recovered[0, 0, 0], 0)
        self.assertEqual(recovered[0, 0, 2], 255)

    def test_full_preprocess_pipeline(self):
        """Test end-to-end preprocessing pipeline output properties."""
        processed = preprocess_image(self.sample_path, target_size=XCEPTION_INPUT_SHAPE, return_normalized=True)
        self.assertEqual(processed.shape, (299, 299, 3))
        self.assertEqual(processed.dtype, np.float32)
        self.assertTrue(np.all(processed >= -1.0) and np.all(processed <= 1.0))

    def test_dataset_verification(self):
        """Test that the entire dataset has zero corrupted files."""
        report = verify_dataset("dataset")
        self.assertEqual(len(report["corrupted_files"]), 0)
        self.assertGreater(report["total_valid_images"], 900)


class TestAugmentationPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_path = "sample_images/sample_normal.png"
        if not os.path.exists(cls.sample_path):
            cls.sample_path = "dataset/train/normal/10 (2) - Copy.png"
        cls.base_img = preprocess_image(cls.sample_path, return_normalized=False)
        cls.augmentor = MedicallyValidAugmentor(seed=123)

    def test_rotation_shape(self):
        """Test rotation preserves (299, 299, 3) shape."""
        rotated = self.augmentor.apply_rotation(self.base_img, angle=10.0)
        self.assertEqual(rotated.shape, (299, 299, 3))

    def test_horizontal_flip_shape(self):
        """Test horizontal flip preserves dimensions and reflects left-to-right."""
        flipped = self.augmentor.apply_horizontal_flip(self.base_img)
        self.assertEqual(flipped.shape, (299, 299, 3))
        # Leftmost column of original should match rightmost column of flipped
        np.testing.assert_array_equal(self.base_img[:, 0, :], flipped[:, -1, :])

    def test_zoom_shape(self):
        """Test zoom preserves (299, 299, 3) shape."""
        zoomed_in = self.augmentor.apply_zoom(self.base_img, zoom_factor=1.06)
        zoomed_out = self.augmentor.apply_zoom(self.base_img, zoom_factor=0.94)
        self.assertEqual(zoomed_in.shape, (299, 299, 3))
        self.assertEqual(zoomed_out.shape, (299, 299, 3))

    def test_shift_shape(self):
        """Test shift preserves (299, 299, 3) shape."""
        shifted = self.augmentor.apply_shift(self.base_img, shift_x=0.05, shift_y=-0.04)
        self.assertEqual(shifted.shape, (299, 299, 3))

    def test_brightness_shape_and_bounds(self):
        """Test brightness scaling stays in [0, 255] for uint8."""
        brightened = self.augmentor.apply_brightness(self.base_img, factor=1.05)
        self.assertEqual(brightened.shape, (299, 299, 3))
        self.assertEqual(brightened.dtype, np.uint8)

    def test_end_to_end_augment(self):
        """Test random composite augmentation preserves (299, 299, 3) shape."""
        for _ in range(5):
            aug = self.augmentor.augment(self.base_img)
            self.assertEqual(aug.shape, (299, 299, 3))


if __name__ == "__main__":
    unittest.main()
