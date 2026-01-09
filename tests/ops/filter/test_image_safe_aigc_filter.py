# flake8: noqa: E501

import os
import unittest
from unittest.mock import MagicMock, patch

from data_juicer.ops.filter.image_safe_aigc_filter import ImageSafeAigcFilter
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ImageSafeAigcFilterTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..",
                             "data")

    img1_path = os.path.join(data_path, "img1.png")
    img2_path = os.path.join(data_path, "img2.jpg")
    img3_path = os.path.join(data_path, "img3.jpg")

    def test_init_default(self):
        """Test filter initialization with default parameters."""
        with patch(
            "data_juicer.ops.filter.image_safe_aigc_filter.prepare_model"
        ) as mock_prepare:
            mock_prepare.return_value = MagicMock()
            op = ImageSafeAigcFilter()
            self.assertEqual(op.min_score, 0.0)
            self.assertEqual(op.max_score, 0.5)
            self.assertTrue(op.any)

    def test_init_custom(self):
        """Test filter initialization with custom parameters."""
        with patch(
            "data_juicer.ops.filter.image_safe_aigc_filter.prepare_model"
        ) as mock_prepare:
            mock_prepare.return_value = MagicMock()
            op = ImageSafeAigcFilter(
                min_score=0.1,
                max_score=0.4,
                any_or_all="all",
            )
            self.assertEqual(op.min_score, 0.1)
            self.assertEqual(op.max_score, 0.4)
            self.assertFalse(op.any)

    def test_invalid_any_or_all(self):
        """Test that invalid any_or_all raises ValueError."""
        with patch(
            "data_juicer.ops.filter.image_safe_aigc_filter.prepare_model"
        ) as mock_prepare:
            mock_prepare.return_value = MagicMock()
            with self.assertRaises(ValueError):
                ImageSafeAigcFilter(any_or_all="invalid")

    def test_process_single_keep_real(self):
        """Test process_single keeps real images (low AIGC score)."""
        with patch(
            "data_juicer.ops.filter.image_safe_aigc_filter.prepare_model"
        ) as mock_prepare:
            mock_prepare.return_value = MagicMock()
            op = ImageSafeAigcFilter(max_score=0.5)

        # Simulate a sample with low AIGC score (real image)
        sample = {
            "images": [self.img1_path],
            Fields.stats: {
                StatsKeys.image_aigc_score: [0.2]  # Low score = real
            }
        }
        result = op.process_single(sample)
        self.assertTrue(result)

    def test_process_single_filter_fake(self):
        """Test process_single filters fake images (high AIGC score)."""
        with patch(
            "data_juicer.ops.filter.image_safe_aigc_filter.prepare_model"
        ) as mock_prepare:
            mock_prepare.return_value = MagicMock()
            op = ImageSafeAigcFilter(max_score=0.5)

        # Simulate a sample with high AIGC score (fake image)
        sample = {
            "images": [self.img1_path],
            Fields.stats: {
                StatsKeys.image_aigc_score: [0.8]  # High score = fake
            }
        }
        result = op.process_single(sample)
        self.assertFalse(result)

    def test_process_single_any_strategy(self):
        """Test 'any' strategy: keep if any image passes."""
        with patch(
            "data_juicer.ops.filter.image_safe_aigc_filter.prepare_model"
        ) as mock_prepare:
            mock_prepare.return_value = MagicMock()
            op = ImageSafeAigcFilter(max_score=0.5, any_or_all="any")

        # Sample with one real (0.3) and one fake (0.8)
        # With 'any', should keep since at least one passes
        sample = {
            "images": [self.img1_path, self.img2_path],
            Fields.stats: {
                StatsKeys.image_aigc_score: [0.8, 0.3]
            }
        }
        result = op.process_single(sample)
        self.assertTrue(result)

    def test_process_single_all_strategy(self):
        """Test 'all' strategy: keep only if all images pass."""
        with patch(
            "data_juicer.ops.filter.image_safe_aigc_filter.prepare_model"
        ) as mock_prepare:
            mock_prepare.return_value = MagicMock()
            op = ImageSafeAigcFilter(max_score=0.5, any_or_all="all")

        # Sample with one real (0.3) and one fake (0.8)
        # With 'all', should filter since not all pass
        sample = {
            "images": [self.img1_path, self.img2_path],
            Fields.stats: {
                StatsKeys.image_aigc_score: [0.8, 0.3]
            }
        }
        result = op.process_single(sample)
        self.assertFalse(result)

        # Sample with both real
        sample2 = {
            "images": [self.img1_path, self.img2_path],
            Fields.stats: {
                StatsKeys.image_aigc_score: [0.2, 0.3]
            }
        }
        result2 = op.process_single(sample2)
        self.assertTrue(result2)

    def test_process_single_empty_images(self):
        """Test handling of samples with no images."""
        with patch(
            "data_juicer.ops.filter.image_safe_aigc_filter.prepare_model"
        ) as mock_prepare:
            mock_prepare.return_value = MagicMock()
            op = ImageSafeAigcFilter(max_score=0.5)

        # Empty images should be kept (no images to filter)
        sample = {
            "images": [],
            Fields.stats: {
                StatsKeys.image_aigc_score: []
            }
        }
        result = op.process_single(sample)
        self.assertTrue(result)

    def test_process_single_min_score(self):
        """Test filtering with min_score threshold."""
        with patch(
            "data_juicer.ops.filter.image_safe_aigc_filter.prepare_model"
        ) as mock_prepare:
            mock_prepare.return_value = MagicMock()
            # Filter images with score between 0.1 and 0.5
            op = ImageSafeAigcFilter(min_score=0.1, max_score=0.5)

        # Score below min - should be filtered
        sample1 = {
            "images": [self.img1_path],
            Fields.stats: {
                StatsKeys.image_aigc_score: [0.05]
            }
        }
        self.assertFalse(op.process_single(sample1))

        # Score in range - should be kept
        sample2 = {
            "images": [self.img1_path],
            Fields.stats: {
                StatsKeys.image_aigc_score: [0.3]
            }
        }
        self.assertTrue(op.process_single(sample2))

        # Score above max - should be filtered
        sample3 = {
            "images": [self.img1_path],
            Fields.stats: {
                StatsKeys.image_aigc_score: [0.7]
            }
        }
        self.assertFalse(op.process_single(sample3))

    def test_compute_stats_single_empty(self):
        """Test compute_stats_single with empty images."""
        with patch(
            "data_juicer.ops.filter.image_safe_aigc_filter.prepare_model"
        ) as mock_prepare:
            mock_prepare.return_value = MagicMock()
            op = ImageSafeAigcFilter()

        sample = {
            "images": [],
            Fields.stats: {}
        }
        result = op.compute_stats_single(sample)
        self.assertEqual(
            list(result[Fields.stats][StatsKeys.image_aigc_score]), []
        )

    def test_compute_stats_single_already_computed(self):
        """Test compute_stats_single skips if already computed."""
        with patch(
            "data_juicer.ops.filter.image_safe_aigc_filter.prepare_model"
        ) as mock_prepare:
            mock_prepare.return_value = MagicMock()
            op = ImageSafeAigcFilter()

        # Pre-computed stats
        existing_scores = [0.3, 0.4]
        sample = {
            "images": [self.img1_path, self.img2_path],
            Fields.stats: {
                StatsKeys.image_aigc_score: existing_scores
            }
        }
        result = op.compute_stats_single(sample)
        # Should return unchanged
        self.assertEqual(
            result[Fields.stats][StatsKeys.image_aigc_score],
            existing_scores
        )


if __name__ == "__main__":
    unittest.main()
