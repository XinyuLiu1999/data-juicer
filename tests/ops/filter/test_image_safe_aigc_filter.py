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
            self.assertTrue(op.keep_real)
            self.assertTrue(op.any)

    def test_init_custom(self):
        """Test filter initialization with custom parameters."""
        with patch(
            "data_juicer.ops.filter.image_safe_aigc_filter.prepare_model"
        ) as mock_prepare:
            mock_prepare.return_value = MagicMock()
            op = ImageSafeAigcFilter(
                keep_real=False,
                any_or_all="all",
            )
            self.assertFalse(op.keep_real)
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
        """Test process_single keeps real images (prediction=0)."""
        with patch(
            "data_juicer.ops.filter.image_safe_aigc_filter.prepare_model"
        ) as mock_prepare:
            mock_prepare.return_value = MagicMock()
            op = ImageSafeAigcFilter(keep_real=True)

        # Real image (prediction=0) should be kept
        sample = {
            "images": [self.img1_path],
            Fields.stats: {
                StatsKeys.image_aigc_score: [0]  # 0 = real
            }
        }
        result = op.process_single(sample)
        self.assertTrue(result)

    def test_process_single_filter_fake(self):
        """Test process_single filters fake images (prediction=1)."""
        with patch(
            "data_juicer.ops.filter.image_safe_aigc_filter.prepare_model"
        ) as mock_prepare:
            mock_prepare.return_value = MagicMock()
            op = ImageSafeAigcFilter(keep_real=True)

        # Fake image (prediction=1) should be filtered
        sample = {
            "images": [self.img1_path],
            Fields.stats: {
                StatsKeys.image_aigc_score: [1]  # 1 = fake
            }
        }
        result = op.process_single(sample)
        self.assertFalse(result)

    def test_process_single_keep_fake(self):
        """Test process_single with keep_real=False keeps fake images."""
        with patch(
            "data_juicer.ops.filter.image_safe_aigc_filter.prepare_model"
        ) as mock_prepare:
            mock_prepare.return_value = MagicMock()
            op = ImageSafeAigcFilter(keep_real=False)

        # Fake image (prediction=1) should be kept when keep_real=False
        sample = {
            "images": [self.img1_path],
            Fields.stats: {
                StatsKeys.image_aigc_score: [1]  # 1 = fake
            }
        }
        result = op.process_single(sample)
        self.assertTrue(result)

        # Real image (prediction=0) should be filtered when keep_real=False
        sample2 = {
            "images": [self.img1_path],
            Fields.stats: {
                StatsKeys.image_aigc_score: [0]  # 0 = real
            }
        }
        result2 = op.process_single(sample2)
        self.assertFalse(result2)

    def test_process_single_any_strategy(self):
        """Test 'any' strategy: keep if any image meets condition."""
        with patch(
            "data_juicer.ops.filter.image_safe_aigc_filter.prepare_model"
        ) as mock_prepare:
            mock_prepare.return_value = MagicMock()
            op = ImageSafeAigcFilter(keep_real=True, any_or_all="any")

        # One real (0) and one fake (1)
        # With 'any', should keep since at least one is real
        sample = {
            "images": [self.img1_path, self.img2_path],
            Fields.stats: {
                StatsKeys.image_aigc_score: [1, 0]  # fake, real
            }
        }
        result = op.process_single(sample)
        self.assertTrue(result)

        # All fake
        sample2 = {
            "images": [self.img1_path, self.img2_path],
            Fields.stats: {
                StatsKeys.image_aigc_score: [1, 1]  # both fake
            }
        }
        result2 = op.process_single(sample2)
        self.assertFalse(result2)

    def test_process_single_all_strategy(self):
        """Test 'all' strategy: keep only if all images meet condition."""
        with patch(
            "data_juicer.ops.filter.image_safe_aigc_filter.prepare_model"
        ) as mock_prepare:
            mock_prepare.return_value = MagicMock()
            op = ImageSafeAigcFilter(keep_real=True, any_or_all="all")

        # One real (0) and one fake (1)
        # With 'all', should filter since not all are real
        sample = {
            "images": [self.img1_path, self.img2_path],
            Fields.stats: {
                StatsKeys.image_aigc_score: [1, 0]  # fake, real
            }
        }
        result = op.process_single(sample)
        self.assertFalse(result)

        # All real
        sample2 = {
            "images": [self.img1_path, self.img2_path],
            Fields.stats: {
                StatsKeys.image_aigc_score: [0, 0]  # both real
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
            op = ImageSafeAigcFilter()

        # Empty images should be kept (no images to filter)
        sample = {
            "images": [],
            Fields.stats: {
                StatsKeys.image_aigc_score: []
            }
        }
        result = op.process_single(sample)
        self.assertTrue(result)

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
        existing_preds = [0, 1]
        sample = {
            "images": [self.img1_path, self.img2_path],
            Fields.stats: {
                StatsKeys.image_aigc_score: existing_preds
            }
        }
        result = op.compute_stats_single(sample)
        # Should return unchanged
        self.assertEqual(
            result[Fields.stats][StatsKeys.image_aigc_score],
            existing_preds
        )


if __name__ == "__main__":
    unittest.main()
