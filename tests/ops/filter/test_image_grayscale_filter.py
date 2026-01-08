import os
import unittest
from io import BytesIO

from PIL import Image

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.filter.image_grayscale_filter import ImageGrayscaleFilter
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ImageGrayscaleFilterTest(DataJuicerTestCaseBase):
    """Test cases for ImageGrayscaleFilter.

    The grayscale filter checks if images are in grayscale mode (PIL mode "L"):
    - keep_grayscale=True: keeps all images (both color and grayscale)
    - keep_grayscale=False: filters out grayscale images, keeping only color

    Note: The filter checks image mode, not actual pixel values.
    A color image with only gray pixels will still be detected as color.
    """

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    img1_path = os.path.join(data_path, 'img1.png')  # Color image
    img2_path = os.path.join(data_path, 'img2.jpg')  # Color image
    img3_path = os.path.join(data_path, 'img3.jpg')  # Color image

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Create a grayscale test image
        cls.grayscale_img_path = os.path.join(cls.data_path,
                                              'test_grayscale.png')
        gray_img = Image.new('L', (100, 100), color=128)
        gray_img.save(cls.grayscale_img_path)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        # Clean up the test grayscale image
        if os.path.exists(cls.grayscale_img_path):
            os.remove(cls.grayscale_img_path)

    def _run_filter(self, dataset: Dataset, target_list, op):
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats)
        dataset = dataset.filter(op.process)
        dataset = dataset.select_columns(column_names=[op.image_key])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_compute_stats(self):
        """Test that compute_stats correctly identifies grayscale images."""
        ds_list = [
            {'images': [self.img1_path]},  # Color
            {'images': [self.grayscale_img_path]}  # Grayscale
        ]
        dataset = Dataset.from_list(ds_list)
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        op = ImageGrayscaleFilter()
        dataset = dataset.map(op.compute_stats)

        # Check color image
        sample_color = dataset[0]
        self.assertIn(StatsKeys.image_grayscale_flags, sample_color[Fields.stats])
        flags_color = sample_color[Fields.stats][StatsKeys.image_grayscale_flags]
        self.assertEqual(len(flags_color), 1)
        self.assertFalse(flags_color[0])  # Color image should not be grayscale

        # Check grayscale image
        sample_gray = dataset[1]
        flags_gray = sample_gray[Fields.stats][StatsKeys.image_grayscale_flags]
        self.assertEqual(len(flags_gray), 1)
        self.assertTrue(flags_gray[0])  # Grayscale image should be detected

    def test_keep_grayscale_true(self):
        """Test that keep_grayscale=True keeps all images."""
        ds_list = [
            {'images': [self.img1_path]},  # Color
            {'images': [self.grayscale_img_path]}  # Grayscale
        ]
        tgt_list = ds_list  # Both should be kept
        dataset = Dataset.from_list(ds_list)
        op = ImageGrayscaleFilter(keep_grayscale=True)
        self._run_filter(dataset, tgt_list, op)

    def test_keep_grayscale_false(self):
        """Test that keep_grayscale=False filters out grayscale images."""
        ds_list = [
            {'images': [self.img1_path]},  # Color - should be kept
            {'images': [self.grayscale_img_path]}  # Grayscale - should be filtered
        ]
        tgt_list = [{'images': [self.img1_path]}]
        dataset = Dataset.from_list(ds_list)
        op = ImageGrayscaleFilter(keep_grayscale=False)
        self._run_filter(dataset, tgt_list, op)

    def test_all_color_images(self):
        """Test with all color images."""
        ds_list = [
            {'images': [self.img1_path]},
            {'images': [self.img2_path]},
            {'images': [self.img3_path]}
        ]
        # With keep_grayscale=False, all color images should pass
        tgt_list = ds_list
        dataset = Dataset.from_list(ds_list)
        op = ImageGrayscaleFilter(keep_grayscale=False)
        self._run_filter(dataset, tgt_list, op)

    def test_any_strategy_with_mixed_images(self):
        """Test 'any' strategy with mixed color and grayscale images."""
        ds_list = [
            {'images': [self.img1_path, self.grayscale_img_path]},  # Mixed
            {'images': [self.img2_path, self.img3_path]}  # All color
        ]
        dataset = Dataset.from_list(ds_list)
        # With keep_grayscale=False and 'any' strategy:
        # First sample: one color passes, so sample passes
        # Second sample: both color, both pass
        op = ImageGrayscaleFilter(keep_grayscale=False, any_or_all='any')
        tgt_list = ds_list
        self._run_filter(dataset, tgt_list, op)

    def test_all_strategy_with_mixed_images(self):
        """Test 'all' strategy with mixed color and grayscale images."""
        ds_list = [
            {'images': [self.img1_path, self.grayscale_img_path]},  # Mixed
            {'images': [self.img2_path, self.img3_path]}  # All color
        ]
        dataset = Dataset.from_list(ds_list)
        # With keep_grayscale=False and 'all' strategy:
        # First sample: grayscale image fails, so sample fails
        # Second sample: both color, so sample passes
        op = ImageGrayscaleFilter(keep_grayscale=False, any_or_all='all')
        tgt_list = [{'images': [self.img2_path, self.img3_path]}]
        self._run_filter(dataset, tgt_list, op)

    def test_empty_images(self):
        """Test that samples with no images pass through."""
        ds_list = [{'images': []}]
        tgt_list = ds_list
        dataset = Dataset.from_list(ds_list)
        op = ImageGrayscaleFilter(keep_grayscale=False)
        self._run_filter(dataset, tgt_list, op)

    def test_invalid_any_or_all(self):
        """Test that invalid any_or_all parameter raises error."""
        with self.assertRaises(ValueError):
            ImageGrayscaleFilter(any_or_all='invalid')

    def test_multiple_grayscale_images(self):
        """Test with multiple grayscale images."""
        # Create second grayscale image
        grayscale_img2_path = os.path.join(self.data_path,
                                           'test_grayscale2.png')
        gray_img2 = Image.new('L', (50, 50), color=200)
        gray_img2.save(grayscale_img2_path)

        try:
            ds_list = [
                {'images': [self.grayscale_img_path, grayscale_img2_path]},
            ]
            dataset = Dataset.from_list(ds_list)
            # With keep_grayscale=False, sample with all grayscale should be filtered
            op = ImageGrayscaleFilter(keep_grayscale=False, any_or_all='all')
            tgt_list = []
            self._run_filter(dataset, tgt_list, op)
        finally:
            if os.path.exists(grayscale_img2_path):
                os.remove(grayscale_img2_path)


if __name__ == '__main__':
    unittest.main()
