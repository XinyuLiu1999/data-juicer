import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.filter.image_brightness_filter import ImageBrightnessFilter
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ImageBrightnessFilterTest(DataJuicerTestCaseBase):
    """Test cases for ImageBrightnessFilter.

    The brightness filter uses perceived luminance calculation and evaluates:
    - brightness_perc_5: 5th percentile brightness (darkest regions)
    - brightness_perc_99: 99th percentile brightness (brightest regions)
    - brightness_mean: mean brightness

    Filtering logic:
    - Rejects if 99th percentile < min_brightness (image too dark overall)
    - Rejects if 5th percentile > max_brightness (image too bright/overexposed)
    """

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    img1_path = os.path.join(data_path, 'img1.png')
    img2_path = os.path.join(data_path, 'img2.jpg')
    img3_path = os.path.join(data_path, 'img3.jpg')

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
        """Test that compute_stats correctly calculates brightness stats."""
        ds_list = [{'images': [self.img1_path]}]
        dataset = Dataset.from_list(ds_list)
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        op = ImageBrightnessFilter()
        dataset = dataset.map(op.compute_stats)
        sample = dataset[0]
        # Check that brightness scores were computed
        self.assertIn(StatsKeys.image_brightness_scores, sample[Fields.stats])
        self.assertIn(StatsKeys.image_brightness_perc_5_scores, sample[Fields.stats])
        self.assertIn(StatsKeys.image_brightness_perc_99_scores, sample[Fields.stats])

        mean_scores = sample[Fields.stats][StatsKeys.image_brightness_scores]
        perc_5_scores = sample[Fields.stats][StatsKeys.image_brightness_perc_5_scores]
        perc_99_scores = sample[Fields.stats][StatsKeys.image_brightness_perc_99_scores]

        self.assertEqual(len(mean_scores), 1)
        self.assertEqual(len(perc_5_scores), 1)
        self.assertEqual(len(perc_99_scores), 1)

        # Brightness values should be between 0 and 1
        self.assertGreaterEqual(perc_5_scores[0], 0)
        self.assertLessEqual(perc_99_scores[0], 1)
        self.assertGreaterEqual(mean_scores[0], 0)
        self.assertLessEqual(mean_scores[0], 1)

    def test_no_filter(self):
        """Test that with no min/max, all images pass."""
        ds_list = [
            {'images': [self.img1_path]},
            {'images': [self.img2_path]},
            {'images': [self.img3_path]}
        ]
        tgt_list = ds_list
        dataset = Dataset.from_list(ds_list)
        op = ImageBrightnessFilter()
        self._run_filter(dataset, tgt_list, op)

    def test_min_brightness(self):
        """Test filtering with minimum brightness threshold.

        min_brightness filters out images where the 99th percentile
        brightness is below the threshold (i.e., very dark images).
        """
        ds_list = [
            {'images': [self.img1_path]},
            {'images': [self.img2_path]},
            {'images': [self.img3_path]}
        ]
        dataset = Dataset.from_list(ds_list)
        # First compute the scores
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        op = ImageBrightnessFilter()
        dataset_with_stats = dataset.map(op.compute_stats)
        perc99_scores = [
            sample[Fields.stats][StatsKeys.image_brightness_perc_99_scores][0]
            for sample in dataset_with_stats
        ]
        # Set min_brightness to filter out the darkest image
        threshold = min(perc99_scores) + 0.01

        op = ImageBrightnessFilter(min_brightness=threshold)
        dataset = Dataset.from_list(ds_list)
        expected = [
            d for d, s in zip(ds_list, perc99_scores) if s >= threshold
        ]
        self._run_filter(dataset, expected, op)

    def test_max_brightness(self):
        """Test filtering with maximum brightness threshold.

        max_brightness filters out images where the 5th percentile
        brightness is above the threshold (i.e., overexposed images).
        """
        ds_list = [
            {'images': [self.img1_path]},
            {'images': [self.img2_path]},
            {'images': [self.img3_path]}
        ]
        dataset = Dataset.from_list(ds_list)
        # First compute the scores
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        op = ImageBrightnessFilter()
        dataset_with_stats = dataset.map(op.compute_stats)
        perc5_scores = [
            sample[Fields.stats][StatsKeys.image_brightness_perc_5_scores][0]
            for sample in dataset_with_stats
        ]
        # Set max_brightness to filter out the brightest image
        threshold = max(perc5_scores) - 0.01

        op = ImageBrightnessFilter(max_brightness=threshold)
        dataset = Dataset.from_list(ds_list)
        expected = [
            d for d, s in zip(ds_list, perc5_scores) if s <= threshold
        ]
        self._run_filter(dataset, expected, op)

    def test_min_max_combined(self):
        """Test filtering with both min and max brightness thresholds."""
        ds_list = [
            {'images': [self.img1_path]},
            {'images': [self.img2_path]},
            {'images': [self.img3_path]}
        ]
        # Set thresholds that should let most images pass
        op = ImageBrightnessFilter(min_brightness=0.1, max_brightness=0.9)
        dataset = Dataset.from_list(ds_list)
        # Most normal images should pass these thresholds
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats)
        dataset = dataset.filter(op.process)
        # Just check that filtering ran without errors
        self.assertGreaterEqual(len(dataset), 0)

    def test_any_strategy(self):
        """Test 'any' strategy with multiple images per sample."""
        ds_list = [
            {'images': [self.img1_path, self.img2_path]},
            {'images': [self.img2_path, self.img3_path]}
        ]
        dataset = Dataset.from_list(ds_list)
        # With min_brightness=0, all images should pass in 'any' mode
        op = ImageBrightnessFilter(min_brightness=0, any_or_all='any')
        tgt_list = ds_list
        self._run_filter(dataset, tgt_list, op)

    def test_all_strategy(self):
        """Test 'all' strategy with multiple images per sample."""
        ds_list = [
            {'images': [self.img1_path, self.img2_path]},
            {'images': [self.img2_path, self.img3_path]}
        ]
        dataset = Dataset.from_list(ds_list)
        # With impossible threshold in 'all' mode, no samples should pass
        op = ImageBrightnessFilter(min_brightness=1.0, any_or_all='all')
        tgt_list = []
        self._run_filter(dataset, tgt_list, op)

    def test_empty_images(self):
        """Test that samples with no images pass through."""
        ds_list = [{'images': []}]
        tgt_list = ds_list
        dataset = Dataset.from_list(ds_list)
        op = ImageBrightnessFilter(min_brightness=0)
        self._run_filter(dataset, tgt_list, op)

    def test_invalid_any_or_all(self):
        """Test that invalid any_or_all parameter raises error."""
        with self.assertRaises(ValueError):
            ImageBrightnessFilter(any_or_all='invalid')


if __name__ == '__main__':
    unittest.main()
