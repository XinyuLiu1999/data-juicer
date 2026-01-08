import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.filter.image_bluriness_filter import ImageBlurrinessFilter
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ImageBlurrinessFilterTest(DataJuicerTestCaseBase):
    """Test cases for ImageBlurrinessFilter.

    The blurriness metric is based on edge detection variance:
    - Higher values indicate sharper images (more edges detected)
    - Lower values indicate blurrier images (fewer edges detected)
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
        """Test that compute_stats correctly calculates blurriness scores."""
        ds_list = [{'images': [self.img1_path]}]
        dataset = Dataset.from_list(ds_list)
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        op = ImageBlurrinessFilter()
        dataset = dataset.map(op.compute_stats)
        sample = dataset[0]
        # Check that blurriness scores were computed
        self.assertIn(StatsKeys.image_blurriness_scores, sample[Fields.stats])
        scores = sample[Fields.stats][StatsKeys.image_blurriness_scores]
        self.assertEqual(len(scores), 1)
        # Blurriness score should be a positive float
        self.assertIsInstance(scores[0], float)
        self.assertGreater(scores[0], 0)

    def test_no_filter(self):
        """Test that with no min/max, all images pass."""
        ds_list = [
            {'images': [self.img1_path]},
            {'images': [self.img2_path]},
            {'images': [self.img3_path]}
        ]
        tgt_list = [
            {'images': [self.img1_path]},
            {'images': [self.img2_path]},
            {'images': [self.img3_path]}
        ]
        dataset = Dataset.from_list(ds_list)
        op = ImageBlurrinessFilter()
        self._run_filter(dataset, tgt_list, op)

    def test_min_blurriness(self):
        """Test filtering with minimum blurriness threshold."""
        ds_list = [
            {'images': [self.img1_path]},
            {'images': [self.img2_path]},
            {'images': [self.img3_path]}
        ]
        dataset = Dataset.from_list(ds_list)
        # First compute the scores to find a good threshold
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        op = ImageBlurrinessFilter()
        dataset_with_stats = dataset.map(op.compute_stats)
        scores = [
            sample[Fields.stats][StatsKeys.image_blurriness_scores][0]
            for sample in dataset_with_stats
        ]
        # Set min_blurriness to filter out the lowest scoring image
        min_val = min(scores)
        max_val = max(scores)
        threshold = (min_val + max_val) / 2

        op = ImageBlurrinessFilter(min_blurriness=threshold)
        dataset = Dataset.from_list(ds_list)
        self._run_filter(dataset, [
            d for d, s in zip(ds_list, scores) if s >= threshold
        ], op)

    def test_max_blurriness(self):
        """Test filtering with maximum blurriness threshold."""
        ds_list = [
            {'images': [self.img1_path]},
            {'images': [self.img2_path]},
            {'images': [self.img3_path]}
        ]
        dataset = Dataset.from_list(ds_list)
        # First compute the scores to find a good threshold
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        op = ImageBlurrinessFilter()
        dataset_with_stats = dataset.map(op.compute_stats)
        scores = [
            sample[Fields.stats][StatsKeys.image_blurriness_scores][0]
            for sample in dataset_with_stats
        ]
        # Set max_blurriness to filter out the highest scoring image
        min_val = min(scores)
        max_val = max(scores)
        threshold = (min_val + max_val) / 2

        op = ImageBlurrinessFilter(max_blurriness=threshold)
        dataset = Dataset.from_list(ds_list)
        self._run_filter(dataset, [
            d for d, s in zip(ds_list, scores) if s <= threshold
        ], op)

    def test_any_strategy(self):
        """Test 'any' strategy with multiple images per sample."""
        ds_list = [
            {'images': [self.img1_path, self.img2_path]},
            {'images': [self.img2_path, self.img3_path]},
            {'images': [self.img1_path, self.img3_path]}
        ]
        dataset = Dataset.from_list(ds_list)
        # With very high min_blurriness, at least one image should pass in 'any' mode
        # if any image has high enough score
        op = ImageBlurrinessFilter(min_blurriness=0, any_or_all='any')
        # All should pass with min_blurriness=0
        tgt_list = ds_list
        self._run_filter(dataset, tgt_list, op)

    def test_all_strategy(self):
        """Test 'all' strategy with multiple images per sample."""
        ds_list = [
            {'images': [self.img1_path, self.img2_path]},
            {'images': [self.img2_path, self.img3_path]},
            {'images': [self.img1_path, self.img3_path]}
        ]
        dataset = Dataset.from_list(ds_list)
        # Compute scores first
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        op = ImageBlurrinessFilter()
        dataset_with_stats = dataset.map(op.compute_stats)

        # With very high threshold in 'all' mode, samples with mixed scores fail
        # Set threshold high enough that not all images pass
        op = ImageBlurrinessFilter(min_blurriness=1000, any_or_all='all')
        dataset = Dataset.from_list(ds_list)
        tgt_list = []  # Likely no sample will have all images with score >= 1000
        self._run_filter(dataset, tgt_list, op)

    def test_empty_images(self):
        """Test that samples with no images pass through."""
        ds_list = [
            {'images': []},
            {'images': [self.img1_path]}
        ]
        tgt_list = ds_list
        dataset = Dataset.from_list(ds_list)
        op = ImageBlurrinessFilter(min_blurriness=0)
        self._run_filter(dataset, tgt_list, op)

    def test_invalid_any_or_all(self):
        """Test that invalid any_or_all parameter raises error."""
        with self.assertRaises(ValueError):
            ImageBlurrinessFilter(any_or_all='invalid')


if __name__ == '__main__':
    unittest.main()
