import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.filter.image_entropy_filter import ImageEntropyFilter
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ImageEntropyFilterTest(DataJuicerTestCaseBase):
    """Test cases for ImageEntropyFilter.

    The entropy filter uses PIL's built-in entropy calculation:
    - Low entropy indicates uniform/simple images (solid colors, patterns)
    - High entropy indicates complex/noisy images (detailed textures, noise)

    Typical entropy values range from 0 (solid color) to ~8 (random noise).
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
        """Test that compute_stats correctly calculates entropy scores."""
        ds_list = [{'images': [self.img1_path]}]
        dataset = Dataset.from_list(ds_list)
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        op = ImageEntropyFilter()
        dataset = dataset.map(op.compute_stats)
        sample = dataset[0]
        # Check that entropy scores were computed
        self.assertIn(StatsKeys.image_entropy_scores, sample[Fields.stats])
        scores = sample[Fields.stats][StatsKeys.image_entropy_scores]
        self.assertEqual(len(scores), 1)
        # Entropy should be a non-negative float
        self.assertIsInstance(scores[0], float)
        self.assertGreaterEqual(scores[0], 0)
        # Entropy is typically between 0 and 8 for 8-bit images
        self.assertLessEqual(scores[0], 10)

    def test_no_filter(self):
        """Test that with no min/max, all images pass."""
        ds_list = [
            {'images': [self.img1_path]},
            {'images': [self.img2_path]},
            {'images': [self.img3_path]}
        ]
        tgt_list = ds_list
        dataset = Dataset.from_list(ds_list)
        op = ImageEntropyFilter()
        self._run_filter(dataset, tgt_list, op)

    def test_min_entropy(self):
        """Test filtering with minimum entropy threshold.

        This filters out images with low entropy (uniform/simple images).
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
        op = ImageEntropyFilter()
        dataset_with_stats = dataset.map(op.compute_stats)
        scores = [
            sample[Fields.stats][StatsKeys.image_entropy_scores][0]
            for sample in dataset_with_stats
        ]
        # Set threshold to filter out the lowest entropy image
        min_val = min(scores)
        max_val = max(scores)
        threshold = (min_val + max_val) / 2

        op = ImageEntropyFilter(min_entropy=threshold)
        dataset = Dataset.from_list(ds_list)
        expected = [
            d for d, s in zip(ds_list, scores) if s >= threshold
        ]
        self._run_filter(dataset, expected, op)

    def test_max_entropy(self):
        """Test filtering with maximum entropy threshold.

        This filters out images with high entropy (noisy/complex images).
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
        op = ImageEntropyFilter()
        dataset_with_stats = dataset.map(op.compute_stats)
        scores = [
            sample[Fields.stats][StatsKeys.image_entropy_scores][0]
            for sample in dataset_with_stats
        ]
        # Set threshold to filter out the highest entropy image
        min_val = min(scores)
        max_val = max(scores)
        threshold = (min_val + max_val) / 2

        op = ImageEntropyFilter(max_entropy=threshold)
        dataset = Dataset.from_list(ds_list)
        expected = [
            d for d, s in zip(ds_list, scores) if s <= threshold
        ]
        self._run_filter(dataset, expected, op)

    def test_min_max_combined(self):
        """Test filtering with both min and max entropy thresholds."""
        ds_list = [
            {'images': [self.img1_path]},
            {'images': [self.img2_path]},
            {'images': [self.img3_path]}
        ]
        dataset = Dataset.from_list(ds_list)
        # First compute the scores to get a sensible range
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        op = ImageEntropyFilter()
        dataset_with_stats = dataset.map(op.compute_stats)
        scores = [
            sample[Fields.stats][StatsKeys.image_entropy_scores][0]
            for sample in dataset_with_stats
        ]
        min_score = min(scores)
        max_score = max(scores)

        # Set range to include all images
        op = ImageEntropyFilter(min_entropy=min_score - 0.1,
                                max_entropy=max_score + 0.1)
        dataset = Dataset.from_list(ds_list)
        tgt_list = ds_list
        self._run_filter(dataset, tgt_list, op)

    def test_any_strategy(self):
        """Test 'any' strategy with multiple images per sample."""
        ds_list = [
            {'images': [self.img1_path, self.img2_path]},
            {'images': [self.img2_path, self.img3_path]}
        ]
        dataset = Dataset.from_list(ds_list)
        # With min_entropy=0, all images should pass in 'any' mode
        op = ImageEntropyFilter(min_entropy=0, any_or_all='any')
        tgt_list = ds_list
        self._run_filter(dataset, tgt_list, op)

    def test_all_strategy(self):
        """Test 'all' strategy with multiple images per sample."""
        ds_list = [
            {'images': [self.img1_path, self.img2_path]},
            {'images': [self.img2_path, self.img3_path]}
        ]
        dataset = Dataset.from_list(ds_list)
        # With impossibly high threshold in 'all' mode, no samples should pass
        op = ImageEntropyFilter(min_entropy=100, any_or_all='all')
        tgt_list = []
        self._run_filter(dataset, tgt_list, op)

    def test_empty_images(self):
        """Test that samples with no images pass through."""
        ds_list = [{'images': []}]
        tgt_list = ds_list
        dataset = Dataset.from_list(ds_list)
        op = ImageEntropyFilter(min_entropy=0)
        self._run_filter(dataset, tgt_list, op)

    def test_invalid_any_or_all(self):
        """Test that invalid any_or_all parameter raises error."""
        with self.assertRaises(ValueError):
            ImageEntropyFilter(any_or_all='invalid')


if __name__ == '__main__':
    unittest.main()
