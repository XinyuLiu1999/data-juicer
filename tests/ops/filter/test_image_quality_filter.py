import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.filter.image_quality_filter import ImageQualityFilter
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ImageQualityFilterTest(DataJuicerTestCaseBase):

    maxDiff = None

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    img1_path = os.path.join(data_path, 'img1.png')  # square image
    img2_path = os.path.join(data_path, 'img2.jpg')  # landscape image
    img3_path = os.path.join(data_path, 'img3.jpg')  # portrait image
    cat_path = os.path.join(data_path, 'cat.jpg')
    blip_path = os.path.join(data_path, 'blip.jpg')

    def _run_image_quality_filter(self, dataset: Dataset, target_list, op,
                                  num_proc=1):
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats, num_proc=num_proc)
        dataset = dataset.filter(op.process, num_proc=num_proc)
        dataset = dataset.select_columns(column_names=[op.image_key])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_blurriness_filter(self):
        """Test filtering based on blurriness threshold."""
        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        dataset = Dataset.from_list(ds_list)

        # First, compute stats to see what values we get
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        op = ImageQualityFilter(min_blurriness=0)  # Compute all
        dataset = dataset.map(op.compute_stats)

        # Check that blurriness scores were computed
        for sample in dataset:
            self.assertIn(StatsKeys.image_blurriness_scores,
                          sample[Fields.stats])
            scores = sample[Fields.stats][StatsKeys.image_blurriness_scores]
            self.assertEqual(len(scores), 1)
            self.assertIsInstance(scores[0], float)
            self.assertGreater(scores[0], 0)

    def test_brightness_filter(self):
        """Test filtering based on brightness threshold."""
        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.cat_path]
        }]
        dataset = Dataset.from_list(ds_list)

        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        op = ImageQualityFilter(min_brightness=0)  # Compute all
        dataset = dataset.map(op.compute_stats)

        # Check that brightness scores were computed
        for sample in dataset:
            self.assertIn(StatsKeys.image_brightness_scores,
                          sample[Fields.stats])
            scores = sample[Fields.stats][StatsKeys.image_brightness_scores]
            self.assertEqual(len(scores), 1)
            self.assertIsInstance(scores[0], float)
            # Brightness should be between 0 and 1
            self.assertGreaterEqual(scores[0], 0)
            self.assertLessEqual(scores[0], 1)

    def test_entropy_filter(self):
        """Test filtering based on entropy threshold."""
        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.cat_path]
        }, {
            'images': [self.blip_path]
        }]
        dataset = Dataset.from_list(ds_list)

        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        op = ImageQualityFilter(min_entropy=0)  # Compute all
        dataset = dataset.map(op.compute_stats)

        # Check that entropy scores were computed
        for sample in dataset:
            self.assertIn(StatsKeys.image_entropy_scores, sample[Fields.stats])
            scores = sample[Fields.stats][StatsKeys.image_entropy_scores]
            self.assertEqual(len(scores), 1)
            self.assertIsInstance(scores[0], float)
            self.assertGreater(scores[0], 0)

    def test_grayscale_filter(self):
        """Test filtering out grayscale images."""
        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.cat_path]
        }, {
            'images': [self.blip_path]
        }]
        dataset = Dataset.from_list(ds_list)

        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        op = ImageQualityFilter(keep_grayscale=False)
        dataset = dataset.map(op.compute_stats)

        # Check that grayscale flags were computed
        for sample in dataset:
            self.assertIn(StatsKeys.image_grayscale_flags, sample[Fields.stats])
            flags = sample[Fields.stats][StatsKeys.image_grayscale_flags]
            self.assertEqual(len(flags), 1)
            self.assertIsInstance(flags[0], (bool, type(True)))

    def test_combined_filter(self):
        """Test filtering with multiple criteria combined."""
        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.cat_path]
        }]
        dataset = Dataset.from_list(ds_list)

        # Apply filter with multiple criteria
        op = ImageQualityFilter(
            min_blurriness=1.0,  # Very low threshold - most should pass
            min_brightness=0.01,  # Very low threshold - most should pass
            min_entropy=0.1,  # Very low threshold - most should pass
        )

        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats)
        filtered = dataset.filter(op.process)

        # With very low thresholds, all images should pass
        self.assertEqual(len(filtered), 3)

    def test_strict_filter(self):
        """Test filtering with strict criteria."""
        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.cat_path]
        }]
        dataset = Dataset.from_list(ds_list)

        # Apply filter with very strict criteria
        op = ImageQualityFilter(
            min_blurriness=1000.0,  # Very high threshold - nothing should pass
        )

        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats)
        filtered = dataset.filter(op.process)

        # With very high threshold, no images should pass
        self.assertEqual(len(filtered), 0)

    def test_any_strategy(self):
        """Test 'any' strategy with multiple images per sample."""
        ds_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.cat_path, self.blip_path]
        }]
        dataset = Dataset.from_list(ds_list)

        # Apply filter with 'any' strategy
        op = ImageQualityFilter(
            min_blurriness=1.0,
            any_or_all='any'
        )

        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats)
        filtered = dataset.filter(op.process)

        # With 'any' strategy and low threshold, samples should pass
        # if at least one image passes
        self.assertGreaterEqual(len(filtered), 1)

    def test_all_strategy(self):
        """Test 'all' strategy with multiple images per sample."""
        ds_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.cat_path, self.blip_path]
        }]
        dataset = Dataset.from_list(ds_list)

        # Apply filter with 'all' strategy and impossibly high threshold
        op = ImageQualityFilter(
            min_blurriness=10000.0,
            any_or_all='all'
        )

        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats)
        filtered = dataset.filter(op.process)

        # With 'all' strategy and very high threshold, nothing should pass
        self.assertEqual(len(filtered), 0)

    def test_empty_images(self):
        """Test handling of samples with no images."""
        ds_list = [{
            'images': []
        }, {
            'images': [self.img1_path]
        }]
        dataset = Dataset.from_list(ds_list)

        op = ImageQualityFilter(min_blurriness=1.0)

        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats)
        filtered = dataset.filter(op.process)

        # Empty image samples should not be filtered out
        self.assertEqual(len(filtered), 2)

    def test_no_filter_criteria(self):
        """Test that samples pass when no filter criteria are specified."""
        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }]
        dataset = Dataset.from_list(ds_list)

        # No filter criteria specified
        op = ImageQualityFilter()

        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats)
        filtered = dataset.filter(op.process)

        # All samples should pass when no criteria are specified
        self.assertEqual(len(filtered), 2)

    def test_multi_process(self):
        """Test filter with multiple processes."""
        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.cat_path]
        }]
        tgt_list = [
            {'images': [self.img1_path]},
            {'images': [self.img2_path]},
            {'images': [self.cat_path]}
        ]
        dataset = Dataset.from_list(ds_list)
        op = ImageQualityFilter(
            min_blurriness=1.0,
            min_entropy=0.1
        )
        self._run_image_quality_filter(dataset, tgt_list, op, num_proc=2)


if __name__ == '__main__':
    unittest.main()
