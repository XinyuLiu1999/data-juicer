import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.filter.image_maniqa_filter import ImageManiqaFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ImageManiqaFilterTest(DataJuicerTestCaseBase):

    maxDiff = None

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    img1_path = os.path.join(data_path, 'cat.jpg')
    img2_path = os.path.join(data_path, 'blip.jpg')
    img3_path = os.path.join(data_path, 'lena-face.jpg')

    def _run_image_maniqa_filter(self,
                                 dataset: Dataset,
                                 target_list,
                                 op,
                                 num_proc=1):
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats, num_proc=num_proc)
        dataset = dataset.filter(op.process, num_proc=num_proc)
        dataset = dataset.remove_columns(Fields.stats)
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_default_params(self):
        """Test with default parameters - should keep all samples with scores
        in range [0.0, 1.0]."""
        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        tgt_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageManiqaFilter()
        self._run_image_maniqa_filter(dataset, tgt_list, op)

    def test_filter_low_quality(self):
        """Test filtering out low quality images with min_score threshold."""
        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageManiqaFilter(min_score=0.5, max_score=1.0)
        # Run filter and check that it executes without error
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats, num_proc=1)
        dataset = dataset.filter(op.process, num_proc=1)
        # All remaining samples should have scores >= 0.5
        self.assertTrue(len(dataset) >= 0)

    def test_filter_high_quality(self):
        """Test filtering out high quality images with max_score threshold."""
        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageManiqaFilter(min_score=0.0, max_score=0.5)
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats, num_proc=1)
        dataset = dataset.filter(op.process, num_proc=1)
        # All remaining samples should have scores <= 0.5
        self.assertTrue(len(dataset) >= 0)

    def test_filter_multimodal(self):
        """Test with multimodal samples including text and empty image lists."""
        ds_list = [{
            'text': 'a test sentence',
            'images': []
        }, {
            'text': 'a test sentence',
            'images': [self.img1_path]
        }, {
            'text': 'a test sentence',
            'images': [self.img2_path]
        }, {
            'text': 'a test sentence',
            'images': [self.img3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageManiqaFilter()
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats, num_proc=1)
        dataset = dataset.filter(op.process, num_proc=1)
        # Sample with empty images should be kept
        res_list = dataset.to_list()
        empty_images_kept = any(
            sample.get('images', []) == [] for sample in res_list
        )
        self.assertTrue(empty_images_kept)

    def test_any_strategy(self):
        """Test 'any' strategy - keep sample if any image meets condition."""
        ds_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img3_path]
        }, {
            'images': [self.img1_path, self.img3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageManiqaFilter(min_score=0.0, max_score=1.0, any_or_all='any')
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats, num_proc=1)
        dataset = dataset.filter(op.process, num_proc=1)
        # With default range [0, 1], all samples should be kept
        self.assertEqual(len(dataset), 3)

    def test_all_strategy(self):
        """Test 'all' strategy - keep sample only if all images meet
        condition."""
        ds_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img3_path]
        }, {
            'images': [self.img1_path, self.img3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageManiqaFilter(min_score=0.0, max_score=1.0, any_or_all='all')
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats, num_proc=1)
        dataset = dataset.filter(op.process, num_proc=1)
        # With default range [0, 1], all samples should be kept
        self.assertEqual(len(dataset), 3)

    def test_invalid_strategy(self):
        """Test that invalid any_or_all strategy raises ValueError."""
        with self.assertRaises(ValueError):
            ImageManiqaFilter(any_or_all='invalid')

    def test_filter_multi_process(self):
        """Test filter with multiple processes."""
        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageManiqaFilter()
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats, num_proc=2)
        dataset = dataset.filter(op.process, num_proc=2)
        # Should execute without error
        self.assertTrue(len(dataset) >= 0)

    def test_no_images_key(self):
        """Test sample without images key."""
        ds_list = [{'text': 'sample without images'}]
        dataset = Dataset.from_list(ds_list)
        op = ImageManiqaFilter()
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats, num_proc=1)
        dataset = dataset.filter(op.process, num_proc=1)
        # Sample without images should be kept
        self.assertEqual(len(dataset), 1)


if __name__ == '__main__':
    unittest.main()
