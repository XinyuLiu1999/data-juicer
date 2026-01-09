# flake8: noqa: E501

import os
import unittest
from unittest.mock import MagicMock, patch

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.filter.image_safe_aigc_filter import ImageSafeAigcFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ImageSafeAigcFilterTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..",
                             "data")

    img1_path = os.path.join(data_path, "img1.png")
    img2_path = os.path.join(data_path, "img2.jpg")
    img3_path = os.path.join(data_path, "img3.jpg")

    def _run_filter(self, dataset: Dataset, target_list, op, num_proc=1):
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)

        dataset = dataset.map(op.compute_stats,
                              num_proc=num_proc,
                              with_rank=True)
        dataset = dataset.filter(op.process, num_proc=num_proc)
        dataset = dataset.select_columns(column_names=["images"])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_init_default(self):
        """Test filter initialization with default parameters."""
        with patch(
            "data_juicer.utils.model_utils.prepare_safe_model"
        ) as mock_prepare:
            mock_prepare.return_value = (MagicMock(), MagicMock())
            op = ImageSafeAigcFilter()
            self.assertEqual(op.min_score, 0.0)
            self.assertEqual(op.max_score, 0.5)
            self.assertTrue(op.any)

    def test_init_custom(self):
        """Test filter initialization with custom parameters."""
        with patch(
            "data_juicer.utils.model_utils.prepare_safe_model"
        ) as mock_prepare:
            mock_prepare.return_value = (MagicMock(), MagicMock())
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
            "data_juicer.utils.model_utils.prepare_safe_model"
        ) as mock_prepare:
            mock_prepare.return_value = (MagicMock(), MagicMock())
            with self.assertRaises(ValueError):
                ImageSafeAigcFilter(any_or_all="invalid")

    @patch("data_juicer.ops.filter.image_safe_aigc_filter.get_model")
    def test_filter_with_mock(self, mock_get_model):
        """Test filter with mocked model."""
        import torch

        # Create mock model that returns different scores for different images
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter(
            [torch.nn.Parameter(torch.zeros(1))]
        )

        # Mock to return different scores: img1=0.8 (fake), img2=0.2 (real), img3=0.3 (real)
        call_count = [0]
        def mock_forward(x):
            scores = [0.8, 0.2, 0.3]  # AIGC scores for each image
            score = scores[call_count[0] % len(scores)]
            call_count[0] += 1
            return torch.tensor([[1 - score, score]])  # [real_prob, fake_prob]

        mock_model.side_effect = mock_forward
        mock_model.__call__ = mock_forward

        # Create mock transform
        mock_transform = MagicMock()
        mock_transform.return_value = torch.zeros(3, 224, 224)

        mock_get_model.return_value = (mock_model, mock_transform)

        ds_list = [
            {"images": [self.img1_path]},  # score=0.8 (fake) - filtered out
            {"images": [self.img2_path]},  # score=0.2 (real) - kept
            {"images": [self.img3_path]},  # score=0.3 (real) - kept
        ]
        tgt_list = [
            {"images": [self.img2_path]},
            {"images": [self.img3_path]},
        ]

        dataset = Dataset.from_list(ds_list)

        # Mock the model preparation
        with patch(
            "data_juicer.utils.model_utils.prepare_model"
        ) as mock_prepare:
            mock_prepare.return_value = MagicMock()
            op = ImageSafeAigcFilter(max_score=0.5)
            op.model_key = MagicMock()

        self._run_filter(dataset, tgt_list, op)

    @patch("data_juicer.ops.filter.image_safe_aigc_filter.get_model")
    def test_any_strategy(self, mock_get_model):
        """Test 'any' strategy: keep if any image passes."""
        import torch

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter(
            [torch.nn.Parameter(torch.zeros(1))]
        )

        # Track which image we're processing
        call_count = [0]
        # Scores: img1=0.8, img2=0.2, img3=0.3
        scores_map = {0: 0.8, 1: 0.2, 2: 0.3}

        def mock_forward(x):
            idx = call_count[0]
            call_count[0] += 1
            # Map: 0,3->0.8, 1,4->0.2, 2,5->0.3
            score = scores_map[idx % 3]
            return torch.tensor([[1 - score, score]])

        mock_model.__call__ = mock_forward

        mock_transform = MagicMock()
        mock_transform.return_value = torch.zeros(3, 224, 224)

        mock_get_model.return_value = (mock_model, mock_transform)

        # Sample 1: img1 (0.8) + img2 (0.2) - img2 passes, keep
        # Sample 2: img1 (0.8) + img3 (0.3) - img3 passes, keep
        ds_list = [
            {"images": [self.img1_path, self.img2_path]},
            {"images": [self.img1_path, self.img3_path]},
        ]
        tgt_list = [
            {"images": [self.img1_path, self.img2_path]},
            {"images": [self.img1_path, self.img3_path]},
        ]

        dataset = Dataset.from_list(ds_list)

        with patch(
            "data_juicer.utils.model_utils.prepare_model"
        ) as mock_prepare:
            mock_prepare.return_value = MagicMock()
            op = ImageSafeAigcFilter(max_score=0.5, any_or_all="any")
            op.model_key = MagicMock()

        self._run_filter(dataset, tgt_list, op)

    @patch("data_juicer.ops.filter.image_safe_aigc_filter.get_model")
    def test_all_strategy(self, mock_get_model):
        """Test 'all' strategy: keep only if all images pass."""
        import torch

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter(
            [torch.nn.Parameter(torch.zeros(1))]
        )

        # Scores: img2=0.2 (real), img3=0.3 (real), img1=0.8 (fake)
        call_count = [0]

        def mock_forward(x):
            idx = call_count[0]
            call_count[0] += 1
            # First sample: img2 (0.2), img3 (0.3) - both pass
            # Second sample: img1 (0.8), img2 (0.2) - img1 fails
            scores = [0.2, 0.3, 0.8, 0.2]
            score = scores[idx % len(scores)]
            return torch.tensor([[1 - score, score]])

        mock_model.__call__ = mock_forward

        mock_transform = MagicMock()
        mock_transform.return_value = torch.zeros(3, 224, 224)

        mock_get_model.return_value = (mock_model, mock_transform)

        ds_list = [
            {"images": [self.img2_path, self.img3_path]},  # both pass
            {"images": [self.img1_path, self.img2_path]},  # img1 fails
        ]
        tgt_list = [
            {"images": [self.img2_path, self.img3_path]},  # only this kept
        ]

        dataset = Dataset.from_list(ds_list)

        with patch(
            "data_juicer.utils.model_utils.prepare_model"
        ) as mock_prepare:
            mock_prepare.return_value = MagicMock()
            op = ImageSafeAigcFilter(max_score=0.5, any_or_all="all")
            op.model_key = MagicMock()

        self._run_filter(dataset, tgt_list, op)

    @patch("data_juicer.ops.filter.image_safe_aigc_filter.get_model")
    def test_empty_images(self, mock_get_model):
        """Test handling of samples with no images."""
        import torch

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter(
            [torch.nn.Parameter(torch.zeros(1))]
        )
        mock_transform = MagicMock()
        mock_get_model.return_value = (mock_model, mock_transform)

        ds_list = [
            {"images": []},
            {"images": [self.img1_path]},
        ]
        # Empty images sample should be kept (no images to filter)
        tgt_list = [
            {"images": []},
        ]

        dataset = Dataset.from_list(ds_list)

        with patch(
            "data_juicer.utils.model_utils.prepare_model"
        ) as mock_prepare:
            mock_prepare.return_value = MagicMock()
            op = ImageSafeAigcFilter(max_score=0.5)
            op.model_key = MagicMock()

        # Mock to return high score (0.8) for the single image
        call_count = [0]
        def mock_forward(x):
            call_count[0] += 1
            return torch.tensor([[0.2, 0.8]])  # AIGC score = 0.8

        mock_model.__call__ = mock_forward

        self._run_filter(dataset, tgt_list, op)

    @patch("data_juicer.ops.filter.image_safe_aigc_filter.get_model")
    def test_min_score(self, mock_get_model):
        """Test filtering with min_score threshold."""
        import torch

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter(
            [torch.nn.Parameter(torch.zeros(1))]
        )

        call_count = [0]
        def mock_forward(x):
            # Return scores: 0.05, 0.2, 0.4
            scores = [0.05, 0.2, 0.4]
            score = scores[call_count[0] % len(scores)]
            call_count[0] += 1
            return torch.tensor([[1 - score, score]])

        mock_model.__call__ = mock_forward

        mock_transform = MagicMock()
        mock_transform.return_value = torch.zeros(3, 224, 224)

        mock_get_model.return_value = (mock_model, mock_transform)

        ds_list = [
            {"images": [self.img1_path]},  # score=0.05 - below min, filtered
            {"images": [self.img2_path]},  # score=0.2 - in range, kept
            {"images": [self.img3_path]},  # score=0.4 - in range, kept
        ]
        tgt_list = [
            {"images": [self.img2_path]},
            {"images": [self.img3_path]},
        ]

        dataset = Dataset.from_list(ds_list)

        with patch(
            "data_juicer.utils.model_utils.prepare_model"
        ) as mock_prepare:
            mock_prepare.return_value = MagicMock()
            # Filter images with score between 0.1 and 0.5
            op = ImageSafeAigcFilter(min_score=0.1, max_score=0.5)
            op.model_key = MagicMock()

        self._run_filter(dataset, tgt_list, op)


if __name__ == "__main__":
    unittest.main()
