import numpy as np
from loguru import logger

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import load_data_with_context, load_image

from ...utils.model_utils import get_model, prepare_model
from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_IMAGES

torch = LazyLoader("torch")

OP_NAME = "image_maniqa_filter"


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageManiqaFilter(Filter):
    """Filter to keep samples with MANIQA image quality scores within a
    specific range.

    MANIQA (Multi-dimension Attention Network for No-Reference Image Quality
    Assessment) is a deep learning model that predicts image quality scores
    without access to a reference image. It uses Vision Transformer (ViT)
    with Transposed Attention Block (TAB) and Scale Swin Transformer Block
    (SSTB) for multi-dimensional quality assessment.

    The model was proposed in CVPR Workshop 2022 and won first place in the
    NTIRE 2022 Perceptual Image Quality Assessment Challenge Track 2.

    Reference: https://github.com/IIGROUP/MANIQA
    """

    _accelerator = "cuda"

    def __init__(
        self,
        min_score: float = 0.0,
        max_score: float = 1.0,
        any_or_all: str = "any",
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param min_score: Minimum MANIQA score for images. MANIQA scores
            typically range from 0 to 1, where higher scores indicate
            better quality. Default is 0.0.
        :param max_score: Maximum MANIQA score for images. Default is 1.0.
        :param any_or_all: Strategy for filtering samples with multiple
            images. 'any': keep the sample if any image meets the condition.
            'all': keep the sample only if all images meet the condition.
            Default is 'any'.
        :param args: Extra positional arguments.
        :param kwargs: Extra keyword arguments.
        """
        kwargs["memory"] = "2GB" if kwargs.get("memory", 0) == 0 else kwargs["memory"]
        super().__init__(*args, **kwargs)
        self.min_score = min_score
        self.max_score = max_score

        if any_or_all not in ["any", "all"]:
            raise ValueError(
                f"Keep strategy [{any_or_all}] is not supported. "
                f'Can only be one of ["any", "all"].'
            )
        self.any = any_or_all == "any"

        self.model_key = prepare_model(
            model_type="pyiqa",
            metric_name="maniqa",
        )

    def compute_stats_single(self, sample, rank=None, context=False):
        # check if it's computed already
        if StatsKeys.image_maniqa_scores in sample[Fields.stats]:
            return sample

        # there is no image in this sample
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][StatsKeys.image_maniqa_scores] = np.array(
                [], dtype=np.float64
            )
            return sample

        # load images
        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(
            sample, context, loaded_image_keys, load_image, mm_bytes_key=self.image_bytes_key
        )

        # compute MANIQA scores
        model = get_model(self.model_key, rank, self.use_cuda())

        maniqa_scores = []
        for image in images.values():
            with torch.no_grad():
                # pyiqa accepts PIL images directly
                score = model(image)
                maniqa_scores.append(score.item())

        logger.debug(f"maniqa_scores: {maniqa_scores}")

        sample[Fields.stats][StatsKeys.image_maniqa_scores] = maniqa_scores
        return sample

    def process_single(self, sample):
        maniqa_scores = sample[Fields.stats][StatsKeys.image_maniqa_scores]
        if len(maniqa_scores) <= 0:
            return True

        keep_bools = np.array(
            [
                self.get_keep_boolean(maniqa_score, self.min_score, self.max_score)
                for maniqa_score in maniqa_scores
            ]
        )

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
