import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import load_data_with_context, load_image
from ..base_op import OPERATORS, Filter


def is_grayscale(image):
    return image.mode == "L"


@OPERATORS.register_module("image_grayscale_filter")
class ImageGrayscaleFilter(Filter):
    def __init__(self, keep_grayscale=True, any_or_all="any", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if any_or_all not in ["any", "all"]:
            raise ValueError("any_or_all must be 'any' or 'all'")
        self.any = any_or_all == "any"
        self.keep_grayscale = keep_grayscale

    def compute_stats_single(self, sample, context=False):
        if StatsKeys.image_grayscale_flags in sample[Fields.stats]:
            return sample

        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][StatsKeys.image_grayscale_flags] = []
            return sample

        keys = sample[self.image_key]
        sample, images = load_data_with_context(
            sample, context, keys, load_image, mm_bytes_key=self.image_bytes_key
        )

        sample[Fields.stats][StatsKeys.image_grayscale_flags] = [
            is_grayscale(images[k]) for k in keys
        ]
        return sample

    def process_single(self, sample):
        flags = sample[Fields.stats][StatsKeys.image_grayscale_flags]
        if not flags:
            return True

        if self.keep_grayscale:
            keep = np.ones(len(flags), dtype=bool)
        else:
            keep = np.array([not f for f in flags])

        return keep.any() if self.any else keep.all()
