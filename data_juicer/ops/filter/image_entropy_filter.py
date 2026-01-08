import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import load_data_with_context, load_image
from ..base_op import OPERATORS, Filter


def calc_entropy(image):
    return float(image.entropy())


@OPERATORS.register_module("image_entropy_filter")
class ImageEntropyFilter(Filter):
    def __init__(self, min_entropy=None, max_entropy=None, any_or_all="any", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if any_or_all not in ["any", "all"]:
            raise ValueError("any_or_all must be 'any' or 'all'")
        self.any = any_or_all == "any"
        self.min_entropy = min_entropy
        self.max_entropy = max_entropy

    def compute_stats_single(self, sample, context=False):
        if StatsKeys.image_entropy_scores in sample[Fields.stats]:
            return sample

        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][StatsKeys.image_entropy_scores] = []
            return sample

        keys = sample[self.image_key]
        sample, images = load_data_with_context(
            sample, context, keys, load_image, mm_bytes_key=self.image_bytes_key
        )

        sample[Fields.stats][StatsKeys.image_entropy_scores] = [
            calc_entropy(images[k]) for k in keys
        ]
        return sample

    def process_single(self, sample):
        vals = sample[Fields.stats][StatsKeys.image_entropy_scores]
        if not vals:
            return True

        keep = []
        for v in vals:
            if self.min_entropy is not None and v < self.min_entropy:
                keep.append(False)
            elif self.max_entropy is not None and v > self.max_entropy:
                keep.append(False)
            else:
                keep.append(True)

        keep = np.array(keep)
        return keep.any() if self.any else keep.all()
