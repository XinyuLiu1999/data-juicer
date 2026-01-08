import numpy as np
from PIL import ImageFilter, ImageStat

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import load_data_with_context, load_image
from ..base_op import OPERATORS, Filter

MAX_RESOLUTION_FOR_BLURRY_DETECTION = 512


def calc_blurriness(image):
    ratio = max(image.width, image.height) / MAX_RESOLUTION_FOR_BLURRY_DETECTION
    if ratio > 1:
        image = image.resize(
            (max(int(image.width // ratio), 1), max(int(image.height // ratio), 1))
        )

    gray = image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    var = ImageStat.Stat(edges).var[0]
    return float(np.sqrt(var))


@OPERATORS.register_module("image_blurriness_filter")
class ImageBlurrinessFilter(Filter):
    def __init__(self, min_blurriness=None, max_blurriness=None, any_or_all="any", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if any_or_all not in ["any", "all"]:
            raise ValueError("any_or_all must be 'any' or 'all'")
        self.any = any_or_all == "any"
        self.min_blurriness = min_blurriness
        self.max_blurriness = max_blurriness

    def compute_stats_single(self, sample, context=False):
        if StatsKeys.image_blurriness_scores in sample[Fields.stats]:
            return sample

        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][StatsKeys.image_blurriness_scores] = []
            return sample

        keys = sample[self.image_key]
        sample, images = load_data_with_context(
            sample, context, keys, load_image, mm_bytes_key=self.image_bytes_key
        )

        sample[Fields.stats][StatsKeys.image_blurriness_scores] = [
            calc_blurriness(images[k]) for k in keys
        ]
        return sample

    def process_single(self, sample):
        vals = sample[Fields.stats][StatsKeys.image_blurriness_scores]
        if not vals:
            return True

        keep = []
        for v in vals:
            if self.min_blurriness is not None and v < self.min_blurriness:
                keep.append(False)
            elif self.max_blurriness is not None and v > self.max_blurriness:
                keep.append(False)
            else:
                keep.append(True)

        keep = np.array(keep)
        return keep.any() if self.any else keep.all()
