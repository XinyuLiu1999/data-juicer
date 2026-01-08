import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import load_data_with_context, load_image
from ..base_op import OPERATORS, Filter


def calculate_brightness(r, g, b):
    return np.sqrt(0.241 * r * r + 0.691 * g * g + 0.068 * b * b) / 255.0


def calc_brightness_stats(image):
    arr = np.asarray(image)

    if arr.ndim == 3:
        r = arr[:, :, 0].astype(int)
        g = arr[:, :, 1].astype(int)
        b = arr[:, :, 2].astype(int)
        brightness = calculate_brightness(r, g, b)
    else:
        brightness = arr / 255.0

    return {
        "brightness_perc_5": float(np.percentile(brightness, 5)),
        "brightness_perc_99": float(np.percentile(brightness, 99)),
        "brightness_mean": float(brightness.mean()),
    }


@OPERATORS.register_module("image_brightness_filter")
class ImageBrightnessFilter(Filter):
    def __init__(self, min_brightness=None, max_brightness=None, any_or_all="any", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if any_or_all not in ["any", "all"]:
            raise ValueError("any_or_all must be 'any' or 'all'")
        self.any = any_or_all == "any"
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

    def compute_stats_single(self, sample, context=False):
        if (
            StatsKeys.image_brightness_scores in sample[Fields.stats]
            and StatsKeys.image_brightness_perc_5_scores in sample[Fields.stats]
            and StatsKeys.image_brightness_perc_99_scores in sample[Fields.stats]
        ):
            return sample

        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][StatsKeys.image_brightness_scores] = []
            sample[Fields.stats][StatsKeys.image_brightness_perc_5_scores] = []
            sample[Fields.stats][StatsKeys.image_brightness_perc_99_scores] = []
            return sample

        keys = sample[self.image_key]
        sample, images = load_data_with_context(
            sample, context, keys, load_image, mm_bytes_key=self.image_bytes_key
        )

        brightness_details = [calc_brightness_stats(images[k]) for k in keys]
        sample[Fields.stats][StatsKeys.image_brightness_scores] = [
            detail["brightness_mean"] for detail in brightness_details
        ]
        sample[Fields.stats][StatsKeys.image_brightness_perc_5_scores] = [
            detail["brightness_perc_5"] for detail in brightness_details
        ]
        sample[Fields.stats][StatsKeys.image_brightness_perc_99_scores] = [
            detail["brightness_perc_99"] for detail in brightness_details
        ]
        return sample

    def process_single(self, sample):
        stats = sample[Fields.stats].get(StatsKeys.image_brightness_scores, [])
        perc_5 = sample[Fields.stats].get(StatsKeys.image_brightness_perc_5_scores)
        perc_99 = sample[Fields.stats].get(StatsKeys.image_brightness_perc_99_scores)
        if not stats:
            return True
        if perc_5 is None or perc_99 is None:
            if isinstance(stats[0], dict):
                perc_5 = [detail["brightness_perc_5"] for detail in stats]
                perc_99 = [detail["brightness_perc_99"] for detail in stats]
            else:
                return True

        keep = []
        for p5, p99 in zip(perc_5, perc_99):
            if self.min_brightness is not None and p99 < self.min_brightness:
                keep.append(False)
            elif self.max_brightness is not None and p5 > self.max_brightness:
                keep.append(False)
            else:
                keep.append(True)

        keep = np.array(keep)
        return keep.any() if self.any else keep.all()
