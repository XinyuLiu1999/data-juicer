import math

import numpy as np
from PIL import ImageFilter, ImageStat

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import load_data_with_context, load_image

from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_IMAGES

# Maximum resolution for blurry detection (to avoid memory issues with large images)
MAX_RESOLUTION_FOR_BLURRY_DETECTION = 512


def calculate_brightness(red, green, blue):
    """Calculate perceived brightness using standard formula."""
    return (
        np.sqrt(0.241 * (red * red) + 0.691 * (green * green) + 0.068 * (blue * blue))
    ) / 255


def calc_blurriness(image):
    """
    Calculate blurriness score using Laplacian edge detection variance.

    Lower values indicate more blur.

    :param image: PIL Image object
    :return: blurriness score (higher = sharper)
    """
    # Resize if too large
    ratio = max(image.width, image.height) / MAX_RESOLUTION_FOR_BLURRY_DETECTION
    if ratio > 1:
        resized_image = image.resize(
            (max(int(image.width // ratio), 1), max(int(image.height // ratio), 1))
        )
    else:
        resized_image = image.copy()

    # Convert to grayscale
    gray_image = resized_image.convert("L")

    # Apply edge detection filter
    edges = gray_image.filter(ImageFilter.FIND_EDGES)

    # Calculate variance of edges - higher variance = sharper image
    blurriness = ImageStat.Stat(edges).var[0]
    return np.sqrt(blurriness)

def calc_entropy(image):
    """
    Calculate image entropy (information content).

    Lower values indicate less information (e.g., solid colors).

    :param image: PIL Image object
    :return: entropy score
    """
    return image.entropy()

def calc_brightness_stats(image):
    """
    Compute robust brightness statistics.

    Returns:
        dict with:
          - brightness_perc_5
          - brightness_perc_99
          - brightness_mean
    """
    imarr = np.asarray(image)

    if imarr.ndim == 3:
        r = imarr[:, :, 0].astype("int")
        g = imarr[:, :, 1].astype("int")
        b = imarr[:, :, 2].astype("int")
        pixel_brightness = calculate_brightness(r, g, b)
    else:
        pixel_brightness = imarr / 255.0

    return {
        "brightness_perc_5": float(np.percentile(pixel_brightness, 5)),
        "brightness_perc_99": float(np.percentile(pixel_brightness, 99)),
        "brightness_mean": float(pixel_brightness.mean()),
    }


def is_grayscale(image):
    return image.mode == "L"


@OPERATORS.register_module("image_quality_filter")
class ImageQualityFilter(Filter):
    def __init__(
        self,
        min_blurriness=None,
        max_blurriness=None,
        min_brightness=None,
        max_brightness=None,
        min_entropy=None,
        max_entropy=None,
        keep_grayscale=True,
        any_or_all="any",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if any_or_all not in ["any", "all"]:
            raise ValueError(f'Keep strategy [{any_or_all}] is not supported.')

        self.any = any_or_all == "any"

        self.min_blurriness = min_blurriness
        self.max_blurriness = max_blurriness
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_entropy = min_entropy
        self.max_entropy = max_entropy
        self.keep_grayscale = keep_grayscale

        self.check_blurriness = min_blurriness is not None or max_blurriness is not None
        self.check_brightness = min_brightness is not None or max_brightness is not None
        self.check_entropy = min_entropy is not None or max_entropy is not None
        self.check_grayscale = not keep_grayscale

    def compute_stats_single(self, sample, context=False):
        # already computed
        if StatsKeys.image_quality_scores in sample[Fields.stats]:
            return sample

        # no images
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][StatsKeys.image_quality_scores] = []
            return sample

        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(
            sample,
            context,
            loaded_image_keys,
            load_image,
            mm_bytes_key=self.image_bytes_key,
        )

        quality_stats = []

        for key in loaded_image_keys:
            image = images[key]
            entry = {}

            if self.check_blurriness:
                entry["blurriness"] = calc_blurriness(image)

            if self.check_brightness:
                entry["brightness"] = calc_brightness_stats(image)

            if self.check_entropy:
                entry["entropy"] = calc_entropy(image)

            if self.check_grayscale:
                entry["is_grayscale"] = is_grayscale(image)

            quality_stats.append(entry)

        sample[Fields.stats][StatsKeys.image_quality_scores] = quality_stats
        return sample

    def process_single(self, sample):
        stats = sample[Fields.stats][StatsKeys.image_quality_scores]

        if len(stats) == 0:
            return True

        keep_bools = []

        for entry in stats:
            if self.check_blurriness:
                v = entry["blurriness"]
                if self.min_blurriness is not None and v < self.min_blurriness:
                    keep_bools.append(False)
                    continue
                if self.max_blurriness is not None and v > self.max_blurriness:
                    keep_bools.append(False)
                    continue

            if self.check_brightness:
                b = entry["brightness"]
                if self.min_brightness is not None and b["brightness_perc_99"] < self.min_brightness:
                    keep_bools.append(False)
                    continue
                if self.max_brightness is not None and b["brightness_perc_5"] > self.max_brightness:
                    keep_bools.append(False)
                    continue

            if self.check_entropy:
                e = entry["entropy"]
                if self.min_entropy is not None and e < self.min_entropy:
                    keep_bools.append(False)
                    continue
                if self.max_entropy is not None and e > self.max_entropy:
                    keep_bools.append(False)
                    continue

            if self.check_grayscale and entry.get("is_grayscale", False):
                keep_bools.append(False)
                continue

            keep_bools.append(True)

        keep_bools = np.array(keep_bools)
        return keep_bools.any() if self.any else keep_bools.all()


