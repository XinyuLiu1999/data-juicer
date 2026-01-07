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


def calc_brightness(image):
    """
    Calculate brightness score using 99th percentile brightness.

    :param image: PIL Image object
    :return: brightness score (0-1, higher = brighter)
    """
    imarr = np.asarray(image)
    if len(imarr.shape) == 3:
        r = imarr[:, :, 0].astype("int")
        g = imarr[:, :, 1].astype("int")
        b = imarr[:, :, 2].astype("int")
        pixel_brightness = calculate_brightness(r, g, b)
    else:
        # Grayscale image
        pixel_brightness = imarr / 255.0

    # Return 99th percentile as brightness score
    return np.percentile(pixel_brightness, 99)


def calc_entropy(image):
    """
    Calculate image entropy (information content).

    Lower values indicate less information (e.g., solid colors).

    :param image: PIL Image object
    :return: entropy score
    """
    return image.entropy()


def is_grayscale(image):
    """
    Check if an image is grayscale.

    :param image: PIL Image object
    :return: True if grayscale, False otherwise
    """
    if image.mode == "L":
        return True
    if image.mode in ["RGB", "RGBA"]:
        imarr = np.asarray(image)
        if len(imarr.shape) == 2:
            return True
        # Check if all channels are equal
        if imarr.shape[2] >= 3:
            return np.allclose(imarr[:, :, 0], imarr[:, :, 1]) and np.allclose(
                imarr[:, :, 1], imarr[:, :, 2]
            )
    return False


@OPERATORS.register_module("image_quality_filter")
@LOADED_IMAGES.register_module("image_quality_filter")
class ImageQualityFilter(Filter):
    """Filter to keep samples with images meeting quality criteria.

    This filter evaluates multiple image quality metrics:
    - Blurriness: Detects blurry images using edge detection variance
    - Brightness: Detects too dark or too bright images
    - Entropy: Detects low-information images (e.g., solid colors)
    - Grayscale: Optionally filters grayscale images

    Each metric can be independently enabled/disabled and configured with
    min/max thresholds. The 'any_or_all' parameter determines the strategy:
    'any' keeps samples if at least one image meets all criteria, while 'all'
    requires all images to meet all criteria.
    """

    _batched_op = True

    def __init__(
        self,
        # Blurriness thresholds (higher = sharper)
        min_blurriness: float = None,
        max_blurriness: float = None,
        # Brightness thresholds (0-1, higher = brighter)
        min_brightness: float = None,
        max_brightness: float = None,
        # Entropy thresholds (higher = more information)
        min_entropy: float = None,
        max_entropy: float = None,
        # Grayscale filter
        keep_grayscale: bool = True,
        # Strategy
        any_or_all: str = "any",
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param min_blurriness: Minimum blurriness score to keep. Images with
            lower scores are considered too blurry. Typical values: 5-20.
            Set to None to disable this check.
        :param max_blurriness: Maximum blurriness score to keep. Set to None
            to disable this check.
        :param min_brightness: Minimum brightness score (0-1) to keep. Images
            with lower scores are considered too dark. Typical value: 0.1.
            Set to None to disable this check.
        :param max_brightness: Maximum brightness score (0-1) to keep. Images
            with higher scores are considered too bright. Typical value: 0.9.
            Set to None to disable this check.
        :param min_entropy: Minimum entropy score to keep. Images with lower
            scores have less information (e.g., solid colors). Typical value: 3.0.
            Set to None to disable this check.
        :param max_entropy: Maximum entropy score to keep. Set to None to
            disable this check.
        :param keep_grayscale: If False, filter out grayscale images.
            Default is True (keep grayscale images).
        :param any_or_all: Keep strategy for multiple images. 'any': keep
            sample if any image meets criteria. 'all': keep sample only if
            all images meet criteria.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_blurriness = min_blurriness
        self.max_blurriness = max_blurriness
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_entropy = min_entropy
        self.max_entropy = max_entropy
        self.keep_grayscale = keep_grayscale

        if any_or_all not in ["any", "all"]:
            raise ValueError(
                f"Keep strategy [{any_or_all}] is not supported. "
                f'Can only be one of ["any", "all"].'
            )
        self.any = any_or_all == "any"

        # Determine which metrics to compute
        self.check_blurriness = (
            min_blurriness is not None or max_blurriness is not None
        )
        self.check_brightness = (
            min_brightness is not None or max_brightness is not None
        )
        self.check_entropy = min_entropy is not None or max_entropy is not None
        self.check_grayscale = not keep_grayscale

    def compute_stats_batched(self, samples, context=False):
        image_list = samples[self.image_key]
        samples_stats = samples[Fields.stats]

        for i, stat in enumerate(samples_stats):
            # Check if already computed
            already_computed = (
                (not self.check_blurriness or StatsKeys.image_blurriness_scores in stat)
                and (not self.check_brightness or StatsKeys.image_brightness_scores in stat)
                and (not self.check_entropy or StatsKeys.image_entropy_scores in stat)
                and (not self.check_grayscale or StatsKeys.image_grayscale_flags in stat)
            )
            if already_computed:
                continue

            # No images in this sample
            loaded_image_keys = image_list[i]
            if not loaded_image_keys:
                if self.check_blurriness:
                    stat[StatsKeys.image_blurriness_scores] = np.array([], dtype=np.float64)
                if self.check_brightness:
                    stat[StatsKeys.image_brightness_scores] = np.array([], dtype=np.float64)
                if self.check_entropy:
                    stat[StatsKeys.image_entropy_scores] = np.array([], dtype=np.float64)
                if self.check_grayscale:
                    stat[StatsKeys.image_grayscale_flags] = np.array([], dtype=bool)
                continue

            # Load images
            samples, images = load_data_with_context(
                samples,
                context,
                loaded_image_keys,
                load_image,
                mm_bytes_key=self.image_bytes_key,
                sample_idx=i,
            )

            # Compute metrics for each image
            if self.check_blurriness:
                blurriness_scores = {
                    key: calc_blurriness(images[key]) for key in images
                }
                stat[StatsKeys.image_blurriness_scores] = [
                    blurriness_scores[key] for key in loaded_image_keys
                ]

            if self.check_brightness:
                brightness_scores = {
                    key: calc_brightness(images[key]) for key in images
                }
                stat[StatsKeys.image_brightness_scores] = [
                    brightness_scores[key] for key in loaded_image_keys
                ]

            if self.check_entropy:
                entropy_scores = {key: calc_entropy(images[key]) for key in images}
                stat[StatsKeys.image_entropy_scores] = [
                    entropy_scores[key] for key in loaded_image_keys
                ]

            if self.check_grayscale:
                grayscale_flags = {key: is_grayscale(images[key]) for key in images}
                stat[StatsKeys.image_grayscale_flags] = [
                    grayscale_flags[key] for key in loaded_image_keys
                ]

        return samples

    def process_batched(self, samples):
        def check_single_image(idx, stat):
            """Check if a single image passes all quality criteria."""
            # Check blurriness
            if self.check_blurriness:
                blurriness = stat[StatsKeys.image_blurriness_scores][idx]
                if self.min_blurriness is not None and blurriness < self.min_blurriness:
                    return False
                if self.max_blurriness is not None and blurriness > self.max_blurriness:
                    return False

            # Check brightness
            if self.check_brightness:
                brightness = stat[StatsKeys.image_brightness_scores][idx]
                if self.min_brightness is not None and brightness < self.min_brightness:
                    return False
                if self.max_brightness is not None and brightness > self.max_brightness:
                    return False

            # Check entropy
            if self.check_entropy:
                entropy = stat[StatsKeys.image_entropy_scores][idx]
                if self.min_entropy is not None and entropy < self.min_entropy:
                    return False
                if self.max_entropy is not None and entropy > self.max_entropy:
                    return False

            # Check grayscale
            if self.check_grayscale:
                is_gray = stat[StatsKeys.image_grayscale_flags][idx]
                if is_gray:
                    return False

            return True

        def process_single(stat):
            # Determine the number of images from any available metric
            num_images = 0
            if self.check_blurriness and StatsKeys.image_blurriness_scores in stat:
                num_images = len(stat[StatsKeys.image_blurriness_scores])
            elif self.check_brightness and StatsKeys.image_brightness_scores in stat:
                num_images = len(stat[StatsKeys.image_brightness_scores])
            elif self.check_entropy and StatsKeys.image_entropy_scores in stat:
                num_images = len(stat[StatsKeys.image_entropy_scores])
            elif self.check_grayscale and StatsKeys.image_grayscale_flags in stat:
                num_images = len(stat[StatsKeys.image_grayscale_flags])

            if num_images == 0:
                return True

            keep_bools = np.array(
                [check_single_image(idx, stat) for idx in range(num_images)]
            )

            if self.any:
                return keep_bools.any()
            else:
                return keep_bools.all()

        return map(lambda stat: process_single(stat), samples[Fields.stats])
