import io

import numpy as np
import PIL.Image

from data_juicer.utils.constant import Fields, StatsKeys

from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_IMAGES

OP_NAME = "image_validity_filter"


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageValidityFilter(Filter):
    """Filter to drop samples whose images cannot be decoded (broken / truncated)
    or are animated (e.g. animated GIFs / WebP / APNG).

    For each image this operator attempts a guarded PIL decode using the SAME
    path as the downstream image ops (open + convert('RGB')). An image is kept
    only if it (a) decodes successfully as a single still frame and (b) is not
    animated (``n_frames`` <= 1). Decode failures are caught per image and never
    propagate, so a broken image only drops its own sample instead of failing
    the entire Ray batch. Per-image validity (1/0) is stored in the
    'image_valid' stats field.

    Place this op FIRST in the process chain: it removes broken/animated images
    before later image ops try to load them, eliminating the whole-batch drops
    those ops would otherwise trigger on an undecodable image."""

    _batched_op = True

    def __init__(self, any_or_all: str = "all", *args, **kwargs):
        """
        Initialization method.

        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all images. 'any': keep this sample if any image is valid.
            'all': keep this sample only if all images are valid (default;
            for single-image samples 'any' and 'all' are equivalent).
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        if any_or_all not in ["any", "all"]:
            raise ValueError(f"Keep strategy [{any_or_all}] is not supported. " f'Can only be one of ["any", "all"].')
        self.any = any_or_all == "any"

    @staticmethod
    def _is_valid(raw):
        """Return True iff `raw` (bytes or path) decodes as a single still frame."""
        try:
            if isinstance(raw, (bytes, bytearray, memoryview)):
                img = PIL.Image.open(io.BytesIO(bytes(raw)))
            else:
                img = PIL.Image.open(raw)
            # animated images (GIF/WebP/APNG) expose n_frames > 1
            if getattr(img, "n_frames", 1) > 1 or getattr(img, "is_animated", False):
                return False
            # force a full decode (mirrors downstream load_image) to catch
            # truncated/corrupt payloads that only fail past the header
            img.convert("RGB")
            return True
        except Exception:
            return False

    def compute_stats_single(self, sample, context=False):
        # check if it's computed already
        if StatsKeys.image_valid in sample[Fields.stats]:
            return sample

        # there is no image in this sample
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][StatsKeys.image_valid] = np.array([], dtype=np.int64)
            return sample

        loaded_image_keys = sample[self.image_key]
        bytes_key = getattr(self, "image_bytes_key", None)
        image_bytes = sample.get(bytes_key) if bytes_key else None

        valids = []
        for idx, key in enumerate(loaded_image_keys):
            raw = None
            if image_bytes is not None and idx < len(image_bytes):
                raw = image_bytes[idx]
            if raw is None or (hasattr(raw, "__len__") and len(raw) == 0):
                raw = key  # fall back to path / url
            valids.append(1 if self._is_valid(raw) else 0)

        sample[Fields.stats][StatsKeys.image_valid] = valids
        return sample

    def process_single(self, sample):
        valids = sample[Fields.stats][StatsKeys.image_valid]
        if len(valids) <= 0:
            return True
        keep_bools = np.array([v > 0 for v in valids])
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
