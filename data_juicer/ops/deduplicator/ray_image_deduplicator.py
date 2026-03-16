import hashlib

import numpy as np

from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import (load_data_with_context, load_image,
                                        load_mm_bytes_from_sample)

from ..base_op import OPERATORS
from .ray_basic_deduplicator import RayBasicDeduplicator

imgdedup_methods = LazyLoader("imagededup.methods")

OP_NAME = "ray_image_deduplicator"

HASH_METHOD = {"phash", "dhash", "whash", "ahash", "md5"}


def get_hash_method(method_name):
    mapping = {
        "phash": imgdedup_methods.PHash,
        "dhash": imgdedup_methods.DHash,
        "whash": imgdedup_methods.WHash,
        "ahash": imgdedup_methods.AHash,
    }

    return mapping[method_name]


@OPERATORS.register_module(OP_NAME)
class RayImageDeduplicator(RayBasicDeduplicator):
    """Deduplicates samples at the document level using exact matching of images in Ray distributed mode.

    This operator uses a specified hash method to compute image hashes and identifies
    duplicates by comparing these hashes. It operates in Ray distributed mode, supporting
    'ray_actor' or 'redis' backends for deduplication. The hash method can be set during
    initialization, with supported methods listed in `HASH_METHOD`. If a sample does not
    contain an image, it is assigned an empty hash value. The operator loads images from the
    specified keys and computes their combined hash for comparison.

    The 'md5' method hashes raw image bytes directly without decoding,
    making it ~50x faster than perceptual methods but only catching
    byte-identical duplicates."""

    def __init__(
        self,
        backend: str = "ray_actor",
        redis_address: str = "redis://localhost:6379",
        method: str = "phash",
        *args,
        **kwargs,
    ):
        """
        Initialization.
        :param backend: the backend for dedup, either 'ray_actor' or 'redis'
        :param redis_address: the address of redis server
        :param method: the hash method to use. 'md5' for fast exact-match
            dedup on raw bytes; 'phash'/'dhash'/'whash'/'ahash' for
            perceptual dedup.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(backend=backend, redis_address=redis_address, *args, **kwargs)
        if method not in HASH_METHOD:
            raise ValueError(f"Keep strategy [{method}] is not supported. " f"Can only be one of {HASH_METHOD}.")
        self.method = method
        if method != "md5":
            self.hasher = get_hash_method(method)()

    def calculate_hash(self, sample, context=False):
        if self.image_key not in sample or not sample[self.image_key]:
            return RayBasicDeduplicator.EMPTY_HASH_VALUE

        if self.method == "md5":
            return self._calculate_md5(sample)

        # load images
        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(
            sample, context, loaded_image_keys, load_image, mm_bytes_key=self.image_bytes_key
        )

        # compute hash
        hash_value = ""
        for key in images:
            hash_value += self.hasher.encode_image(image_array=np.array(images[key]))

        return hash_value

    def _calculate_md5(self, sample):
        """Hash raw image bytes with md5. Skips image decoding entirely."""
        hash_value = ""
        loaded_image_keys = sample[self.image_key]
        for idx in range(len(loaded_image_keys)):
            bytes_data = load_mm_bytes_from_sample(
                sample, idx, self.image_bytes_key
            )
            if bytes_data is not None:
                hash_value += hashlib.md5(bytes_data).hexdigest()
            else:
                # fall back to reading from file path
                with open(loaded_image_keys[idx], "rb") as f:
                    hash_value += hashlib.md5(f.read()).hexdigest()
        return hash_value
