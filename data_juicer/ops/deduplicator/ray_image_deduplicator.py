import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from data_juicer.utils.constant import HashKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import load_data_with_context, load_image

from ..base_op import OPERATORS
from .ray_basic_deduplicator import MERSENNE_PRIME, ActorBackend, RayBasicDeduplicator

imgdedup_methods = LazyLoader("imagededup.methods")
ray = LazyLoader("ray")
torch = LazyLoader("torch")

OP_NAME = "ray_image_deduplicator"

HASH_METHOD = {"phash", "dhash", "whash", "ahash"}


def get_hash_method(method_name):
    mapping = {
        "phash": imgdedup_methods.PHash,
        "dhash": imgdedup_methods.DHash,
        "whash": imgdedup_methods.WHash,
        "ahash": imgdedup_methods.AHash,
    }

    return mapping[method_name]


def _build_dct_matrix(n):
    """Precompute the DCT-II basis matrix of size n x n.

    Matches scipy.fftpack.dct type 2 (no normalization):
        DCT[k] = 2 * sum_{i} x[i] * cos(pi * k * (2*i + 1) / (2*N))
    """
    mat = np.zeros((n, n), dtype=np.float32)
    for k in range(n):
        for i in range(n):
            mat[k, i] = 2.0 * math.cos(math.pi * k * (2 * i + 1) / (2 * n))
    return mat


# Precompute once at module level (32x32 for phash with highfreq_factor=4)
_DCT_MATRIX_32 = _build_dct_matrix(32)


def _gpu_batch_phash(pil_images, device):
    """Compute phash for a batch of PIL images on GPU.

    Reproduces the imagededup PHash algorithm:
    1. Resize to 32x32
    2. Convert to grayscale
    3. Apply 2D DCT
    4. Take top-left 8x8 low-frequency coefficients
    5. Hash bits = (value > median)
    6. Pack into hex string

    Args:
        pil_images: list of PIL.Image objects (RGB)
        device: torch device string

    Returns:
        list of hex hash strings
    """
    import torch

    import PIL.Image

    # CPU: convert to grayscale and resize to 32x32.
    # Use PIL operations to match imagededup's load_image behavior.
    tensors = []
    for img in pil_images:
        gray_img = img.convert("L").resize(
            (32, 32), PIL.Image.BILINEAR)
        t = torch.from_numpy(
            np.array(gray_img, dtype=np.float32))  # (32, 32)
        tensors.append(t)

    gray = torch.stack(tensors).to(device)  # (B, 32, 32)

    # 2D DCT via matrix multiplication: DCT = M @ X @ M^T
    # Uses scipy-compatible DCT-II (no ortho normalization)
    dct_mat = torch.from_numpy(_DCT_MATRIX_32).to(device)
    dct_result = torch.matmul(torch.matmul(dct_mat, gray), dct_mat.t())

    # Take top-left 8x8 low-frequency block
    low_freq = dct_result[:, :8, :8].reshape(-1, 64)  # (B, 64)

    # Median threshold per image, excluding DC term (index 0)
    # to match imagededup: np.median(flatten(dct_reduced_coef)[1:])
    medians = low_freq[:, 1:].median(dim=1, keepdim=True).values

    # Use >= to match imagededup: hash_mat = dct_reduced_coef >= median
    hash_bits = (low_freq >= medians).cpu().numpy().astype(np.uint8)  # (B, 64)

    # Pack 64 bits into 8 bytes → 16-char hex string
    hash_bytes = np.packbits(hash_bits, axis=1)  # (B, 8)
    return [bytes(row).hex() for row in hash_bytes]


@OPERATORS.register_module(OP_NAME)
class RayImageDeduplicator(RayBasicDeduplicator):
    """Deduplicates samples at the document level using exact matching of images in Ray distributed mode.

    This operator uses a specified hash method to compute image hashes and identifies
    duplicates by comparing these hashes. It operates in Ray distributed mode, supporting
    'ray_actor' or 'redis' backends for deduplication. The hash method can be set during
    initialization, with supported methods listed in `HASH_METHOD`. If a sample does not
    contain an image, it is assigned an empty hash value. The operator loads images from the
    specified keys and computes their combined hash for comparison.

    When accelerator='cuda' and method='phash', uses GPU-accelerated batch hashing
    for significantly higher throughput on large datasets."""

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
        :param method: the hash method to use
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(backend=backend, redis_address=redis_address, *args, **kwargs)
        if method not in HASH_METHOD:
            raise ValueError(f"Keep strategy [{method}] is not supported. " f"Can only be one of {HASH_METHOD}.")
        self.method = method
        self.hasher = get_hash_method(method)()

    def calculate_hash(self, sample, context=False):
        if self.image_key not in sample or not sample[self.image_key]:
            return RayBasicDeduplicator.EMPTY_HASH_VALUE

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

    def compute_stats_batched(self, samples, rank=None, context=False):
        """Compute hashes and check uniqueness for a batch of samples.

        When running on GPU with method='phash', uses batched GPU hashing
        for much higher throughput. Falls back to CPU otherwise.
        """
        keys = list(samples.keys())
        num_samples = len(samples[keys[0]])

        if self.use_cuda() and self.method == "phash":
            hash_values = self._compute_hashes_gpu(
                samples, keys, num_samples, rank, context)
        else:
            hash_values = self._compute_hashes_cpu(
                samples, keys, num_samples, context)

        # Batch uniqueness checks via Ray actors
        if isinstance(self.backend, ActorBackend):
            self.backend._ensure_actors()
            futures = []
            for hash_val in hash_values:
                dedup_set_id = (
                    int.from_bytes(
                        hash_val.encode(), byteorder="little"
                    )
                    % MERSENNE_PRIME
                    % self.backend.dedup_set_num
                )
                futures.append(
                    self.backend._dedup_sets[dedup_set_id]
                    .is_unique.remote(hash_val)
                )
            results = ray.get(futures)
        else:
            results = [self.backend.is_unique(hv) for hv in hash_values]

        samples[HashKeys.is_unique] = results
        return samples

    def _compute_hashes_gpu(self, samples, keys, num_samples, rank, context):
        """Collect all images, batch-hash on GPU, scatter results back."""
        import torch as th

        device_count = th.cuda.device_count()
        device = f"cuda:{rank % device_count}" if rank is not None else "cuda:0"

        # Phase 1: Collect all images from all samples (CPU, threaded)
        all_images = []
        image_counts = []  # number of images per sample, 0 = no images

        def _load_sample_images(i):
            this_sample = {key: samples[key][i] for key in keys}
            if self.image_key not in this_sample or not this_sample[self.image_key]:
                return []
            loaded_image_keys = this_sample[self.image_key]
            _, images = load_data_with_context(
                this_sample, context, loaded_image_keys, load_image,
                mm_bytes_key=self.image_bytes_key
            )
            return list(images.values())

        num_workers = min(num_samples, 4)
        if num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                per_sample_images = list(executor.map(
                    _load_sample_images, range(num_samples)))
        else:
            per_sample_images = [_load_sample_images(i)
                                 for i in range(num_samples)]

        for img_list in per_sample_images:
            image_counts.append(len(img_list))
            all_images.extend(img_list)

        # Phase 2: Batch GPU phash
        if all_images:
            all_hashes = _gpu_batch_phash(all_images, device)
        else:
            all_hashes = []

        # Phase 3: Scatter — combine per-image hashes into per-sample hash
        hash_values = []
        offset = 0
        for count in image_counts:
            if count == 0:
                hash_values.append(RayBasicDeduplicator.EMPTY_HASH_VALUE)
            else:
                combined = "".join(all_hashes[offset:offset + count])
                hash_values.append(combined)
                offset += count

        return hash_values

    def _compute_hashes_cpu(self, samples, keys, num_samples, context):
        """CPU path with ThreadPoolExecutor parallelism."""
        def _compute_hash(i):
            this_sample = {key: samples[key][i] for key in keys}
            return self.calculate_hash(this_sample, context)

        num_workers = min(num_samples, 4)
        if num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                return list(executor.map(_compute_hash, range(num_samples)))
        else:
            return [_compute_hash(i) for i in range(num_samples)]
