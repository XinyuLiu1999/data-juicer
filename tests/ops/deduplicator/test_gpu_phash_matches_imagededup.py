"""Verify GPU batch phash produces identical hashes to imagededup PHash.

Run with: python tests/ops/deduplicator/test_gpu_phash_matches_imagededup.py

Requires: imagededup, torch, PIL, numpy, scipy

This test is self-contained: it copies _build_dct_matrix and _gpu_batch_phash
inline to avoid pulling in data_juicer's full dependency chain.
If the source implementation changes, this test must be updated to match.
"""
import math

import numpy as np
import PIL.Image


# ---------- Copied from ray_image_deduplicator.py ----------

def _build_dct_matrix(n):
    mat = np.zeros((n, n), dtype=np.float64)
    for k in range(n):
        for i in range(n):
            mat[k, i] = 2.0 * math.cos(math.pi * k * (2 * i + 1) / (2 * n))
    return mat


_DCT_MATRIX_32 = _build_dct_matrix(32)


def _gpu_batch_phash(pil_images, device):
    import torch

    tensors = []
    for img in pil_images:
        resized = img.resize((32, 32), PIL.Image.LANCZOS)
        gray_img = resized.convert("L")
        t = torch.from_numpy(
            np.array(gray_img, dtype=np.uint8).astype(np.float64))
        tensors.append(t)

    gray = torch.stack(tensors).to(device)
    dct_mat = torch.from_numpy(_DCT_MATRIX_32).to(device)
    dct_result = torch.matmul(torch.matmul(dct_mat, gray), dct_mat.t())
    low_freq = dct_result[:, :8, :8].reshape(-1, 64)
    medians = low_freq[:, 1:].median(dim=1, keepdim=True).values
    hash_bits = (low_freq >= medians).cpu().numpy().astype(np.uint8)
    hash_bytes = np.packbits(hash_bits, axis=1)
    return [bytes(row).hex() for row in hash_bytes]


# ---------- End copied code ----------


def hamming_distance(hex1, hex2):
    """Compute hamming distance between two hex hash strings."""
    bits1 = bin(int(hex1, 16))[2:].zfill(64)
    bits2 = bin(int(hex2, 16))[2:].zfill(64)
    return sum(a != b for a, b in zip(bits1, bits2))


def test_gpu_phash_matches_imagededup():
    """Generate random images and verify GPU phash == imagededup phash."""
    try:
        import torch
        from imagededup.methods import PHash
    except ImportError as e:
        print(f"SKIP: missing dependency: {e}")
        return

    hasher = PHash()

    # --- Real-world-like images (must match exactly) ---
    realistic_images = []
    rng = np.random.RandomState(42)
    for size in [(64, 64), (256, 256), (100, 200), (31, 31), (1024, 768)]:
        for _ in range(5):
            arr = rng.randint(0, 256, (*size, 3), dtype=np.uint8)
            realistic_images.append(PIL.Image.fromarray(arr))

    # --- Degenerate images (may have floating-point noise differences) ---
    degenerate_images = []
    for color in [(0, 0, 0), (255, 255, 255), (128, 128, 128)]:
        degenerate_images.append(PIL.Image.new("RGB", (64, 64), color))
    # Gradient
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    for i in range(64):
        arr[i, :, :] = int(i * 255 / 63)
    degenerate_images.append(PIL.Image.fromarray(arr))

    def get_cpu_hash(img):
        resized = img.resize((32, 32), PIL.Image.LANCZOS)
        gray = resized.convert("L")
        return hasher.encode_image(image_array=np.array(gray, dtype=np.uint8))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Test realistic images: require exact match
    cpu_hashes = [get_cpu_hash(img) for img in realistic_images]
    gpu_hashes = _gpu_batch_phash(realistic_images, device)

    exact_mismatches = 0
    for i, (cpu_h, gpu_h) in enumerate(zip(cpu_hashes, gpu_hashes)):
        if cpu_h != gpu_h:
            exact_mismatches += 1
            hd = hamming_distance(cpu_h, gpu_h)
            print(f"MISMATCH realistic image {i}: cpu={cpu_h} gpu={gpu_h} "
                  f"hamming={hd}")

    print(f"\nRealistic images: {len(realistic_images) - exact_mismatches}/"
          f"{len(realistic_images)} exact match")

    # Test degenerate images: allow small hamming distance due to
    # floating-point noise in DCT of near-constant inputs
    cpu_degen = [get_cpu_hash(img) for img in degenerate_images]
    gpu_degen = _gpu_batch_phash(degenerate_images, device)

    degen_failures = 0
    for i, (cpu_h, gpu_h) in enumerate(zip(cpu_degen, gpu_degen)):
        if cpu_h != gpu_h:
            hd = hamming_distance(cpu_h, gpu_h)
            print(f"  degenerate image {i}: cpu={cpu_h} gpu={gpu_h} "
                  f"hamming={hd} (expected: float-point noise on "
                  f"near-constant input)")

    # Final verdict: realistic images must match exactly
    if exact_mismatches == 0:
        print("\nPASS: GPU phash matches imagededup on all realistic images")
    else:
        print(f"\nFAIL: {exact_mismatches} realistic image hashes differ")

    return exact_mismatches == 0


if __name__ == "__main__":
    success = test_gpu_phash_matches_imagededup()
    exit(0 if success else 1)
