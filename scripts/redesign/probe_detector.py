"""PROBE-DINOv2 input preparation, not a live checkpoint integration.

Implements the documented numeric input/aggregation contract independently.
Inference is blocked pending checkpoint licence and exact backbone configuration.
No downloaded code, network access, path-derived labels or fallback detector.
"""
import hashlib
import io
import math
from numbers import Real

import numpy as np
from PIL import Image

PATCH_SIZE = 336
MAX_PIXELS = 16_000_000
MAX_PATCHES = 64
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


class ProbeUnavailable(RuntimeError):
    pass


def load_probe_detector(*args, **kwargs):
    """Fail explicitly instead of substituting the historical ViT or a VLM."""
    raise ProbeUnavailable(
        'PROBE inference is not implemented: checkpoint licence and exact pinned '
        'backbone configuration must be established before loading weights.')


def prepare_image(image_bytes):
    """Return normalized NCHW float32 patches for a bounded RGB image.

    Repeat undersized dimensions, then take a row-major, nonoverlapping 336 grid.
    Right/bottom remainders are discarded. No resizing, EXIF orientation change,
    stochastic augmentation, path input or arbitrary metadata enters the array.
    Oversized/invalid inputs fail; they do not receive a neutral detector score.
    Real development samples must enter through prepare_sample below.
    """
    if type(image_bytes) is not bytes or not image_bytes:
        raise ValueError('Nonempty image bytes required')
    with Image.open(io.BytesIO(image_bytes)) as opened:
        width, height = opened.size
        if width * height > MAX_PIXELS:
            raise ValueError('PROBE image exceeds the local pixel bound')
        tiled_width = width * max(1, math.ceil(PATCH_SIZE / width))
        tiled_height = height * max(1, math.ceil(PATCH_SIZE / height))
        columns, rows = tiled_width // PATCH_SIZE, tiled_height // PATCH_SIZE
        if not 1 <= columns * rows <= MAX_PATCHES:
            raise ValueError('PROBE patch count exceeds the local bound')
        pixels = np.asarray(opened.convert('RGB'))
    patches = []
    # Modular indices implement repetition without allocating a huge tiled image.
    for row in range(rows):
        ys = (row * PATCH_SIZE + np.arange(PATCH_SIZE)) % height
        for column in range(columns):
            xs = (column * PATCH_SIZE + np.arange(PATCH_SIZE)) % width
            patch = pixels[ys[:, None], xs[None, :], :].transpose(2, 0, 1)
            patches.append((patch.astype(np.float32) / np.float32(255) - MEAN) / STD)
    return np.stack(patches)


def prepare_sample(sample, admission):
    """Revalidate membership and image identity on every invocation."""
    record = admission.check(sample['sample_id'])
    data = sample['image_bytes']
    if type(data) is not bytes or hashlib.sha256(data).hexdigest() != record['image_byte_sha256']:
        raise ValueError('PROBE image identity mismatch')
    return prepare_image(data)


def aggregate_logits(logits):
    """Sigmoid of mean patch logits; output is a generation signal only.

    This function accepts computed logits, not benchmark annotations. Probability
    is uncalibrated. It does not establish manipulation, provenance or truth.
    """
    if type(logits) not in (list, tuple) or not 1 <= len(logits) <= MAX_PATCHES:
        raise ValueError('A bounded nonempty sequence of patch logits is required')
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) or not math.isfinite(v) for v in logits):
        raise ValueError('Patch logits must be finite scalar numbers')
    # Divide first to avoid overflowing a finite-input sum.
    mean = math.fsum(float(v) / len(logits) for v in logits)
    p = 1 / (1 + math.exp(-mean)) if mean >= 0 else math.exp(mean) / (1 + math.exp(mean))
    return {'p_ai_generated': p, 'ai_generated': p > 0.5, 'patch_count': len(logits)}
