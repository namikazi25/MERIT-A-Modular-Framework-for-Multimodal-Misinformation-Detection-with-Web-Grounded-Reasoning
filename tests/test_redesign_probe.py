import hashlib
import io
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from scripts.redesign.probe_detector import (
    MEAN, STD, ProbeUnavailable, aggregate_logits, load_probe_detector,
    prepare_image, prepare_sample,
)


def png(array):
    out = io.BytesIO()
    Image.fromarray(array).save(out, format='PNG')
    return out.getvalue()


class ProbePreparationTests(unittest.TestCase):
    def test_repeat_padding_preserves_pixels_without_resize(self):
        pixels = np.asarray([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [1, 2, 3]]], dtype=np.uint8)
        result = prepare_image(png(pixels))
        self.assertEqual(result.shape, (1, 3, 336, 336))
        self.assertEqual(result.dtype, np.float32)
        expected = np.tile(pixels, (168, 168, 1)).transpose(2, 0, 1).astype(np.float32)
        np.testing.assert_allclose(result[0], (expected / 255 - MEAN) / STD, rtol=1e-6)

    def test_row_major_grid_discards_only_partial_edges(self):
        pixels = np.zeros((673, 674, 3), dtype=np.uint8)
        for y, x, value in ((0, 0, 20), (0, 336, 40), (336, 0, 60), (336, 336, 80)):
            pixels[y:y + 336, x:x + 336] = value
        pixels[672:, :] = 255
        pixels[:, 672:] = 255
        result = prepare_image(png(pixels))
        self.assertEqual(result.shape, (4, 3, 336, 336))
        for index, value in enumerate((20, 40, 60, 80)):
            expected = (np.full((3, 336, 336), value, dtype=np.float32) / 255 - MEAN) / STD
            np.testing.assert_allclose(result[index], expected, rtol=1e-6)

    def test_sigmoid_after_mean_not_mean_probability_and_strict_threshold(self):
        result = aggregate_logits([-2, 4])
        self.assertAlmostEqual(result['p_ai_generated'], 0.7310585786)
        self.assertTrue(result['ai_generated'])
        self.assertFalse(aggregate_logits([-1, 1])['ai_generated'])
        self.assertEqual(aggregate_logits([-1000])['p_ai_generated'], 0)
        self.assertEqual(aggregate_logits([1000])['p_ai_generated'], 1)

    def test_invalid_logits_never_become_valid_signals(self):
        for values in ([], [1] * 65, [True], ['1'], [{}], [[1]], [float('nan')], [float('inf')]):
            with self.subTest(values=repr(values)[:40]), self.assertRaises(ValueError):
                aggregate_logits(values)

    def test_membership_and_identity_are_checked_before_image_processing(self):
        guard = Mock()
        guard.check.side_effect = ValueError('reserved')
        with patch('scripts.redesign.probe_detector.prepare_image') as prepare:
            with self.assertRaises(ValueError):
                prepare_sample({'sample_id': 'reserved', 'image_bytes': b'not read'}, guard)
            prepare.assert_not_called()
        data = png(np.zeros((2, 2, 3), dtype=np.uint8))
        guard.check.side_effect = None
        guard.check.return_value = {'image_byte_sha256': hashlib.sha256(data).hexdigest()}
        sample = {'sample_id': 'synthetic', 'image_bytes': data, 'label': 'private', 'path': '/fake/private.png'}
        self.assertEqual(prepare_sample(sample, guard).shape, (1, 3, 336, 336))
        sample['image_bytes'] += b'changed'
        with self.assertRaises(ValueError):
            prepare_sample(sample, guard)
        self.assertEqual(guard.check.call_count, 3)

    def test_resource_bounds_and_invalid_images_fail_explicitly(self):
        for data in (b'', b'invalid', '/fake/image.png'):
            with self.assertRaises((ValueError, OSError)):
                prepare_image(data)
        image = png(np.zeros((336, 336, 3), dtype=np.uint8))
        with patch('scripts.redesign.probe_detector.MAX_PIXELS', 100):
            with self.assertRaises(ValueError):
                prepare_image(image)
        with patch('scripts.redesign.probe_detector.MAX_PATCHES', 0):
            with self.assertRaises(ValueError):
                prepare_image(image)

    def test_production_loading_is_blocked_without_fallback(self):
        with self.assertRaisesRegex(ProbeUnavailable, 'licence'):
            load_probe_detector()
