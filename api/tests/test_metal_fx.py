"""Tests for api/metal_fx/scaler.py — MetalFX spatial upscaling."""
import sys
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from metal_fx import scaler as mfx
from metal_fx.scaler import (
    is_metalfx_supported, MetalFXScaler, create_scaler,
)


class IsMetalFxSupportedTests(unittest.TestCase):
    def test_false_on_non_darwin(self) -> None:
        with patch.object(sys, 'platform', 'linux'):
            self.assertFalse(is_metalfx_supported())

    def test_false_on_windows(self) -> None:
        with patch.object(sys, 'platform', 'win32'):
            self.assertFalse(is_metalfx_supported())

    def test_false_on_darwin_old_macos(self) -> None:
        with patch.object(sys, 'platform', 'darwin'), \
             patch('metal_fx.scaler.platform.mac_ver', return_value=("12.5", "", "")), \
             patch('metal_fx.scaler.ctypes.util.find_library', return_value="/lib/MetalFX.dylib"):
            self.assertFalse(is_metalfx_supported())

    def test_false_on_darwin_when_library_missing(self) -> None:
        with patch.object(sys, 'platform', 'darwin'), \
             patch('metal_fx.scaler.platform.mac_ver', return_value=("13.0", "", "")), \
             patch('metal_fx.scaler.ctypes.util.find_library', return_value=None):
            self.assertFalse(is_metalfx_supported())

    def test_true_on_darwin_13_with_library(self) -> None:
        with patch.object(sys, 'platform', 'darwin'), \
             patch('metal_fx.scaler.platform.mac_ver', return_value=("13.0", "", "")), \
             patch('metal_fx.scaler.ctypes.util.find_library', return_value="/lib/MetalFX.dylib"):
            self.assertTrue(is_metalfx_supported())


class MetalFXScalerInitTests(unittest.TestCase):
    def test_raises_runtime_error_on_unsupported_platform(self) -> None:
        with patch('metal_fx.scaler.is_metalfx_supported', return_value=False):
            with self.assertRaises(RuntimeError):
                MetalFXScaler(1280, 720, 2560, 1440)

    def test_init_stores_dimensions_on_supported(self) -> None:
        with patch('metal_fx.scaler.is_metalfx_supported', return_value=True):
            s = MetalFXScaler(1280, 720, 2560, 1440)
        self.assertEqual(s.input_width, 1280)
        self.assertEqual(s.input_height, 720)
        self.assertEqual(s.output_width, 2560)
        self.assertEqual(s.output_height, 1440)
        self.assertTrue(s._initialized)

    def test_init_default_color_format(self) -> None:
        with patch('metal_fx.scaler.is_metalfx_supported', return_value=True):
            s = MetalFXScaler(100, 100, 200, 200)
        self.assertEqual(s.color_format, "rgba8")

    def test_init_custom_color_format(self) -> None:
        with patch('metal_fx.scaler.is_metalfx_supported', return_value=True):
            s = MetalFXScaler(100, 100, 200, 200, color_format="bgra8")
        self.assertEqual(s.color_format, "bgra8")


class UpscaleValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._patch = patch('metal_fx.scaler.is_metalfx_supported', return_value=True)
        self._patch.start()
        self.scaler = MetalFXScaler(100, 100, 200, 200)

    def tearDown(self) -> None:
        self._patch.stop()

    def test_raises_runtime_error_when_not_initialized(self) -> None:
        self.scaler._initialized = False
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        with self.assertRaises(RuntimeError):
            self.scaler.upscale(img)

    def test_raises_value_error_for_wrong_dimensions(self) -> None:
        img = np.zeros((50, 50, 4), dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.scaler.upscale(img)

    def test_raises_value_error_for_wrong_width(self) -> None:
        img = np.zeros((100, 50, 4), dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.scaler.upscale(img)


class UpscaleOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self._patch = patch('metal_fx.scaler.is_metalfx_supported', return_value=True)
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()

    def test_upscale_produces_correct_shape_2x(self) -> None:
        s = MetalFXScaler(100, 100, 200, 200)
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        out = s.upscale(img)
        self.assertEqual(out.shape, (200, 200, 4))
        self.assertEqual(out.dtype, np.uint8)

    def test_upscale_produces_correct_shape_1_5x(self) -> None:
        s = MetalFXScaler(100, 100, 150, 150)
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        out = s.upscale(img)
        self.assertEqual(out.shape, (150, 150, 4))

    def test_upscale_produces_correct_shape_4x(self) -> None:
        s = MetalFXScaler(50, 50, 200, 200)
        img = np.zeros((50, 50, 4), dtype=np.uint8)
        out = s.upscale(img)
        self.assertEqual(out.shape, (200, 200, 4))

    def test_upscale_solid_color_stays_solid(self) -> None:
        s = MetalFXScaler(50, 50, 100, 100)
        img = np.full((50, 50, 4), 200, dtype=np.uint8)
        out = s.upscale(img)
        # A solid color should remain (approximately) solid after Lanczos.
        self.assertTrue(np.all(out >= 190))
        self.assertTrue(np.all(out <= 210))

    def test_upscale_converts_float_input_to_uint8(self) -> None:
        s = MetalFXScaler(50, 50, 100, 100)
        img = np.ones((50, 50, 4), dtype=np.float32)  # 1.0 -> 255
        out = s.upscale(img)
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue(np.all(out >= 240))


class LanczosUpscaleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._patch = patch('metal_fx.scaler.is_metalfx_supported', return_value=True)
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()

    def test_lanczos_produces_correct_output_dimensions(self) -> None:
        s = MetalFXScaler(40, 30, 80, 60)
        img = np.zeros((30, 40, 4), dtype=np.uint8)
        out = s._lanczos_upscale(img)
        self.assertEqual(out.shape, (60, 80, 4))

    def test_lanczos_preserves_rgba_channels(self) -> None:
        s = MetalFXScaler(20, 20, 40, 40)
        img = np.zeros((20, 20, 4), dtype=np.uint8)
        img[..., 0] = 255  # red channel
        out = s._lanczos_upscale(img)
        self.assertEqual(out.shape[2], 4)
        # Red channel should dominate
        self.assertTrue(np.all(out[..., 0] >= out[..., 1]))


class CloseTests(unittest.TestCase):
    def test_close_sets_initialized_false(self) -> None:
        with patch('metal_fx.scaler.is_metalfx_supported', return_value=True):
            s = MetalFXScaler(100, 100, 200, 200)
        self.assertTrue(s._initialized)
        s.close()
        self.assertFalse(s._initialized)

    def test_upscale_after_close_raises(self) -> None:
        with patch('metal_fx.scaler.is_metalfx_supported', return_value=True):
            s = MetalFXScaler(100, 100, 200, 200)
        s.close()
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        with self.assertRaises(RuntimeError):
            s.upscale(img)


class CreateScalerTests(unittest.TestCase):
    def test_returns_scaler_on_supported(self) -> None:
        with patch('metal_fx.scaler.is_metalfx_supported', return_value=True):
            s = create_scaler(100, 100, 200, 200)
        self.assertIsNotNone(s)
        self.assertIsInstance(s, MetalFXScaler)

    def test_returns_none_on_unsupported(self) -> None:
        with patch('metal_fx.scaler.is_metalfx_supported', return_value=False):
            s = create_scaler(100, 100, 200, 200)
        self.assertIsNone(s)


if __name__ == "__main__":
    unittest.main()
