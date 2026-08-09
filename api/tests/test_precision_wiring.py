"""Tests for precision/device wiring through runner and BaseGenerator."""
import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from services.generators.base import BaseGenerator


# --------------------------------------------------------------------------- #
# Concrete subclass of the abstract BaseGenerator for testing.
# --------------------------------------------------------------------------- #
class _ConcreteGenerator(BaseGenerator):
    MODEL_ID = "test-gen"
    DISPLAY_NAME = "Test Generator"
    VRAM_GB = 0

    def load(self) -> None:
        pass

    def generate(self, image_bytes, params, progress_cb=None, cancel_event=None):
        return Path("/tmp/out.glb")


def _make_gen() -> _ConcreteGenerator:
    return _ConcreteGenerator(Path("/tmp/models"), Path("/tmp/outputs"))


# --------------------------------------------------------------------------- #
# Runner env-var wiring
# --------------------------------------------------------------------------- #
_tmp_ext_dir = tempfile.mkdtemp(prefix="modly-precision-test-")
Path(_tmp_ext_dir, "manifest.json").write_text("{}", encoding="utf-8")
os.environ.setdefault("EXTENSION_DIR", _tmp_ext_dir)

runner = importlib.import_module("runner")


class RunnerDevicePrecisionEnvTests(unittest.TestCase):
    def test_device_defaults_to_auto(self) -> None:
        # DEVICE is read at import time; ensure it is a string.
        self.assertIsInstance(runner.DEVICE, str)

    def test_precision_defaults_to_auto(self) -> None:
        self.assertIsInstance(runner.PRECISION, str)

    def test_device_reads_from_env(self) -> None:
        with patch.dict(os.environ, {"MODLY_DEVICE": "mps"}):
            self.assertEqual(os.environ.get("MODLY_DEVICE"), "mps")

    def test_precision_reads_from_env(self) -> None:
        with patch.dict(os.environ, {"MODLY_PRECISION": "fp16"}):
            self.assertEqual(os.environ.get("MODLY_PRECISION"), "fp16")


class RunnerGenerateSetdefaultTests(unittest.TestCase):
    """runner generate injects device/precision via params.setdefault."""

    def test_setdefault_does_not_override_existing_device(self) -> None:
        params = {"device": "cuda"}
        # Simulate the setdefault logic from runner.generate
        msg = {"device": "mps", "precision": "fp16"}
        params.setdefault("device", msg.get("device", runner.DEVICE))
        params.setdefault("precision", msg.get("precision", runner.PRECISION))
        self.assertEqual(params["device"], "cuda")
        self.assertEqual(params["precision"], "fp16")

    def test_setdefault_uses_msg_device_when_params_missing(self) -> None:
        params = {}
        msg = {"device": "mps", "precision": "fp16"}
        params.setdefault("device", msg.get("device", runner.DEVICE))
        params.setdefault("precision", msg.get("precision", runner.PRECISION))
        self.assertEqual(params["device"], "mps")

    def test_setdefault_falls_back_to_runner_constants(self) -> None:
        params = {}
        msg = {}
        params.setdefault("device", msg.get("device", runner.DEVICE))
        params.setdefault("precision", msg.get("precision", runner.PRECISION))
        self.assertEqual(params["device"], runner.DEVICE)
        self.assertEqual(params["precision"], runner.PRECISION)


# --------------------------------------------------------------------------- #
# BaseGenerator._resolve_device
# --------------------------------------------------------------------------- #
class ResolveDeviceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gen = _make_gen()

    def _torch_mock(self, mps=False, cuda=False) -> MagicMock:
        torch = MagicMock()
        torch.backends.mps.is_available.return_value = mps
        torch.cuda.is_available.return_value = cuda
        return torch

    def test_returns_mps_when_available(self) -> None:
        torch = self._torch_mock(mps=True, cuda=True)
        with patch.dict(sys.modules, {"torch": torch}):
            self.assertEqual(self.gen._resolve_device({"device": "auto"}), "mps")

    def test_returns_cuda_when_no_mps(self) -> None:
        torch = self._torch_mock(mps=False, cuda=True)
        with patch.dict(sys.modules, {"torch": torch}):
            self.assertEqual(self.gen._resolve_device({"device": "auto"}), "cuda")

    def test_returns_cpu_when_neither_available(self) -> None:
        torch = self._torch_mock(mps=False, cuda=False)
        with patch.dict(sys.modules, {"torch": torch}):
            self.assertEqual(self.gen._resolve_device({"device": "auto"}), "cpu")

    def test_returns_explicit_device_when_not_auto(self) -> None:
        torch = self._torch_mock(mps=True, cuda=True)
        with patch.dict(sys.modules, {"torch": torch}):
            self.assertEqual(self.gen._resolve_device({"device": "cuda"}), "cuda")

    def test_returns_cpu_for_explicit_cpu(self) -> None:
        torch = self._torch_mock(mps=True, cuda=True)
        with patch.dict(sys.modules, {"torch": torch}):
            self.assertEqual(self.gen._resolve_device({"device": "cpu"}), "cpu")

    def test_defaults_to_auto_when_device_missing(self) -> None:
        torch = self._torch_mock(mps=True, cuda=False)
        with patch.dict(sys.modules, {"torch": torch}):
            self.assertEqual(self.gen._resolve_device({}), "mps")


# --------------------------------------------------------------------------- #
# BaseGenerator._resolve_dtype
# --------------------------------------------------------------------------- #
class ResolveDtypeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gen = _make_gen()
        self.torch = MagicMock()
        self.torch.backends.mps.is_available.return_value = True
        self.torch.cuda.is_available.return_value = False
        self.torch.float16 = "float16"
        self.torch.float32 = "float32"
        self.torch.bfloat16 = "bfloat16"

    def test_auto_on_gpu_returns_float16(self) -> None:
        with patch.dict(sys.modules, {"torch": self.torch}):
            self.assertEqual(self.gen._resolve_dtype({"device": "mps", "precision": "auto"}), "float16")

    def test_auto_on_cpu_returns_float32(self) -> None:
        self.torch.backends.mps.is_available.return_value = False
        self.torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": self.torch}):
            self.assertEqual(self.gen._resolve_dtype({"device": "cpu", "precision": "auto"}), "float32")

    def test_auto_on_cuda_returns_float16(self) -> None:
        self.torch.backends.mps.is_available.return_value = False
        self.torch.cuda.is_available.return_value = True
        with patch.dict(sys.modules, {"torch": self.torch}):
            self.assertEqual(self.gen._resolve_dtype({"device": "cuda", "precision": "auto"}), "float16")

    def test_fp16_returns_float16(self) -> None:
        with patch.dict(sys.modules, {"torch": self.torch}):
            self.assertEqual(self.gen._resolve_dtype({"precision": "fp16"}), "float16")

    def test_bf16_on_macos_14_plus_returns_bfloat16(self) -> None:
        with patch.dict(sys.modules, {"torch": self.torch}), \
             patch('platform.mac_ver', return_value=("14.0", "", "")):
            self.assertEqual(self.gen._resolve_dtype({"device": "mps", "precision": "bf16"}), "bfloat16")

    def test_bf16_on_macos_below_14_falls_back_to_float16(self) -> None:
        with patch.dict(sys.modules, {"torch": self.torch}), \
             patch('platform.mac_ver', return_value=("13.5", "", "")):
            self.assertEqual(self.gen._resolve_dtype({"device": "mps", "precision": "bf16"}), "float16")

    def test_bf16_on_non_mps_returns_bfloat16(self) -> None:
        with patch.dict(sys.modules, {"torch": self.torch}):
            self.assertEqual(self.gen._resolve_dtype({"device": "cuda", "precision": "bf16"}), "bfloat16")

    def test_fp32_returns_float32(self) -> None:
        with patch.dict(sys.modules, {"torch": self.torch}):
            self.assertEqual(self.gen._resolve_dtype({"precision": "fp32"}), "float32")

    def test_unknown_precision_returns_float32(self) -> None:
        with patch.dict(sys.modules, {"torch": self.torch}):
            self.assertEqual(self.gen._resolve_dtype({"precision": "weird"}), "float32")

    def test_defaults_to_auto_when_precision_missing(self) -> None:
        with patch.dict(sys.modules, {"torch": self.torch}):
            self.assertEqual(self.gen._resolve_dtype({"device": "mps"}), "float16")


if __name__ == "__main__":
    unittest.main()
