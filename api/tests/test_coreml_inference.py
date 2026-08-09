"""Tests for api/services/coreml_inference.py — CoreML / ANE inference."""
import sys
import unittest
from unittest.mock import patch, MagicMock

from services import coreml_inference as ci
from services.coreml_inference import (
    is_coreml_available, is_ane_available, get_ane_info,
    CoreMLInferenceEngine, convert_pytorch_to_coreml,
)


class IsCoremlAvailableTests(unittest.TestCase):
    def test_false_on_non_darwin(self) -> None:
        with patch.object(sys, 'platform', 'linux'):
            self.assertFalse(is_coreml_available())

    def test_false_on_darwin_without_coremltools(self) -> None:
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "coremltools":
                raise ImportError("no coremltools")
            return real_import(name, *args, **kwargs)

        with patch.object(sys, 'platform', 'darwin'), \
             patch('services.coreml_inference.platform.mac_ver', return_value=("13.0", "", "")), \
             patch('builtins.__import__', side_effect=fake_import):
            self.assertFalse(is_coreml_available())

    def test_true_on_darwin_with_coremltools(self) -> None:
        with patch.object(sys, 'platform', 'darwin'), \
             patch('services.coreml_inference.platform.mac_ver', return_value=("13.0", "", "")):
            # Inject a fake coremltools module so the import succeeds.
            fake_ct = MagicMock()
            fake_ct.ComputeUnit.ALL = "ALL"
            with patch.dict(sys.modules, {"coremltools": fake_ct}):
                self.assertTrue(is_coreml_available())

    def test_false_on_darwin_old_macos(self) -> None:
        with patch.object(sys, 'platform', 'darwin'), \
             patch('services.coreml_inference.platform.mac_ver', return_value=("11.0", "", "")):
            self.assertFalse(is_coreml_available())


class IsAneAvailableTests(unittest.TestCase):
    def test_false_when_coreml_unavailable(self) -> None:
        with patch('services.coreml_inference.is_coreml_available', return_value=False):
            self.assertFalse(is_ane_available())

    def test_true_when_coreml_available_and_macos_13_plus(self) -> None:
        with patch('services.coreml_inference.is_coreml_available', return_value=True), \
             patch('services.coreml_inference.platform.mac_ver', return_value=("13.0", "", "")):
            self.assertTrue(is_ane_available())

    def test_false_when_coreml_available_but_old_macos(self) -> None:
        with patch('services.coreml_inference.is_coreml_available', return_value=True), \
             patch('services.coreml_inference.platform.mac_ver', return_value=("12.0", "", "")):
            self.assertFalse(is_ane_available())


class GetAneInfoTests(unittest.TestCase):
    def test_returns_dict_with_expected_keys(self) -> None:
        with patch('services.coreml_inference.is_coreml_available', return_value=False), \
             patch('services.coreml_inference.is_ane_available', return_value=False), \
             patch('services.coreml_inference.platform.mac_ver', return_value=("13.0", "", "")):
            info = get_ane_info()
        expected_keys = {
            "coreml_available", "ane_available", "macos_version",
            "min_coreml_macos", "min_ane_macos",
        }
        self.assertEqual(set(info.keys()), expected_keys)

    def test_reports_unavailable_when_coreml_missing(self) -> None:
        with patch('services.coreml_inference.is_coreml_available', return_value=False), \
             patch('services.coreml_inference.is_ane_available', return_value=False), \
             patch('services.coreml_inference.platform.mac_ver', return_value=("13.0", "", "")):
            info = get_ane_info()
        self.assertFalse(info["coreml_available"])
        self.assertFalse(info["ane_available"])


class CoreMLInferenceEngineInitTests(unittest.TestCase):
    def test_raises_runtime_error_when_coreml_unavailable(self) -> None:
        with patch('services.coreml_inference.is_coreml_available', return_value=False):
            with self.assertRaises(RuntimeError):
                CoreMLInferenceEngine("model.mlpackage")

    def test_raises_file_not_found_for_missing_model(self) -> None:
        fake_ct = MagicMock()
        with patch('services.coreml_inference.is_coreml_available', return_value=True), \
             patch.dict(sys.modules, {"coremltools": fake_ct}):
            with self.assertRaises(FileNotFoundError):
                CoreMLInferenceEngine("/nonexistent/path/model.mlpackage")


class CoreMLInferenceEngineComputeUnitsTests(unittest.TestCase):
    """compute_units string -> CoreML enum mapping + prefer_ane override."""

    def _make_engine(self, prefer_ane: bool, compute_units: str):
        fake_ct = MagicMock()
        fake_ct.ComputeUnit.ALL = "ALL"
        fake_ct.ComputeUnit.CPU_AND_GPU = "CPU_AND_GPU"
        fake_ct.ComputeUnit.CPU_AND_NEURAL_ENGINE = "CPU_AND_NEURAL_ENGINE"
        fake_ct.ComputeUnit.CPU_ONLY = "CPU_ONLY"
        spec = MagicMock()
        spec.description.input = [MagicMock(name="input")]
        spec.description.output = [MagicMock(name="output")]
        model = MagicMock()
        model.get_spec.return_value = spec
        fake_ct.models.MLModel.return_value = model

        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(suffix=".mlpackage", delete=False)
        tmp.close()
        try:
            with patch('services.coreml_inference.is_coreml_available', return_value=True), \
                 patch.dict(sys.modules, {"coremltools": fake_ct}):
                eng = CoreMLInferenceEngine(tmp.name, prefer_ane=prefer_ane,
                                             compute_units=compute_units)
            return eng, fake_ct.models.MLModel
        finally:
            os.unlink(tmp.name)

    def test_prefer_ane_maps_to_cpu_and_neural_engine(self) -> None:
        eng, mlmodel_ctor = self._make_engine(prefer_ane=True, compute_units="all")
        _, kwargs = mlmodel_ctor.call_args
        self.assertEqual(kwargs.get("compute_units"), "CPU_AND_NEURAL_ENGINE")

    def test_all_compute_units(self) -> None:
        eng, mlmodel_ctor = self._make_engine(prefer_ane=False, compute_units="all")
        _, kwargs = mlmodel_ctor.call_args
        self.assertEqual(kwargs.get("compute_units"), "ALL")

    def test_cpu_and_gpu_compute_units(self) -> None:
        eng, mlmodel_ctor = self._make_engine(prefer_ane=False, compute_units="cpu_and_gpu")
        _, kwargs = mlmodel_ctor.call_args
        self.assertEqual(kwargs.get("compute_units"), "CPU_AND_GPU")

    def test_cpu_only_compute_units(self) -> None:
        eng, mlmodel_ctor = self._make_engine(prefer_ane=False, compute_units="cpu_only")
        _, kwargs = mlmodel_ctor.call_args
        self.assertEqual(kwargs.get("compute_units"), "CPU_ONLY")

    def test_cpu_and_neural_engine_compute_units(self) -> None:
        eng, mlmodel_ctor = self._make_engine(prefer_ane=False, compute_units="cpu_and_neural_engine")
        _, kwargs = mlmodel_ctor.call_args
        self.assertEqual(kwargs.get("compute_units"), "CPU_AND_NEURAL_ENGINE")

    def test_prefer_ane_does_not_override_explicit_compute_units(self) -> None:
        eng, mlmodel_ctor = self._make_engine(prefer_ane=True, compute_units="cpu_only")
        _, kwargs = mlmodel_ctor.call_args
        self.assertEqual(kwargs.get("compute_units"), "CPU_ONLY")


class CoreMLInferenceEnginePredictTests(unittest.TestCase):
    def test_predict_delegates_to_model(self) -> None:
        fake_ct = MagicMock()
        fake_ct.ComputeUnit.ALL = "ALL"
        spec = MagicMock()
        spec.description.input = [MagicMock(name="input")]
        spec.description.output = [MagicMock(name="output")]
        model = MagicMock()
        model.get_spec.return_value = spec
        model.predict.return_value = {"output": "result"}
        fake_ct.models.MLModel.return_value = model

        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(suffix=".mlpackage", delete=False)
        tmp.close()
        try:
            with patch('services.coreml_inference.is_coreml_available', return_value=True), \
                 patch.dict(sys.modules, {"coremltools": fake_ct}):
                eng = CoreMLInferenceEngine(tmp.name)
            result = eng.predict({"input": "data"})
            self.assertEqual(result, {"output": "result"})
            eng.close()
            self.assertIsNone(eng._model)
        finally:
            os.unlink(tmp.name)


class ConvertPytorchToCoremlTests(unittest.TestCase):
    def test_raises_runtime_error_when_coreml_unavailable(self) -> None:
        with patch('services.coreml_inference.is_coreml_available', return_value=False):
            with self.assertRaises(RuntimeError):
                convert_pytorch_to_coreml(MagicMock(), (1, 3, 256, 256))

    def test_conversion_success_path(self) -> None:
        fake_ct = MagicMock()
        fake_ct.target.macOS13 = "macOS13"
        fake_ct.target.macOS14 = "macOS14"
        fake_ct.precision.FLOAT16 = "FLOAT16"
        fake_ct.precision.FLOAT32 = "FLOAT32"
        fake_ct.TensorType = MagicMock()
        fake_ct.convert.return_value = "converted-model"

        fake_torch = MagicMock()
        fake_torch.rand.return_value = "rand-input"
        fake_torch.jit.trace.return_value = "traced-model"

        model = MagicMock()

        with patch('services.coreml_inference.is_coreml_available', return_value=True), \
             patch.dict(sys.modules, {"coremltools": fake_ct, "torch": fake_torch}):
            result = convert_pytorch_to_coreml(model, (1, 3, 256, 256), use_fp16=True)
        self.assertEqual(result, "converted-model")
        model.eval.assert_called_once()
        fake_torch.jit.trace.assert_called_once_with(model, "rand-input")
        fake_ct.convert.assert_called_once()


if __name__ == "__main__":
    unittest.main()
