"""
CoreML inference service for Apple Neural Engine (ANE) acceleration.

Provides an alternative inference path for extensions that want to use
CoreML + ANE instead of PyTorch + MPS. This is particularly beneficial for:
  - Fixed-shape models (ANE requires static shapes)
  - Compute-bound operations (ANE can be 10x faster than GPU for some ops)
  - Memory-constrained scenarios (ANE uses less memory than GPU)

Usage from extension generators:
    from services.coreml_inference import CoreMLInferenceEngine
    
    engine = CoreMLInferenceEngine("model.mlpackage", prefer_ane=True)
    output = engine.predict({"input": image_array})

Conversion from PyTorch:
    from services.coreml_inference import convert_pytorch_to_coreml
    mlmodel = convert_pytorch_to_coreml(pytorch_model, input_shape=(1, 3, 256, 256))
    mlmodel.save("model.mlpackage")
"""
import os
import sys
import platform
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, List

logger = logging.getLogger(__name__)

# Minimum macOS version for CoreML ML Programs
_COREML_MIN_MACOS = (12, 0)
# Minimum macOS version for good ANE support
_ANE_MIN_MACOS = (13, 0)
# Minimum macOS version for stateful models (KV cache)
_STATEFUL_MIN_MACOS = (14, 0)


def is_coreml_available() -> bool:
    """Check if CoreML is available on this platform."""
    if sys.platform != "darwin":
        return False
    mac_ver = tuple(int(x) for x in platform.mac_ver()[0].split(".")[:2])
    if mac_ver < _COREML_MIN_MACOS:
        return False
    try:
        import coremltools
        return True
    except ImportError:
        return False


def is_ane_available() -> bool:
    """Check if the Apple Neural Engine is available."""
    if not is_coreml_available():
        return False
    mac_ver = tuple(int(x) for x in platform.mac_ver()[0].split(".")[:2])
    return mac_ver >= _ANE_MIN_MACOS


class CoreMLInferenceEngine:
    """
    CoreML inference engine with ANE preference.
    
    Loads a CoreML model and runs inference, preferring the Neural Engine
    for compute-bound operations. Falls back to GPU or CPU as needed.
    
    Args:
        model_path: Path to .mlpackage or .mlmodel file
        prefer_ane: If True, prefer ANE. If False, use default (all units).
        compute_units: Override compute units. One of 'all', 'cpu_and_gpu',
                       'cpu_and_neural_engine', 'cpu_only'.
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        prefer_ane: bool = True,
        compute_units: str = "all",
    ):
        if not is_coreml_available():
            raise RuntimeError(
                "CoreML is not available. Install coremltools: pip install coremltools"
            )
        
        import coremltools as ct
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"CoreML model not found: {self.model_path}")
        
        # Map compute units string to CoreML enum
        compute_units_map = {
            "all": ct.ComputeUnit.ALL,
            "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
            "cpu_and_neural_engine": ct.ComputeUnit.CPU_AND_NEURAL_ENGINE,
            "cpu_only": ct.ComputeUnit.CPU_ONLY,
        }
        
        if prefer_ane and compute_units == "all":
            # Prefer ANE by requesting CPU_AND_NEURAL_ENGINE
            cu = ct.ComputeUnit.CPU_AND_NEURAL_ENGINE
        else:
            cu = compute_units_map.get(compute_units, ct.ComputeUnit.ALL)
        
        # Load the model with the specified compute units
        self._model = ct.models.MLModel(
            str(self.model_path),
            compute_units=cu
        )
        
        self._spec = self._model.get_spec()
        self._input_names = [inp.name for inp in self._spec.description.input]
        self._output_names = [out.name for out in self._spec.description.output]
        
        logger.info(
            f"CoreML model loaded: {self.model_path.name}, "
            f"inputs={self._input_names}, outputs={self._output_names}, "
            f"compute_units={compute_units}"
        )
    
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference on the CoreML model.
        
        Args:
            inputs: Dict mapping input names to numpy arrays or PIL Images
        
        Returns:
            Dict mapping output names to numpy arrays (MLMultiArray)
        """
        return self._model.predict(inputs)
    
    def predict_single(self, input_name: str, input_data: Any) -> Dict[str, Any]:
        """Convenience method for single-input models."""
        return self._model.predict({input_name: input_data})
    
    @property
    def input_names(self) -> List[str]:
        return self._input_names
    
    @property
    def output_names(self) -> List[str]:
        return self._output_names
    
    def close(self):
        """Release the model."""
        self._model = None


def convert_pytorch_to_coreml(
    model,
    input_shape: tuple,
    input_name: str = "input",
    output_name: str = "output",
    min_macos: tuple = (13, 0),
    use_fp16: bool = True,
) -> "ct.models.MLModel":
    """
    Convert a PyTorch model to CoreML format.
    
    Args:
        model: PyTorch model (must be in eval mode)
        input_shape: Tuple describing the input tensor shape, e.g. (1, 3, 256, 256)
        input_name: Name for the input feature
        output_name: Name for the output feature
        min_macos: Minimum macOS version target (affects model format)
        use_fp16: Use float16 precision (recommended for ANE)
    
    Returns:
        CoreML MLModel object
    
    Example:
        import torch
        model = MyModel().eval()
        mlmodel = convert_pytorch_to_coreml(model, input_shape=(1, 3, 256, 256))
        mlmodel.save("my_model.mlpackage")
    """
    if not is_coreml_available():
        raise RuntimeError("CoreML is not available. Install coremltools.")
    
    import coremltools as ct
    import torch
    
    # Ensure model is in eval mode
    model.eval()
    
    # Create example input for tracing
    example_input = torch.rand(*input_shape)
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Convert to CoreML
    target_map = {
        (12, 0): ct.target.macOS12,
        (13, 0): ct.target.macOS13,
        (14, 0): ct.target.macOS14,
        (15, 0): ct.target.macOS15,
    }
    deployment_target = target_map.get(min_macos, ct.target.macOS13)
    
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name=input_name, shape=input_shape)],
        outputs=[ct.TensorType(name=output_name)],
        minimum_deployment_target=deployment_target,
        compute_precision=ct.precision.FLOAT16 if use_fp16 else ct.precision.FLOAT32,
    )
    
    return mlmodel


def get_ane_info() -> Dict[str, Any]:
    """Return information about ANE availability and capabilities."""
    return {
        "coreml_available": is_coreml_available(),
        "ane_available": is_ane_available(),
        "macos_version": platform.mac_ver()[0],
        "min_coreml_macos": ".".join(str(x) for x in _COREML_MIN_MACOS),
        "min_ane_macos": ".".join(str(x) for x in _ANE_MIN_MACOS),
    }
