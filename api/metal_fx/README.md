# MetalFX Spatial Upscaling

This module provides MetalFX-based spatial upscaling for rendered frames and textures on Apple Silicon.

## Usage

```python
from metal_fx import MetalFXScaler, is_metalfx_supported

if is_metalfx_supported():
    scaler = MetalFXScaler(
        input_width=1280, input_height=720,
        output_width=2560, output_height=1440
    )
    upscaled = scaler.upscale(low_res_rgba_image)
```

## Architecture

The MetalFX spatial scaler uses Apple's MTLFXSpatialScaler API (macOS 13.0+)
to perform ML-based spatial upscaling. When the native MetalFX bridge is not
compiled, a high-quality Lanczos-3 fallback is used.

For the Electron 3D viewer (WebGL-based), a GLSL upscaling post-processing
pass is used in the frontend since WebGL cannot access MetalFX directly.
