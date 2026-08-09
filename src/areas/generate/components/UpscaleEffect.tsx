import { useEffect, useMemo } from 'react'
import { useThree } from '@react-three/fiber'
import { Effect } from 'postprocessing'
import { Uniform, Vector2 } from 'three'

// Lanczos-3 upscaling shader for high-quality spatial upscaling.
// This is the WebGL fallback for MetalFX on non-Apple-Silicon platforms.
// On Apple Silicon, MetalFX would be used via a native bridge, but since
// the 3D viewer runs in WebGL (not native Metal), this GLSL implementation
// provides the same quality level within the rendering pipeline.
const UPSCALE_FRAGMENT_SHADER = `
uniform sampler2D tDiffuse;
uniform vec2 inputResolution;
uniform vec2 outputResolution;
varying vec2 vUv;

// Lanczos sinc function
float sinc(float x) {
  if (x == 0.0) return 1.0;
  float px = 3.14159265 * x;
  return sin(px) / px;
}

// Lanczos kernel (a=3)
float lanczos(float x) {
  if (abs(x) >= 3.0) return 0.0;
  return sinc(x) * sinc(x / 3.0);
}

void main() {
  // Calculate the source pixel coordinate
  vec2 srcPixel = vUv * inputResolution - 0.5;
  vec2 srcCoord = floor(srcPixel);
  vec2 frac = srcPixel - srcCoord;

  vec4 color = vec4(0.0);
  float totalWeight = 0.0;

  // Sample a 6x6 kernel (Lanczos-3 support is [-3, 3])
  for (int dy = -2; dy <= 3; ++dy) {
    for (int dx = -2; dx <= 3; ++dx) {
      vec2 sampleCoord = (srcCoord + vec2(float(dx), float(dy)) + 0.5) / inputResolution;
      sampleCoord = clamp(sampleCoord, vec2(0.0), vec2(1.0));

      float weightX = lanczos(float(dx) - frac.x);
      float weightY = lanczos(float(dy) - frac.y);
      float weight = weightX * weightY;

      color += texture2D(tDiffuse, sampleCoord) * weight;
      totalWeight += weight;
    }
  }

  if (totalWeight > 0.0) {
    color /= totalWeight;
  }

  gl_FragColor = color;
}
`

class UpscaleEffectImpl extends Effect {
  constructor() {
    const uniforms: Map<string, Uniform<unknown>> = new Map()
    uniforms.set('tDiffuse', new Uniform<unknown>(null))
    uniforms.set('inputResolution', new Uniform<unknown>(new Vector2()))
    uniforms.set('outputResolution', new Uniform<unknown>(new Vector2()))
    super('UpscaleEffect', UPSCALE_FRAGMENT_SHADER, {
      uniforms,
    })
  }

  setSize(width: number, height: number) {
    // inputResolution is the render resolution (half of output)
    // outputResolution is the display resolution
    ;(this.uniforms.get('inputResolution')!.value as Vector2).set(width * 0.5, height * 0.5)
    ;(this.uniforms.get('outputResolution')!.value as Vector2).set(width, height)
  }
}

export function UpscaleEffect() {
  const { size } = useThree()
  const effect = useMemo(() => new UpscaleEffectImpl(), [])

  useEffect(() => {
    effect.setSize(size.width, size.height)
  }, [size, effect])

  return <primitive object={effect} dispose={null} />
}
