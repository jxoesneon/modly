import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

const MODULE_PATH = fileURLToPath(new URL('./UpscaleEffect.tsx', import.meta.url))

async function readSource(): Promise<string> {
  return readFile(MODULE_PATH, 'utf-8')
}

test('shader contains the Lanczos kernel function', async () => {
  const src = await readSource()
  assert.ok(src.includes('float lanczos(float x)'), 'shader must define a lanczos function')
})

test('shader contains the sinc helper function', async () => {
  const src = await readSource()
  assert.ok(src.includes('float sinc(float x)'), 'shader must define a sinc function')
})

test('shader uses the tDiffuse uniform', async () => {
  const src = await readSource()
  assert.ok(src.includes('uniform sampler2D tDiffuse'), 'shader must declare tDiffuse uniform')
})

test('shader uses the inputResolution uniform', async () => {
  const src = await readSource()
  assert.ok(src.includes('uniform vec2 inputResolution'), 'shader must declare inputResolution uniform')
})

test('shader uses the outputResolution uniform', async () => {
  const src = await readSource()
  assert.ok(src.includes('uniform vec2 outputResolution'), 'shader must declare outputResolution uniform')
})

test('shader samples tDiffuse via texture2D', async () => {
  const src = await readSource()
  assert.ok(src.includes('texture2D(tDiffuse'), 'shader must sample tDiffuse using texture2D')
})

test('shader uses a 6x6 Lanczos-3 kernel loop', async () => {
  const src = await readSource()
  assert.ok(src.includes('for (int dy = -2; dy <= 3; ++dy)'), 'shader must loop dy from -2 to 3')
  assert.ok(src.includes('for (int dx = -2; dx <= 3; ++dx)'), 'shader must loop dx from -2 to 3')
})

test('shader normalizes by totalWeight', async () => {
  const src = await readSource()
  assert.ok(src.includes('totalWeight'), 'shader must track totalWeight')
  assert.ok(src.includes('color /= totalWeight'), 'shader must normalize color by totalWeight')
})

test('UpscaleEffectImpl class extends Effect', async () => {
  const src = await readSource()
  assert.ok(src.includes('class UpscaleEffectImpl extends Effect'), 'UpscaleEffectImpl must extend Effect')
})

test('UpscaleEffectImpl registers tDiffuse, inputResolution, and outputResolution uniforms', async () => {
  const src = await readSource()
  assert.ok(src.includes("uniforms.set('tDiffuse'"), 'must register tDiffuse uniform')
  assert.ok(src.includes("uniforms.set('inputResolution'"), 'must register inputResolution uniform')
  assert.ok(src.includes("uniforms.set('outputResolution'"), 'must register outputResolution uniform')
})

test('setSize updates inputResolution to half of output dimensions', async () => {
  const src = await readSource()
  assert.ok(src.includes('width * 0.5'), 'setSize must set inputResolution to half width')
  assert.ok(src.includes('height * 0.5'), 'setSize must set inputResolution to half height')
})

test('setSize updates outputResolution to full dimensions', async () => {
  const src = await readSource()
  assert.ok(src.includes("this.uniforms.get('outputResolution')"), 'setSize must update outputResolution uniform')
})

test('UpscaleEffect is exported as a function component', async () => {
  const src = await readSource()
  assert.ok(src.includes('export function UpscaleEffect'), 'UpscaleEffect must be exported as a function')
})

test('shader clamps sample coordinates to valid range', async () => {
  const src = await readSource()
  assert.ok(src.includes('clamp(sampleCoord'), 'shader must clamp sample coordinates')
})

test('shader uses Lanczos-3 with a=3 support window', async () => {
  const src = await readSource()
  // Lanczos-3 returns 0 for |x| >= 3
  assert.ok(src.includes('abs(x) >= 3.0'), 'lanczos function must window at |x| >= 3.0')
  assert.ok(src.includes('sinc(x / 3.0)'), 'lanczos must use sinc(x / 3.0) for the lobes')
})
