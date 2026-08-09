import assert from 'node:assert/strict'
import test from 'node:test'

async function loadModule() {
  return import(new URL('./chip-detection.ts', import.meta.url).href)
}

test('getChipFamily returns a string', async () => {
  const { getChipFamily } = await loadModule()
  const family = getChipFamily()
  assert.equal(typeof family, 'string')
  assert.ok(family.length > 0)
})

test('getChipFamily returns one of the known families or "unknown"', async () => {
  const { getChipFamily } = await loadModule()
  const family = getChipFamily()
  const known = ['m1', 'm2', 'm3', 'm4', 'unknown']
  assert.ok(known.includes(family), `expected one of ${known.join(', ')}, got "${family}"`)
})

test('getChipFamily returns "m4" on this M4 Donan machine', async () => {
  const { getChipFamily } = await loadModule()
  const family = getChipFamily()
  assert.equal(family, 'm4')
})

test('getChipTier returns a number', async () => {
  const { getChipTier } = await loadModule()
  const tier = getChipTier()
  assert.equal(typeof tier, 'number')
  assert.ok(Number.isInteger(tier))
  assert.ok(tier >= 0 && tier <= 4)
})

test('getChipTier returns 4 on this M4 machine', async () => {
  const { getChipTier } = await loadModule()
  const tier = getChipTier()
  assert.equal(tier, 4)
})

test('getChipTier is consistent with getChipFamily', async () => {
  const { getChipFamily, getChipTier } = await loadModule()
  const family = getChipFamily()
  const tier = getChipTier()
  const expected: Record<string, number> = { m1: 1, m2: 2, m3: 3, m4: 4, unknown: 0 }
  assert.equal(tier, expected[family])
})

test('getCpuFamily returns a number or null', async () => {
  const { getCpuFamily } = await loadModule()
  const cpufamily = getCpuFamily()
  assert.ok(cpufamily === null || typeof cpufamily === 'number')
})

test('getCpuFamily returns the M4 Donan cpufamily constant on this machine', async () => {
  const { getCpuFamily } = await loadModule()
  const cpufamily = getCpuFamily()
  // CPUFAMILY_ARM_DONAN = 0x6f5129ac (M4 base / A18)
  assert.equal(cpufamily, 0x6f5129ac)
})

test('isAppleSilicon returns a boolean', async () => {
  const { isAppleSilicon } = await loadModule()
  const result = isAppleSilicon()
  assert.equal(typeof result, 'boolean')
})

test('isAppleSilicon returns true on this Apple Silicon machine', async () => {
  const { isAppleSilicon } = await loadModule()
  assert.equal(isAppleSilicon(), true)
})

test('getChipInfo returns an object with all expected fields', async () => {
  const { getChipInfo } = await loadModule()
  const info = getChipInfo()
  assert.equal(typeof info, 'object')
  assert.ok(info !== null)
  // Verify all expected fields exist with correct types
  assert.equal(typeof info.family, 'string')
  assert.equal(typeof info.tier, 'number')
  assert.equal(typeof info.brand, 'string')
  assert.equal(typeof info.bfloat16, 'boolean')
  assert.equal(typeof info.sme, 'boolean')
  assert.equal(typeof info.sme2, 'boolean')
  assert.ok(info.cpufamily === null || typeof info.cpufamily === 'number')
})

test('getChipInfo fields are internally consistent', async () => {
  const { getChipInfo, getChipFamily, getChipTier, getBrandString, hasBfloat16, hasSME, hasSME2, getCpuFamily } = await loadModule()
  const info = getChipInfo()
  assert.equal(info.family, getChipFamily())
  assert.equal(info.tier, getChipTier())
  assert.equal(info.brand, getBrandString())
  assert.equal(info.bfloat16, hasBfloat16())
  assert.equal(info.sme, hasSME())
  assert.equal(info.sme2, hasSME2())
  assert.equal(info.cpufamily, getCpuFamily())
})

test('getChipInfo on this M4 machine returns m4 family and tier 4', async () => {
  const { getChipInfo } = await loadModule()
  const info = getChipInfo()
  assert.equal(info.family, 'm4')
  assert.equal(info.tier, 4)
  assert.equal(info.cpufamily, 0x6f5129ac)
  assert.ok(info.brand.includes('Apple M4'))
})

test('hasBfloat16, hasSME, hasSME2 return booleans', async () => {
  const { hasBfloat16, hasSME, hasSME2 } = await loadModule()
  assert.equal(typeof hasBfloat16(), 'boolean')
  assert.equal(typeof hasSME(), 'boolean')
  assert.equal(typeof hasSME2(), 'boolean')
})

test('getBrandString returns a non-empty string on macOS', async () => {
  const { getBrandString } = await loadModule()
  const brand = getBrandString()
  assert.equal(typeof brand, 'string')
  assert.ok(brand.length > 0)
  assert.ok(brand.includes('Apple'))
})
