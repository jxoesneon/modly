import { execFileSync } from 'child_process'

// hw.cpufamily values from XNU kernel source
const CPUFAMILY_ARM_FIRESTORM_ICESTORM = 0x1b588bb3  // M1 / A14
const CPUFAMILY_ARM_BLIZZARD_AVALANCHE = 0xda33d83d   // M2 / A15
const CPUFAMILY_ARM_EVEREST_SAWTOOTH   = 0x8765edea   // M3 / A17 Pro
const CPUFAMILY_ARM_DONAN              = 0x6f5129ac   // M4 (base) / A18
const CPUFAMILY_ARM_BRAVA              = 0x17d5b93a   // M4 Pro / M4 Max
const CPUFAMILY_ARM_TAHITI             = 0x75d4acb9   // M4 Ultra

const CHIP_FAMILY_NAMES: Record<number, string> = {
  [CPUFAMILY_ARM_FIRESTORM_ICESTORM]: 'm1',
  [CPUFAMILY_ARM_BLIZZARD_AVALANCHE]: 'm2',
  [CPUFAMILY_ARM_EVEREST_SAWTOOTH]:   'm3',
  [CPUFAMILY_ARM_DONAN]:              'm4',
  [CPUFAMILY_ARM_BRAVA]:              'm4',
  [CPUFAMILY_ARM_TAHITI]:             'm4',
}

function sysctl(key: string): string {
  if (process.platform !== 'darwin') return ''
  try {
    return execFileSync('sysctl', ['-n', key], { encoding: 'utf-8', timeout: 5000 }).trim()
  } catch {
    return ''
  }
}

function sysctlInt(key: string): number | null {
  const val = sysctl(key)
  if (!val) return null
  const n = parseInt(val, 10)
  return isNaN(n) ? null : n
}

export function getCpuFamily(): number | null {
  return sysctlInt('hw.cpufamily')
}

export function getChipFamily(): string {
  const family = getCpuFamily()
  if (family === null) return 'unknown'
  return CHIP_FAMILY_NAMES[family] ?? 'unknown'
}

export function getChipTier(): number {
  const family = getChipFamily()
  if (family === 'm1') return 1
  if (family === 'm2') return 2
  if (family === 'm3') return 3
  if (family === 'm4') return 4
  return 0
}

export function getBrandString(): string {
  return sysctl('machdep.cpu.brand_string')
}

export function hasBfloat16(): boolean {
  return sysctlInt('hw.optional.arm.FEAT_BF16') === 1
}

export function hasSME(): boolean {
  return sysctlInt('hw.optional.arm.FEAT_SME') === 1
}

export function hasSME2(): boolean {
  return sysctlInt('hw.optional.arm.FEAT_SME2') === 1
}

export function isAppleSilicon(): boolean {
  return process.platform === 'darwin' && process.arch === 'arm64'
}

export interface ChipInfo {
  family: string
  tier: number
  brand: string
  bfloat16: boolean
  sme: boolean
  sme2: boolean
  cpufamily: number | null
}

export function getChipInfo(): ChipInfo {
  return {
    family: getChipFamily(),
    tier: getChipTier(),
    brand: getBrandString(),
    bfloat16: hasBfloat16(),
    sme: hasSME(),
    sme2: hasSME2(),
    cpufamily: getCpuFamily(),
  }
}
