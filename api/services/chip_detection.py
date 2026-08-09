"""
Apple Silicon chip family detection via sysctl.

Detects M1/M2/M3/M4 family and feature flags (BF16, SME, etc.)
Used by:
  - texture_baker/setup.py and uv_unwrapper/setup.py for -mcpu selection
  - extension_process.py for precision/device env vars
  - runner.py for capability reporting
"""
import subprocess
import sys
from functools import lru_cache
from typing import Optional, Dict

# hw.cpufamily values (from XNU kernel source)
_CPUFAMILY_ARM_FIRESTORM_ICESTORM = 0x1b588bb3  # M1 / A14
_CPUFAMILY_ARM_BLIZZARD_AVALANCHE = 0xda33d83d   # M2 / A15
_CPUFAMILY_ARM_EVEREST_SAWTOOTH   = 0x8765edea   # M3 / A17 Pro
_CPUFAMILY_ARM_DONAN              = 0x6f5129ac   # M4 (base) / A18
_CPUFAMILY_ARM_BRAVA              = 0x17d5b93a   # M4 Pro / M4 Max
_CPUFAMILY_ARM_TAHITI             = 0x75d4acb9   # M4 Ultra

CHIP_FAMILY_NAMES = {
    _CPUFAMILY_ARM_FIRESTORM_ICESTORM: "m1",
    _CPUFAMILY_ARM_BLIZZARD_AVALANCHE: "m2",
    _CPUFAMILY_ARM_EVEREST_SAWTOOTH:   "m3",
    _CPUFAMILY_ARM_DONAN:              "m4",
    _CPUFAMILY_ARM_BRAVA:              "m4",
    _CPUFAMILY_ARM_TAHITI:             "m4",
}

# Map chip family to clang -mcpu target string
CHIP_CPU_TARGETS = {
    "m1": "apple-m1",
    "m2": "apple-m2",
    "m3": "apple-m3",
    "m4": "apple-m4",
}


def _sysctl(key: str) -> str:
    """Read a sysctl key, return empty string on failure."""
    if sys.platform != "darwin":
        return ""
    try:
        result = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _sysctl_int(key: str) -> Optional[int]:
    """Read a sysctl key as int, return None on failure."""
    val = _sysctl(key)
    if not val:
        return None
    try:
        return int(val)
    except ValueError:
        return None


@lru_cache(maxsize=1)
def get_cpufamily() -> Optional[int]:
    """Return the hw.cpufamily value, or None on non-Apple platforms."""
    return _sysctl_int("hw.cpufamily")


@lru_cache(maxsize=1)
def get_chip_family() -> str:
    """Return 'm1', 'm2', 'm3', 'm4', or 'unknown'."""
    family = get_cpufamily()
    if family is None:
        return "unknown"
    return CHIP_FAMILY_NAMES.get(family, "unknown")


@lru_cache(maxsize=1)
def get_chip_tier() -> int:
    """Return a numeric tier: 1=M1, 2=M2, 3=M3, 4=M4, 0=unknown/non-Apple."""
    family = get_chip_family()
    if family == "m1": return 1
    if family == "m2": return 2
    if family == "m3": return 3
    if family == "m4": return 4
    return 0


@lru_cache(maxsize=1)
def get_brand_string() -> str:
    """Return the CPU brand string, e.g. 'Apple M4 Pro'."""
    return _sysctl("machdep.cpu.brand_string")


@lru_cache(maxsize=1)
def has_bfloat16() -> bool:
    """Check if the CPU supports bfloat16 (FEAT_BF16). M2+, not M1."""
    val = _sysctl_int("hw.optional.arm.FEAT_BF16")
    return val == 1


@lru_cache(maxsize=1)
def has_sme() -> bool:
    """Check if the CPU supports Scalable Matrix Extension (M4+)."""
    val = _sysctl_int("hw.optional.arm.FEAT_SME")
    return val == 1


@lru_cache(maxsize=1)
def has_sme2() -> bool:
    """Check if the CPU supports SME2 (M4+)."""
    val = _sysctl_int("hw.optional.arm.FEAT_SME2")
    return val == 1


@lru_cache(maxsize=1)
def has_fp16fml() -> bool:
    """Check if the CPU supports FP16 FMLA (fused multiply-accumulate)."""
    val = _sysctl_int("hw.optional.arm.FEAT_FP16FML")
    return val == 1


@lru_cache(maxsize=1)
def get_clang_cpu_target() -> str:
    """Return the best clang -mcpu target for this chip, or 'apple-m1' as safe default."""
    family = get_chip_family()
    return CHIP_CPU_TARGETS.get(family, "apple-m1")


@lru_cache(maxsize=1)
def get_capabilities() -> Dict:
    """Return a dict of all detected capabilities for logging/IPC."""
    return {
        "chip_family": get_chip_family(),
        "chip_tier": get_chip_tier(),
        "brand_string": get_brand_string(),
        "cpufamily": get_cpufamily(),
        "bfloat16": has_bfloat16(),
        "sme": has_sme(),
        "sme2": has_sme2(),
        "fp16fml": has_fp16fml(),
        "clang_cpu_target": get_clang_cpu_target(),
        "platform": sys.platform,
    }


def is_apple_silicon() -> bool:
    """Return True on Apple Silicon (arm64 macOS)."""
    return sys.platform == "darwin" and sys.maxsize > 2**32
