"""Tests for api/services/chip_detection.py — Apple Silicon chip family detection."""
import subprocess
import sys
import unittest
from unittest.mock import patch, MagicMock

from services import chip_detection
from services.chip_detection import (
    get_chip_family, get_chip_tier, get_clang_cpu_target,
    has_bfloat16, has_sme, has_sme2, has_fp16fml,
    get_capabilities, is_apple_silicon, get_brand_string,
    _sysctl, _sysctl_int, get_cpufamily,
    CHIP_FAMILY_NAMES, CHIP_CPU_TARGETS,
    _CPUFAMILY_ARM_FIRESTORM_ICESTORM,
    _CPUFAMILY_ARM_BLIZZARD_AVALANCHE,
    _CPUFAMILY_ARM_EVEREST_SAWTOOTH,
    _CPUFAMILY_ARM_DONAN,
    _CPUFAMILY_ARM_BRAVA,
    _CPUFAMILY_ARM_TAHITI,
)

# All lru_cache-decorated functions that must be cleared between tests.
_CACHED_FUNCS = [
    get_cpufamily, get_chip_family, get_chip_tier, get_clang_cpu_target,
    get_brand_string, has_bfloat16, has_sme, has_sme2, has_fp16fml,
    get_capabilities,
]


def _clear_all_caches() -> None:
    for fn in _CACHED_FUNCS:
        fn.cache_clear()


class ChipFamilyMappingTests(unittest.TestCase):
    """The 6 cpufamily constants map to the correct chip family names."""

    def test_m1_family_maps_to_m1(self) -> None:
        self.assertEqual(CHIP_FAMILY_NAMES[_CPUFAMILY_ARM_FIRESTORM_ICESTORM], "m1")

    def test_m2_family_maps_to_m2(self) -> None:
        self.assertEqual(CHIP_FAMILY_NAMES[_CPUFAMILY_ARM_BLIZZARD_AVALANCHE], "m2")

    def test_m3_family_maps_to_m3(self) -> None:
        self.assertEqual(CHIP_FAMILY_NAMES[_CPUFAMILY_ARM_EVEREST_SAWTOOTH], "m3")

    def test_m4_donan_maps_to_m4(self) -> None:
        self.assertEqual(CHIP_FAMILY_NAMES[_CPUFAMILY_ARM_DONAN], "m4")

    def test_m4_brava_maps_to_m4(self) -> None:
        self.assertEqual(CHIP_FAMILY_NAMES[_CPUFAMILY_ARM_BRAVA], "m4")

    def test_m4_tahiti_maps_to_m4(self) -> None:
        self.assertEqual(CHIP_FAMILY_NAMES[_CPUFAMILY_ARM_TAHITI], "m4")

    def test_all_six_families_present(self) -> None:
        self.assertEqual(len(CHIP_FAMILY_NAMES), 6)

    def test_cpu_targets_map(self) -> None:
        self.assertEqual(CHIP_CPU_TARGETS["m1"], "apple-m1")
        self.assertEqual(CHIP_CPU_TARGETS["m2"], "apple-m2")
        self.assertEqual(CHIP_CPU_TARGETS["m3"], "apple-m3")
        self.assertEqual(CHIP_CPU_TARGETS["m4"], "apple-m4")


class GetChipFamilyTests(unittest.TestCase):
    def setUp(self) -> None:
        _clear_all_caches()

    @patch('services.chip_detection._sysctl_int')
    def test_returns_m1_for_firestorm_icestorm(self, mock_sysctl) -> None:
        mock_sysctl.return_value = _CPUFAMILY_ARM_FIRESTORM_ICESTORM
        self.assertEqual(get_chip_family(), "m1")

    @patch('services.chip_detection._sysctl_int')
    def test_returns_m2_for_blizzard_avalanche(self, mock_sysctl) -> None:
        mock_sysctl.return_value = _CPUFAMILY_ARM_BLIZZARD_AVALANCHE
        self.assertEqual(get_chip_family(), "m2")

    @patch('services.chip_detection._sysctl_int')
    def test_returns_m3_for_everest_sawtooth(self, mock_sysctl) -> None:
        mock_sysctl.return_value = _CPUFAMILY_ARM_EVEREST_SAWTOOTH
        self.assertEqual(get_chip_family(), "m3")

    @patch('services.chip_detection._sysctl_int')
    def test_returns_m4_for_donan(self, mock_sysctl) -> None:
        mock_sysctl.return_value = _CPUFAMILY_ARM_DONAN
        self.assertEqual(get_chip_family(), "m4")

    @patch('services.chip_detection._sysctl_int')
    def test_returns_m4_for_brava(self, mock_sysctl) -> None:
        mock_sysctl.return_value = _CPUFAMILY_ARM_BRAVA
        self.assertEqual(get_chip_family(), "m4")

    @patch('services.chip_detection._sysctl_int')
    def test_returns_m4_for_tahiti(self, mock_sysctl) -> None:
        mock_sysctl.return_value = _CPUFAMILY_ARM_TAHITI
        self.assertEqual(get_chip_family(), "m4")

    @patch('services.chip_detection._sysctl_int')
    def test_returns_unknown_for_unrecognized_cpufamily(self, mock_sysctl) -> None:
        mock_sysctl.return_value = 0xdeadbeef
        self.assertEqual(get_chip_family(), "unknown")

    @patch('services.chip_detection._sysctl_int')
    def test_returns_unknown_when_cpufamily_is_none(self, mock_sysctl) -> None:
        mock_sysctl.return_value = None
        self.assertEqual(get_chip_family(), "unknown")


class GetChipTierTests(unittest.TestCase):
    def setUp(self) -> None:
        _clear_all_caches()

    @patch('services.chip_detection._sysctl_int')
    def test_tier_1_for_m1(self, mock_sysctl) -> None:
        mock_sysctl.return_value = _CPUFAMILY_ARM_FIRESTORM_ICESTORM
        self.assertEqual(get_chip_tier(), 1)

    @patch('services.chip_detection._sysctl_int')
    def test_tier_2_for_m2(self, mock_sysctl) -> None:
        mock_sysctl.return_value = _CPUFAMILY_ARM_BLIZZARD_AVALANCHE
        self.assertEqual(get_chip_tier(), 2)

    @patch('services.chip_detection._sysctl_int')
    def test_tier_3_for_m3(self, mock_sysctl) -> None:
        mock_sysctl.return_value = _CPUFAMILY_ARM_EVEREST_SAWTOOTH
        self.assertEqual(get_chip_tier(), 3)

    @patch('services.chip_detection._sysctl_int')
    def test_tier_4_for_m4_donan(self, mock_sysctl) -> None:
        mock_sysctl.return_value = _CPUFAMILY_ARM_DONAN
        self.assertEqual(get_chip_tier(), 4)

    @patch('services.chip_detection._sysctl_int')
    def test_tier_0_for_unknown(self, mock_sysctl) -> None:
        mock_sysctl.return_value = 0xdeadbeef
        self.assertEqual(get_chip_tier(), 0)

    @patch('services.chip_detection._sysctl_int')
    def test_tier_0_when_none(self, mock_sysctl) -> None:
        mock_sysctl.return_value = None
        self.assertEqual(get_chip_tier(), 0)


class GetClangCpuTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        _clear_all_caches()

    @patch('services.chip_detection._sysctl_int')
    def test_m1_target(self, mock_sysctl) -> None:
        mock_sysctl.return_value = _CPUFAMILY_ARM_FIRESTORM_ICESTORM
        self.assertEqual(get_clang_cpu_target(), "apple-m1")

    @patch('services.chip_detection._sysctl_int')
    def test_m4_target(self, mock_sysctl) -> None:
        mock_sysctl.return_value = _CPUFAMILY_ARM_BRAVA
        self.assertEqual(get_clang_cpu_target(), "apple-m4")

    @patch('services.chip_detection._sysctl_int')
    def test_unknown_falls_back_to_apple_m1(self, mock_sysctl) -> None:
        mock_sysctl.return_value = 0xdeadbeef
        self.assertEqual(get_clang_cpu_target(), "apple-m1")

    @patch('services.chip_detection._sysctl_int')
    def test_none_falls_back_to_apple_m1(self, mock_sysctl) -> None:
        mock_sysctl.return_value = None
        self.assertEqual(get_clang_cpu_target(), "apple-m1")


class FeatureFlagTests(unittest.TestCase):
    """has_bfloat16 / has_sme / has_sme2 / has_fp16fml read sysctl ints."""

    def setUp(self) -> None:
        _clear_all_caches()

    @patch('services.chip_detection._sysctl_int')
    def test_bfloat16_true_when_sysctl_one(self, mock_sysctl) -> None:
        mock_sysctl.return_value = 1
        self.assertTrue(has_bfloat16())

    @patch('services.chip_detection._sysctl_int')
    def test_bfloat16_false_when_sysctl_zero(self, mock_sysctl) -> None:
        mock_sysctl.return_value = 0
        self.assertFalse(has_bfloat16())

    @patch('services.chip_detection._sysctl_int')
    def test_bfloat16_false_when_none(self, mock_sysctl) -> None:
        mock_sysctl.return_value = None
        self.assertFalse(has_bfloat16())

    @patch('services.chip_detection._sysctl_int')
    def test_sme_true_when_one(self, mock_sysctl) -> None:
        mock_sysctl.return_value = 1
        self.assertTrue(has_sme())

    @patch('services.chip_detection._sysctl_int')
    def test_sme_false_when_zero(self, mock_sysctl) -> None:
        mock_sysctl.return_value = 0
        self.assertFalse(has_sme())

    @patch('services.chip_detection._sysctl_int')
    def test_sme2_true_when_one(self, mock_sysctl) -> None:
        mock_sysctl.return_value = 1
        self.assertTrue(has_sme2())

    @patch('services.chip_detection._sysctl_int')
    def test_sme2_false_when_none(self, mock_sysctl) -> None:
        mock_sysctl.return_value = None
        self.assertFalse(has_sme2())

    @patch('services.chip_detection._sysctl_int')
    def test_fp16fml_true_when_one(self, mock_sysctl) -> None:
        mock_sysctl.return_value = 1
        self.assertTrue(has_fp16fml())

    @patch('services.chip_detection._sysctl_int')
    def test_fp16fml_false_when_zero(self, mock_sysctl) -> None:
        mock_sysctl.return_value = 0
        self.assertFalse(has_fp16fml())


class SysctlHelperTests(unittest.TestCase):
    """_sysctl / _sysctl_int low-level helpers."""

    def test_sysctl_returns_empty_on_non_darwin(self) -> None:
        with patch.object(sys, 'platform', 'linux'):
            self.assertEqual(_sysctl("hw.cpufamily"), "")

    def test_sysctl_returns_stdout_on_darwin_success(self) -> None:
        proc = MagicMock(returncode=0, stdout="0x6f5129ac\n")
        with patch.object(sys, 'platform', 'darwin'), \
             patch('services.chip_detection.subprocess.run', return_value=proc):
            self.assertEqual(_sysctl("hw.cpufamily"), "0x6f5129ac")

    def test_sysctl_returns_empty_on_nonzero_returncode(self) -> None:
        proc = MagicMock(returncode=1, stdout="something")
        with patch.object(sys, 'platform', 'darwin'), \
             patch('services.chip_detection.subprocess.run', return_value=proc):
            self.assertEqual(_sysctl("hw.cpufamily"), "")

    def test_sysctl_returns_empty_on_subprocess_exception(self) -> None:
        with patch.object(sys, 'platform', 'darwin'), \
             patch('services.chip_detection.subprocess.run', side_effect=subprocess.TimeoutExpired("sysctl", 5)):
            self.assertEqual(_sysctl("hw.cpufamily"), "")

    def test_sysctl_int_returns_int_for_valid_value(self) -> None:
        with patch('services.chip_detection._sysctl', return_value="42"):
            self.assertEqual(_sysctl_int("hw.optional.arm.FEAT_BF16"), 42)

    def test_sysctl_int_returns_none_for_empty(self) -> None:
        with patch('services.chip_detection._sysctl', return_value=""):
            self.assertIsNone(_sysctl_int("hw.cpufamily"))

    def test_sysctl_int_returns_none_for_non_numeric(self) -> None:
        with patch('services.chip_detection._sysctl', return_value="not-a-number"):
            self.assertIsNone(_sysctl_int("hw.cpufamily"))


class GetCpufamilyTests(unittest.TestCase):
    def setUp(self) -> None:
        _clear_all_caches()

    @patch('services.chip_detection._sysctl_int')
    def test_returns_value_from_sysctl(self, mock_sysctl) -> None:
        mock_sysctl.return_value = _CPUFAMILY_ARM_DONAN
        self.assertEqual(get_cpufamily(), _CPUFAMILY_ARM_DONAN)

    @patch('services.chip_detection._sysctl_int')
    def test_returns_none_when_unavailable(self, mock_sysctl) -> None:
        mock_sysctl.return_value = None
        self.assertIsNone(get_cpufamily())


class GetBrandStringTests(unittest.TestCase):
    def setUp(self) -> None:
        _clear_all_caches()

    @patch('services.chip_detection._sysctl')
    def test_returns_brand_string(self, mock_sysctl) -> None:
        mock_sysctl.return_value = "Apple M4 Pro"
        self.assertEqual(get_brand_string(), "Apple M4 Pro")

    @patch('services.chip_detection._sysctl')
    def test_returns_empty_when_unavailable(self, mock_sysctl) -> None:
        mock_sysctl.return_value = ""
        self.assertEqual(get_brand_string(), "")


class GetCapabilitiesTests(unittest.TestCase):
    def setUp(self) -> None:
        _clear_all_caches()

    @patch('services.chip_detection._sysctl')
    @patch('services.chip_detection._sysctl_int')
    def test_returns_all_expected_keys(self, mock_int, mock_str) -> None:
        mock_int.return_value = _CPUFAMILY_ARM_DONAN
        mock_str.return_value = "Apple M4"
        caps = get_capabilities()
        expected_keys = {
            "chip_family", "chip_tier", "brand_string", "cpufamily",
            "bfloat16", "sme", "sme2", "fp16fml", "clang_cpu_target", "platform",
        }
        self.assertEqual(set(caps.keys()), expected_keys)

    @patch('services.chip_detection._sysctl')
    @patch('services.chip_detection._sysctl_int')
    def test_capabilities_values_for_m4(self, mock_int, mock_str) -> None:
        mock_int.return_value = _CPUFAMILY_ARM_DONAN
        mock_str.return_value = "Apple M4"
        caps = get_capabilities()
        self.assertEqual(caps["chip_family"], "m4")
        self.assertEqual(caps["chip_tier"], 4)
        self.assertEqual(caps["clang_cpu_target"], "apple-m4")
        self.assertEqual(caps["cpufamily"], _CPUFAMILY_ARM_DONAN)


class IsAppleSiliconTests(unittest.TestCase):
    def test_true_on_darwin_64bit(self) -> None:
        with patch.object(sys, 'platform', 'darwin'):
            self.assertTrue(is_apple_silicon())

    def test_false_on_linux(self) -> None:
        with patch.object(sys, 'platform', 'linux'):
            self.assertFalse(is_apple_silicon())

    def test_false_on_windows(self) -> None:
        with patch.object(sys, 'platform', 'win32'):
            self.assertFalse(is_apple_silicon())


class CacheInvalidationTests(unittest.TestCase):
    """get_chip_family is cached; cache_clear must produce fresh results."""

    def setUp(self) -> None:
        _clear_all_caches()

    @patch('services.chip_detection._sysctl_int')
    def test_cache_returns_same_value_across_calls(self, mock_sysctl) -> None:
        mock_sysctl.return_value = _CPUFAMILY_ARM_DONAN
        first = get_chip_family()
        second = get_chip_family()
        self.assertEqual(first, second)
        # _sysctl_int should only be called once due to caching
        self.assertEqual(mock_sysctl.call_count, 1)

    @patch('services.chip_detection._sysctl_int')
    def test_cache_clear_picks_up_new_value(self, mock_sysctl) -> None:
        mock_sysctl.return_value = _CPUFAMILY_ARM_FIRESTORM_ICESTORM
        self.assertEqual(get_chip_family(), "m1")
        # Clear both the family cache and the underlying cpufamily cache so the
        # new sysctl return value is observed.
        get_chip_family.cache_clear()
        get_cpufamily.cache_clear()
        mock_sysctl.return_value = _CPUFAMILY_ARM_DONAN
        self.assertEqual(get_chip_family(), "m4")


if __name__ == "__main__":
    unittest.main()
