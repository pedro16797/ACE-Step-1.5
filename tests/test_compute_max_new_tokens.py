"""Unit tests for LLMHandler._compute_max_new_tokens.

Validates that the progress bar total (max_new_tokens) is computed correctly
for both the CoT and codes generation phases, ensuring the progress bar
reaches ~100% instead of appearing to finish early.
"""

import unittest
from unittest.mock import MagicMock


class TestComputeMaxNewTokens(unittest.TestCase):
    """Tests for _compute_max_new_tokens helper method."""

    def _make_handler(self, max_model_len: int = 4096):
        """Create a minimal LLMHandler with mocked dependencies."""
        from acestep.llm_inference import LLMHandler

        handler = LLMHandler.__new__(LLMHandler)
        handler.max_model_len = max_model_len
        return handler

    # ------------------------------------------------------------------
    # Codes phase: should use target_codes + 10 (small buffer)
    # ------------------------------------------------------------------

    def test_codes_phase_195s(self):
        """195s duration in codes phase -> 975 + 10 = 985."""
        handler = self._make_handler()
        result = handler._compute_max_new_tokens(
            target_duration=195.0, generation_phase="codes"
        )
        self.assertEqual(result, 985)

    def test_codes_phase_60s(self):
        """60s duration in codes phase -> 300 + 10 = 310."""
        handler = self._make_handler()
        result = handler._compute_max_new_tokens(
            target_duration=60.0, generation_phase="codes"
        )
        self.assertEqual(result, 310)

    # ------------------------------------------------------------------
    # CoT phase: should use target_codes + 500 (large buffer for metadata)
    # ------------------------------------------------------------------

    def test_cot_phase_195s(self):
        """195s duration in cot phase -> 975 + 500 = 1475."""
        handler = self._make_handler()
        result = handler._compute_max_new_tokens(
            target_duration=195.0, generation_phase="cot"
        )
        self.assertEqual(result, 1475)

    def test_cot_phase_60s(self):
        """60s duration in cot phase -> 300 + 500 = 800."""
        handler = self._make_handler()
        result = handler._compute_max_new_tokens(
            target_duration=60.0, generation_phase="cot"
        )
        self.assertEqual(result, 800)

    # ------------------------------------------------------------------
    # Capping at max_model_len
    # ------------------------------------------------------------------

    def test_capped_by_max_model_len(self):
        """Result should be capped at max_model_len - 64."""
        handler = self._make_handler(max_model_len=512)
        result = handler._compute_max_new_tokens(
            target_duration=195.0, generation_phase="codes"
        )
        self.assertEqual(result, 512 - 64)

    # ------------------------------------------------------------------
    # Duration clamping
    # ------------------------------------------------------------------

    def test_duration_clamp_low(self):
        """Duration below 10s is clamped to 10s -> 50 + 10 = 60."""
        handler = self._make_handler()
        result = handler._compute_max_new_tokens(
            target_duration=3.0, generation_phase="codes"
        )
        self.assertEqual(result, 60)

    def test_duration_clamp_high(self):
        """Duration above 600s is clamped to 600s -> 3000 + 10 = 3010."""
        handler = self._make_handler()
        result = handler._compute_max_new_tokens(
            target_duration=999.0, generation_phase="codes"
        )
        self.assertEqual(result, 3010)

    # ------------------------------------------------------------------
    # Fallback when target_duration is None
    # ------------------------------------------------------------------

    def test_fallback_with_explicit_value(self):
        """When target_duration is None, use fallback_max."""
        handler = self._make_handler()
        result = handler._compute_max_new_tokens(
            target_duration=None, generation_phase="codes", fallback_max=2048
        )
        self.assertEqual(result, 2048)

    def test_fallback_default(self):
        """When target_duration is None and no fallback_max, use max_model_len - 64."""
        handler = self._make_handler(max_model_len=4096)
        result = handler._compute_max_new_tokens(
            target_duration=None, generation_phase="codes"
        )
        self.assertEqual(result, 4096 - 64)

    # ------------------------------------------------------------------
    # Regression: the original bug scenario
    # ------------------------------------------------------------------

    def test_regression_progress_bar_not_inflated(self):
        """
        Regression test for the misleading progress bar issue.

        With 195s duration and codes phase, the old code produced 1475 tokens
        (975 + 500) but the constrained decoder forced EOS at 975, making the
        progress bar stop at 66%. The fix should produce 985 (975 + 10).
        """
        handler = self._make_handler()
        result = handler._compute_max_new_tokens(
            target_duration=195.0, generation_phase="codes"
        )
        target_codes = int(195.0 * 5)  # 975
        # max_new_tokens should be close to target_codes, not inflated by +500
        self.assertLessEqual(result - target_codes, 20)
        self.assertGreater(result, target_codes)


if __name__ == "__main__":
    unittest.main()
