"""End-to-end pipeline test: rollout → replay → join metrics.

Requires model download — marked with @pytest.mark.model.
Run with:  pytest tests/test_pipeline.py -v -m model
"""

import pytest
import torch

from tests.conftest import model as requires_model


@requires_model
class TestPipeline:

    @pytest.fixture(scope="class")
    def pipeline_results(self, model_name, device):
        """Run the full pipeline on 1 prompt, 4 tokens."""
        from rollout.Qwen3.rollout_simulator import Qwen3RolloutSimulator
        from align.Qwen3.bf16_replay import Qwen3BF16Replayer
        from join_metrics import join_and_compute

        prompt = "What is the meaning of life?"

        # ── Rollout ──
        sim = Qwen3RolloutSimulator(model_name=model_name, device=device, seed=42)
        enc = sim.tokenizer(prompt, return_tensors="pt")
        rollout = sim.generate(
            prompt_id=0,
            input_ids=enc["input_ids"],
            max_new_tokens=4,
            temperature=1.0,
        )
        sim.cleanup()
        del sim
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # ── Replay ──
        rp = Qwen3BF16Replayer(model_name=model_name, device=device)
        prompt_ids = torch.tensor([rollout.input_ids])
        gen_ids = torch.tensor([rollout.generated_ids])
        replay = rp.replay(prompt_id=0, prompt_ids=prompt_ids, generated_ids=gen_ids)
        rp.cleanup()
        del rp
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # ── Metrics ──
        metrics = join_and_compute(rollout, replay)
        return rollout, replay, metrics

    def test_rollout_replay_same_prompt(self, pipeline_results):
        rollout, replay, _ = pipeline_results
        assert rollout.prompt_id == replay.prompt_id

    def test_logprob_deltas_nonzero(self, pipeline_results):
        _, _, metrics = pipeline_results
        assert len(metrics.logprob_deltas) > 0
        # At least some deltas should be nonzero (FP8 ≠ BF16)
        assert any(abs(d) > 1e-6 for d in metrics.logprob_deltas), (
            "Expected nonzero logprob deltas due to FP8 mismatch"
        )

    def test_hidden_mismatch_positive(self, pipeline_results):
        _, _, metrics = pipeline_results
        assert len(metrics.step_metrics) > 0
        # Hidden L2 should be positive (FP8 introduces mismatch)
        l2_values = [sm.hidden_l2 for sm in metrics.step_metrics]
        assert any(v > 0 for v in l2_values), (
            "Expected positive hidden-state L2 mismatch"
        )

    def test_sequence_accumulation(self, pipeline_results):
        _, _, metrics = pipeline_results
        assert metrics.abs_delta_T > 0, (
            "abs_Delta_T should be positive for FP8 vs BF16"
        )

    def test_attention_entropy_positive(self, pipeline_results):
        _, _, metrics = pipeline_results
        entropies = [sm.attn_entropy for sm in metrics.step_metrics]
        assert any(e > 0 for e in entropies), "Expected positive attention entropy"
