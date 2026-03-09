"""Integration tests for the FP8 rollout simulator.

Requires model download — marked with @pytest.mark.model.
Run with:  pytest tests/test_rollout.py -v -m model
"""

import pytest
import torch

from tests.conftest import model as requires_model


@requires_model
class TestQwen3RolloutSimulator:

    @pytest.fixture(scope="class")
    def simulator(self, model_name, device):
        from rollout.Qwen3.rollout_simulator import Qwen3RolloutSimulator
        sim = Qwen3RolloutSimulator(model_name=model_name, device=device, seed=42)
        yield sim
        sim.cleanup()

    @pytest.fixture(scope="class")
    def rollout_result(self, simulator):
        prompt = "Hello, how are you?"
        enc = simulator.tokenizer(prompt, return_tensors="pt")
        return simulator.generate(
            prompt_id=0,
            input_ids=enc["input_ids"],
            max_new_tokens=4,
            temperature=1.0,
        )

    def test_generates_tokens(self, rollout_result):
        assert len(rollout_result.generated_ids) > 0
        assert len(rollout_result.generated_ids) <= 4

    def test_logprobs_match_length(self, rollout_result):
        assert len(rollout_result.logprobs) == len(rollout_result.generated_ids)

    def test_logprobs_are_negative(self, rollout_result):
        for lp in rollout_result.logprobs:
            assert lp <= 0.0, f"logprob should be <= 0, got {lp}"

    def test_hidden_states_captured(self, rollout_result):
        num_steps = len(rollout_result.generated_ids)
        assert len(rollout_result.hidden_states) == num_steps
        # Each step should have at least one layer captured
        for step, layers in rollout_result.hidden_states.items():
            assert len(layers) > 0, f"Step {step} has no hidden states"

    def test_attention_probs_captured(self, rollout_result):
        num_steps = len(rollout_result.generated_ids)
        assert len(rollout_result.attention_probs) == num_steps
        for step, layers in rollout_result.attention_probs.items():
            assert len(layers) > 0, f"Step {step} has no attention probs"

    def test_attention_probs_sum_to_one(self, rollout_result):
        """Attention probs along the key dimension should sum to ~1."""
        for step, layers in rollout_result.attention_probs.items():
            for layer_idx, probs in layers.items():
                # probs shape: [1, num_heads, 1, seq_len]
                row_sums = probs.sum(dim=-1)
                assert torch.allclose(
                    row_sums,
                    torch.ones_like(row_sums),
                    atol=1e-2,
                ), f"Step {step} layer {layer_idx}: attn probs don't sum to 1"
