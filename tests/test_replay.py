"""Integration tests for the BF16 replay module.

Requires model download — marked with @pytest.mark.model.
Run with:  pytest tests/test_replay.py -v -m model
"""

import pytest
import torch

from tests.conftest import model as requires_model


@requires_model
class TestQwen3BF16Replayer:

    @pytest.fixture(scope="class")
    def replayer(self, model_name, device):
        from align.Qwen3.bf16_replay import Qwen3BF16Replayer
        rp = Qwen3BF16Replayer(model_name=model_name, device=device)
        yield rp
        rp.cleanup()

    @pytest.fixture(scope="class")
    def replay_result(self, replayer):
        prompt = "Hello, how are you?"
        enc = replayer.tokenizer(prompt, return_tensors="pt")
        # Fake some generated token ids (just use common tokens)
        gen_ids = torch.tensor([[29892, 358, 1097, 1532]])  # ", I am fine"
        return replayer.replay(
            prompt_id=0,
            prompt_ids=enc["input_ids"],
            generated_ids=gen_ids,
        )

    def test_result_shape(self, replay_result):
        assert replay_result.prompt_id == 0
        assert replay_result.num_generated == 4

    def test_logprobs_length(self, replay_result):
        assert len(replay_result.logprobs) == replay_result.num_generated

    def test_logits_shape(self, replay_result):
        assert replay_result.logits.ndim == 2
        assert replay_result.logits.shape[0] == replay_result.num_generated

    def test_hidden_states_captured(self, replay_result):
        assert len(replay_result.hidden_states) > 0
        for layer, h in replay_result.hidden_states.items():
            total_len = replay_result.prompt_length + replay_result.num_generated
            assert h.shape[1] == total_len, (
                f"Layer {layer}: expected seq_len={total_len}, got {h.shape[1]}"
            )

    def test_attention_probs_captured(self, replay_result):
        assert len(replay_result.attention_probs) > 0
        for layer, a in replay_result.attention_probs.items():
            # [1, H, S, S]
            assert a.ndim == 4
            assert a.shape[2] == a.shape[3]  # square attention matrix

    def test_attention_probs_sum_to_one(self, replay_result):
        for layer, a in replay_result.attention_probs.items():
            # Each row of the attention matrix should sum to ~1
            row_sums = a.sum(dim=-1)  # [1, H, S]
            assert torch.allclose(
                row_sums,
                torch.ones_like(row_sums),
                atol=1e-3,
            ), f"Layer {layer}: attn probs rows don't sum to 1"
