"""Unit tests for join_metrics — synthetic data, no model needed."""

import pytest
import torch

from join_metrics import (
    StepMetrics,
    SequenceMetrics,
    join_and_compute,
    _l2,
    _rel_l2,
    _cosine_distance,
    _entropy,
    _top1_prob,
    _tail_mass,
)


# ---------------------------------------------------------------------------
# Metric helper tests
# ---------------------------------------------------------------------------

class TestMetricHelpers:

    def test_l2_identical(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        assert _l2(a, a) == pytest.approx(0.0)

    def test_l2_known(self):
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 0.0])
        assert _l2(a, b) == pytest.approx(1.0)

    def test_rel_l2(self):
        a = torch.tensor([2.0, 0.0])
        b = torch.tensor([1.0, 0.0])
        assert _rel_l2(a, b) == pytest.approx(1.0)  # ||diff||/||b|| = 1/1

    def test_cosine_distance_identical(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        assert _cosine_distance(a, a) == pytest.approx(0.0, abs=1e-6)

    def test_cosine_distance_orthogonal(self):
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        assert _cosine_distance(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_entropy_uniform(self):
        """Uniform distribution → max entropy."""
        n = 8
        p = torch.ones(1, n) / n
        ent = _entropy(p)
        expected = -torch.log(torch.tensor(1.0 / n)).item()
        assert ent == pytest.approx(expected, rel=1e-4)

    def test_entropy_peaked(self):
        """Delta distribution → zero entropy."""
        p = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        assert _entropy(p) == pytest.approx(0.0, abs=1e-6)

    def test_top1_prob_uniform(self):
        p = torch.ones(2, 4) / 4
        assert _top1_prob(p) == pytest.approx(0.25)

    def test_tail_mass_peaked(self):
        p = torch.tensor([[1.0, 0.0, 0.0]])
        assert _tail_mass(p) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# join_and_compute with synthetic stubs
# ---------------------------------------------------------------------------

class _FakeRollout:
    def __init__(self):
        self.prompt_id = 0
        self.input_ids = [1, 2, 3]
        self.generated_ids = [10, 11]
        self.logprobs = [-1.0, -2.0]
        # step -> layer -> Tensor[D]
        self.hidden_states = {
            0: {0: torch.ones(1, 64)},
            1: {0: torch.ones(1, 64) * 2},
        }
        self.attention_probs = {}


class _FakeReplay:
    def __init__(self):
        self.prompt_id = 0
        self.prompt_length = 3
        self.num_generated = 2
        self.logprobs = [-1.5, -2.5]
        self.logits = torch.randn(2, 100)
        # layer -> Tensor [1, seq_len, D]
        self.hidden_states = {
            0: torch.ones(1, 5, 64) * 1.1,
        }
        # layer -> Tensor [1, H, S, S]
        self.attention_probs = {
            0: torch.ones(1, 4, 5, 5) / 5,
        }


class TestJoinAndCompute:

    def test_basic_join(self):
        rr = _FakeRollout()
        rp = _FakeReplay()
        m = join_and_compute(rr, rp)

        assert m.prompt_id == 0
        assert len(m.logprob_deltas) == 2
        # replay - rollout: (-1.5 - (-1.0)) = -0.5
        assert m.logprob_deltas[0] == pytest.approx(-0.5)
        assert m.logprob_deltas[1] == pytest.approx(-0.5)
        assert m.delta_T == pytest.approx(-1.0)
        assert m.abs_delta_T == pytest.approx(1.0)

    def test_step_metrics_populated(self):
        rr = _FakeRollout()
        rp = _FakeReplay()
        m = join_and_compute(rr, rp)

        # 2 steps × 1 layer = 2 StepMetrics
        assert len(m.step_metrics) == 2
        for sm in m.step_metrics:
            assert sm.hidden_l2 > 0  # rollout ≠ replay hidden
            assert sm.attn_entropy > 0  # uniform probs → positive entropy
