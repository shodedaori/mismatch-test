"""Join rollout and replay outputs and compute mismatch metrics.

Metrics follow the plan:
  - Hidden-state mismatch  (L2, relative L2, cosine distance)
  - Output mismatch        (logprob delta, logit L2, logit L-inf)
  - Attention sharpness    (entropy, top1 prob, tail mass)
  - Sequence accumulation  (Delta_T, abs_Delta_T)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# Per-step record
# ---------------------------------------------------------------------------

@dataclass
class StepMetrics:
    """Metrics for one (prompt, decode_step, layer)."""

    prompt_id: int
    step: int
    layer: int

    # Hidden-state mismatch
    hidden_l2: float = 0.0
    hidden_rel_l2: float = 0.0
    hidden_cosine_distance: float = 0.0

    # Attention sharpness (from replay BF16 probs)
    attn_entropy: float = 0.0
    top1_prob: float = 0.0
    tail_mass: float = 0.0


@dataclass
class SequenceMetrics:
    """Sequence-level metrics for one prompt."""

    prompt_id: int

    # Output mismatch per generated token
    logprob_deltas: List[float] = field(default_factory=list)

    # Accumulation
    delta_T: float = 0.0       # sum of logprob deltas
    abs_delta_T: float = 0.0   # sum of |logprob delta|

    # Per-step / per-layer detail
    step_metrics: List[StepMetrics] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).norm(p=2).item()


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.float() - b.float()).norm(p=2).item()
    ref = b.float().norm(p=2).item()
    return diff / max(ref, 1e-12)


def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    cos_sim = F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()
    return 1.0 - cos_sim


def _entropy(probs: torch.Tensor) -> float:
    """Attention entropy averaged over heads.

    Args:
        probs: [num_heads, seq_len] or [1, num_heads, 1, seq_len]
    """
    p = probs.float().flatten(end_dim=-2)  # [H, S]
    # Mask zeros to avoid log(0)
    log_p = torch.where(p > 0, p.log(), torch.zeros_like(p))
    ent = -(p * log_p).sum(dim=-1)  # [H]
    return ent.mean().item()


def _top1_prob(probs: torch.Tensor) -> float:
    p = probs.float().flatten(end_dim=-2)  # [H, S]
    return p.max(dim=-1).values.mean().item()


def _tail_mass(probs: torch.Tensor) -> float:
    p = probs.float().flatten(end_dim=-2)  # [H, S]
    top1 = p.max(dim=-1).values  # [H]
    return (1.0 - top1).mean().item()


# Need F for cosine_similarity
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Main join function
# ---------------------------------------------------------------------------

def join_and_compute(
    rollout_result,   # RolloutResult
    replay_result,    # ReplayResult
) -> SequenceMetrics:
    """Join rollout and replay by (prompt_id, step, layer) and compute metrics.

    Args:
        rollout_result: Output from ``Qwen3RolloutSimulator.generate``.
        replay_result: Output from ``Qwen3BF16Replayer.replay``.

    Returns:
        :class:`SequenceMetrics` with all computed metrics.
    """
    prompt_id = rollout_result.prompt_id
    prompt_len = replay_result.prompt_length
    num_steps = len(rollout_result.generated_ids)

    # ── Logprob deltas ──
    logprob_deltas: List[float] = []
    for t in range(num_steps):
        rollout_lp = rollout_result.logprobs[t]
        replay_lp = replay_result.logprobs[t] if t < len(replay_result.logprobs) else 0.0
        logprob_deltas.append(replay_lp - rollout_lp)

    delta_T = sum(logprob_deltas)
    abs_delta_T = sum(abs(d) for d in logprob_deltas)

    # ── Per-step per-layer metrics ──
    step_metrics_list: List[StepMetrics] = []

    # Determine available layers from rollout data
    all_layers = set()
    for step_dict in rollout_result.hidden_states.values():
        all_layers.update(step_dict.keys())

    for t in range(num_steps):
        for layer in sorted(all_layers):
            sm = StepMetrics(prompt_id=prompt_id, step=t, layer=layer)

            # --- Hidden-state mismatch ---
            rollout_h = rollout_result.hidden_states.get(t, {}).get(layer)
            replay_h_full = replay_result.hidden_states.get(layer)

            if rollout_h is not None and replay_h_full is not None:
                # replay hidden: extract position (prompt_len + t)
                pos = prompt_len + t
                if pos < replay_h_full.shape[1]:
                    replay_h = replay_h_full[:, pos, :]  # [1, D]
                    r_h = rollout_h.flatten()
                    p_h = replay_h.flatten()
                    sm.hidden_l2 = _l2(r_h, p_h)
                    sm.hidden_rel_l2 = _rel_l2(r_h, p_h)
                    sm.hidden_cosine_distance = _cosine_distance(r_h, p_h)

            # --- Attention sharpness (from replay probs) ---
            replay_attn_full = replay_result.attention_probs.get(layer)
            if replay_attn_full is not None:
                # replay attn: [1, H, S, S] → row for position (prompt_len + t)
                pos = prompt_len + t
                if pos < replay_attn_full.shape[-2]:
                    row = replay_attn_full[:, :, pos, : pos + 1]  # [1, H, 1, pos+1]
                    # Actually we want the full row up to pos+1
                    row = replay_attn_full[0, :, pos, : pos + 1]  # [H, pos+1]
                    sm.attn_entropy = _entropy(row)
                    sm.top1_prob = _top1_prob(row)
                    sm.tail_mass = _tail_mass(row)

            step_metrics_list.append(sm)

    return SequenceMetrics(
        prompt_id=prompt_id,
        logprob_deltas=logprob_deltas,
        delta_T=delta_T,
        abs_delta_T=abs_delta_T,
        step_metrics=step_metrics_list,
    )
