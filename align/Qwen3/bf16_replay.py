"""Qwen3 BF16 Replay Module (alignment reference).

Loads Qwen3 in clean BF16 (no FP8), performs a single-pass prefill over
the full sequence [prompt; generated_tokens], and captures per-layer
attention probabilities, hidden states, logits, and logprobs.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ReplayResult:
    """Captures from a single-pass BF16 replay."""

    prompt_id: int
    prompt_length: int
    num_generated: int

    # Per generated-token logprob  (len = num_generated)
    logprobs: List[float] = field(default_factory=list)
    # Logits at generated positions  [num_generated, vocab_size]
    logits: Optional[torch.Tensor] = None

    # layer_idx -> Tensor [1, seq_len, hidden_dim]  (full sequence)
    hidden_states: Dict[int, torch.Tensor] = field(default_factory=dict)
    # layer_idx -> Tensor [1, num_heads, seq_len, seq_len]
    attention_probs: Dict[int, torch.Tensor] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Replayer
# ---------------------------------------------------------------------------

class Qwen3BF16Replayer:
    """BF16 reference replayer for Qwen3 models.

    Runs the full sequence in a single forward pass (like training-time
    prefill) so that every position's attention probs and hidden states
    are produced in one shot.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: str = "auto",
    ):
        self.device = self._resolve_device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",  # needed to get attention weights
        ).to(self.device).eval()

        # Capture containers
        self._captured_hidden: Dict[int, torch.Tensor] = {}
        self._captured_attn: Dict[int, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

        self._setup_hooks()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _setup_hooks(self):
        """Register forward hooks on decoder layers and attention modules."""
        for idx, layer in enumerate(self.model.model.layers):
            # Hidden states (decoder-layer output)
            h = layer.register_forward_hook(self._make_hidden_hook(idx))
            self._hooks.append(h)
            # Attention probs (attention-module output)
            h = layer.self_attn.register_forward_hook(self._make_attn_hook(idx))
            self._hooks.append(h)

    def _make_hidden_hook(self, layer_idx: int):
        def fn(_mod, _inp, output):
            # Qwen3DecoderLayer returns a single hidden_states tensor
            self._captured_hidden[layer_idx] = output.detach().cpu()
        return fn

    def _make_attn_hook(self, layer_idx: int):
        def fn(_mod, _inp, output):
            # Qwen3Attention returns (attn_output, attn_weights)
            _, attn_weights = output
            if attn_weights is not None:
                self._captured_attn[layer_idx] = attn_weights.detach().cpu()
        return fn

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    @torch.no_grad()
    def replay(
        self,
        prompt_id: int,
        prompt_ids: torch.Tensor,
        generated_ids: torch.Tensor,
    ) -> ReplayResult:
        """Single-pass BF16 prefill replay.

        Args:
            prompt_id: Identifier for this prompt.
            prompt_ids: Tokenised prompt, shape ``[1, prompt_len]``.
            generated_ids: Generated tokens, shape ``[1, gen_len]``.

        Returns:
            :class:`ReplayResult` with all captured data.
        """
        prompt_len = prompt_ids.shape[1]
        gen_len = generated_ids.shape[1]

        # Build full sequence
        full_ids = torch.cat([prompt_ids, generated_ids], dim=1).to(self.device)  # [1, S]

        # Reset capture containers
        self._captured_hidden = {}
        self._captured_attn = {}

        # Single forward (no KV cache, full causal mask)
        outputs = self.model(input_ids=full_ids, use_cache=False)

        # ── Extract logits at generated positions ──
        # logits[i] predicts token at position i+1
        # Generated tokens occupy positions [prompt_len, ..., prompt_len+gen_len-1]
        # So predicting logits are at positions [prompt_len-1, ..., prompt_len+gen_len-2]
        gen_logits = outputs.logits[:, prompt_len - 1 : prompt_len + gen_len - 1, :]  # [1, gen_len, V]

        # Compute per-token logprobs
        gen_log_probs = F.log_softmax(gen_logits, dim=-1)  # [1, gen_len, V]
        token_logprobs = gen_log_probs.gather(
            -1, generated_ids.to(self.device).unsqueeze(-1)
        ).squeeze(-1)  # [1, gen_len]

        return ReplayResult(
            prompt_id=prompt_id,
            prompt_length=prompt_len,
            num_generated=gen_len,
            logprobs=token_logprobs[0].cpu().tolist(),
            logits=gen_logits[0].cpu(),  # [gen_len, V]
            hidden_states=dict(self._captured_hidden),
            attention_probs=dict(self._captured_attn),
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Remove all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
