"""Qwen3 FP8 Attention Rollout Simulator.

Monkey-patches every Qwen3Attention module to apply FP8 precision on Q/K/V
(after RoPE), then performs autoregressive decoding with full per-step /
per-layer state capture.

Two FP8 backends are supported, selected via :class:`~utils.fp8_ops.Fp8Backend`:

* ``Fp8Backend.SIMULATE`` *(default)* – Software simulation using
  :func:`~utils.fp8_ops.simulate_fp8`.  Works on **any** GPU.

* ``Fp8Backend.FLASHINFER`` – Uses the FlashInfer FP8 kernel
  (``flashinfer.single_prefill_with_kv_cache``).
  Requires GPU Compute Capability ≥ 8.9 (Ada / Hopper) and the
  ``flashinfer`` package to be installed.

Both backends align with production RL rollout behaviour:
* Q/K/V are quantised to FP8 E4M3 (per-tensor max-abs scaling).
* Attention scores and output are computed / accumulated in BF16/FP32
  (matching FlashInfer's kernel semantics).
* All other components (MLP, norms, embeddings, LM head) stay BF16.
"""

from __future__ import annotations

import importlib
import types
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.fp8_ops import (
    Fp8Backend,
    flashinfer_fp8_attention,
    simulate_fp8,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RolloutResult:
    """Captures from a single prompt rollout."""

    prompt_id: int
    input_ids: List[int]        # prompt token ids
    generated_ids: List[int]    # sampled token ids
    logprobs: List[float]       # logprob of each sampled token

    # step -> layer_idx -> Tensor
    hidden_states: Dict[int, Dict[int, torch.Tensor]] = field(default_factory=dict)
    # step -> layer_idx -> Tensor  [num_heads, 1, seq_len_so_far]
    attention_probs: Dict[int, Dict[int, torch.Tensor]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class Qwen3RolloutSimulator:
    """FP8-attention rollout simulator for Qwen3 models.

    Only attention Q/K/V run in FP8; all other components (MLP, norms,
    embeddings, LM head) stay BF16, matching production RL rollout behaviour.

    Args:
        model_name: HuggingFace model identifier.
        device: ``"auto"`` picks CUDA if available, else CPU.
        seed: RNG seed for reproducibility.
        fp8_backend: Which FP8 backend to use.

            * :attr:`~utils.fp8_ops.Fp8Backend.SIMULATE` *(default)* –
              Software simulation, any GPU.
            * :attr:`~utils.fp8_ops.Fp8Backend.FLASHINFER` –
              FlashInfer real FP8 kernel, requires CC ≥ 8.9 + flashinfer.

    Example::

        from utils.fp8_ops import Fp8Backend

        # Software simulation (default, any GPU)
        sim = Qwen3RolloutSimulator("Qwen/Qwen3-0.6B")

        # FlashInfer real FP8 kernel (H100/L40/RTX-4090)
        sim = Qwen3RolloutSimulator(
            "Qwen/Qwen3-0.6B",
            fp8_backend=Fp8Backend.FLASHINFER,
        )
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: str = "auto",
        seed: int = 42,
        fp8_backend: Fp8Backend = Fp8Backend.SIMULATE,
    ):
        self.seed = seed
        self.fp8_backend = fp8_backend
        self.device = self._resolve_device(device)
        torch.manual_seed(seed)

        # Load tokenizer & model -----------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to(self.device).eval()

        # Import model-specific helpers -----------------------------------
        self._import_helpers()

        # Patch attention for FP8 ----------------------------------------
        self._patch_attention()

        # Hidden-state hooks ---------------------------------------------
        self._capture_enabled = False
        self._captured_hidden: Dict[int, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._setup_hidden_hooks()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _import_helpers(self):
        """Dynamically import apply_rotary_pos_emb / repeat_kv from
        the same module that defines the loaded attention class."""
        attn = self.model.model.layers[0].self_attn
        mod = importlib.import_module(type(attn).__module__)
        self._apply_rotary_pos_emb: Callable = mod.apply_rotary_pos_emb
        self._repeat_kv: Callable = mod.repeat_kv

    # ---------- FP8 attention patching ---------------------------------

    def _patch_attention(self):
        """Replace every Qwen3Attention.forward with an FP8 variant."""
        apply_rope = self._apply_rotary_pos_emb
        repeat_kv = self._repeat_kv
        fp8_backend = self.fp8_backend

        if fp8_backend is Fp8Backend.SIMULATE:
            self._patch_attention_simulate(apply_rope, repeat_kv)
        elif fp8_backend is Fp8Backend.FLASHINFER:
            self._patch_attention_flashinfer(apply_rope)
        else:
            raise ValueError(f"Unknown FP8 backend: {fp8_backend!r}")

    # ---- Backend A: SIMULATE ----

    def _patch_attention_simulate(self, apply_rope, repeat_kv):
        """Patch with simulate_fp8 on Q/K/V only (score & output in BF16/FP32).

        Alignment with FlashInfer semantics:
        - Q/K/V quantised to FP8 E4M3 (per-tensor max-abs)
        - Attention score (QKᵀ) computed in BF16, softmax in FP32
        - Attention output (softmax × V) in BF16
        """
        def fp8_forward_simulate(
            self_attn,
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values=None,
            **kwargs,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self_attn.head_dim)

            # QKV proj + QK-norm
            q = self_attn.q_norm(
                self_attn.q_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            k = self_attn.k_norm(
                self_attn.k_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            v = self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            # RoPE
            cos, sin = position_embeddings
            q, k = apply_rope(q, k, cos, sin)

            # ── FP8 simulation on Q / K / V (per-tensor max-abs) ──
            q = simulate_fp8(q)
            k = simulate_fp8(k)
            v = simulate_fp8(v)

            # KV cache
            if past_key_values is not None:
                k, v = past_key_values.update(k, v, self_attn.layer_idx)

            # Expand KV for GQA
            k_rep = repeat_kv(k, self_attn.num_key_value_groups)
            v_rep = repeat_kv(v, self_attn.num_key_value_groups)

            # Attention scores in BF16, softmax in FP32 (no FP8 on scores)
            attn_w = torch.matmul(q, k_rep.transpose(2, 3)) * self_attn.scaling

            if attention_mask is not None:
                attn_w = attn_w + attention_mask

            attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(q.dtype)

            # ── Capture attention probs ──
            self_attn._captured_attn_probs = attn_w.detach().cpu()

            # Weighted sum in BF16 (no FP8 on output)
            out = torch.matmul(attn_w, v_rep)

            out = out.transpose(1, 2).contiguous()
            out = out.reshape(*input_shape, -1).contiguous()
            out = self_attn.o_proj(out)

            return out, self_attn._captured_attn_probs

        for layer in self.model.model.layers:
            layer.self_attn.forward = types.MethodType(
                fp8_forward_simulate, layer.self_attn
            )

    # ---- Backend B: FLASHINFER ----

    def _patch_attention_flashinfer(self, apply_rope):
        """Patch with FlashInfer real FP8 kernel on Q/K/V.

        Requires flashinfer installed and GPU CC ≥ 8.9.
        KV cache is **not** stored in FP8 (BF16 cache, FP8 only in compute).
        Attention probs are not captured (FlashInfer kernel is fused).
        """
        # Validate import eagerly so failure is loud at init time
        from utils.fp8_ops import _check_flashinfer
        _check_flashinfer()

        def fp8_forward_flashinfer(
            self_attn,
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values=None,
            **kwargs,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self_attn.head_dim)

            # QKV proj + QK-norm
            q = self_attn.q_norm(
                self_attn.q_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)  # [B, H, S, D]
            k = self_attn.k_norm(
                self_attn.k_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            v = self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            # RoPE
            cos, sin = position_embeddings
            q, k = apply_rope(q, k, cos, sin)

            # KV cache (stored in BF16; only attention compute is FP8)
            if past_key_values is not None:
                k, v = past_key_values.update(k, v, self_attn.layer_idx)

            # FlashInfer expects [H, S, D] (batch=1 assumed)
            # GQA handled internally by FlashInfer via num_kv_heads
            q_s = q.squeeze(0)  # [H_q, S, D]
            k_s = k.squeeze(0)  # [H_kv, S, D]
            v_s = v.squeeze(0)

            causal = (attention_mask is None or
                      attention_mask.shape[-1] == attention_mask.shape[-2])

            out = flashinfer_fp8_attention(
                q_s, k_s, v_s,
                sm_scale=self_attn.scaling,
                causal=causal,
            )  # [H_q, S, D], BF16

            # FlashInfer doesn't expose attention probs (fused kernel)
            self_attn._captured_attn_probs = None

            out = out.unsqueeze(0)  # [1, H, S, D]
            out = out.transpose(1, 2).contiguous()
            out = out.reshape(*input_shape, -1).contiguous()
            out = self_attn.o_proj(out)

            return out, self_attn._captured_attn_probs

        for layer in self.model.model.layers:
            layer.self_attn.forward = types.MethodType(
                fp8_forward_flashinfer, layer.self_attn
            )

    # ---------- Hidden-state hooks ------------------------------------

    def _setup_hidden_hooks(self):
        for idx, layer in enumerate(self.model.model.layers):
            hook = layer.register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def fn(_module, _input, output):
            if self._capture_enabled:
                # Decoder layer returns a single tensor [B, S, D]
                self._captured_hidden[layer_idx] = output[:, -1, :].detach().cpu()
        return fn

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt_id: int,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
    ) -> RolloutResult:
        """Autoregressive decoding with FP8-patched attention.

        Args:
            prompt_id: Identifier for this prompt.
            input_ids: Tokenised prompt, shape ``[1, prompt_len]``.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature (0 → greedy).

        Returns:
            :class:`RolloutResult` with all captured data.
        """
        input_ids = input_ids.to(self.device)

        generated_ids: List[int] = []
        logprobs: List[float] = []
        all_hidden: Dict[int, Dict[int, torch.Tensor]] = {}
        all_attn: Dict[int, Dict[int, torch.Tensor]] = {}

        # ── Prefill (FP8 active, but capture disabled) ──
        self._capture_enabled = False
        outputs = self.model(input_ids=input_ids, use_cache=True)
        past_kv = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]

        # ── Autoregressive generation ──
        self._capture_enabled = True

        for step in range(max_new_tokens):
            # Sample
            if temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                token_id = torch.multinomial(probs, num_samples=1)  # [1, 1]
            else:
                token_id = next_logits.argmax(dim=-1, keepdim=True)  # [1, 1]

            log_p = F.log_softmax(next_logits, dim=-1)
            token_logprob = log_p.gather(-1, token_id).item()

            generated_ids.append(token_id.item())
            logprobs.append(token_logprob)

            # Forward new token
            self._captured_hidden = {}
            outputs = self.model(
                input_ids=token_id,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]

            # Collect hidden states
            all_hidden[step] = dict(self._captured_hidden)

            # Collect attention probs (None for FLASHINFER backend)
            step_attn: Dict[int, torch.Tensor] = {}
            for li, layer in enumerate(self.model.model.layers):
                probs_t = getattr(layer.self_attn, "_captured_attn_probs", None)
                if probs_t is not None:
                    step_attn[li] = probs_t.clone()
            all_attn[step] = step_attn

            # EOS check
            if (
                self.tokenizer.eos_token_id is not None
                and token_id.item() == self.tokenizer.eos_token_id
            ):
                break

        return RolloutResult(
            prompt_id=prompt_id,
            input_ids=input_ids[0].cpu().tolist(),
            generated_ids=generated_ids,
            logprobs=logprobs,
            hidden_states=all_hidden,
            attention_probs=all_attn,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Remove all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
