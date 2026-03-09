"""Run Config A experiment: FP8-attention rollout → BF16 replay → metrics.

Usage:
    python run_experiment.py                    # use defaults from config.yaml
    python run_experiment.py --config my.yaml   # custom config
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import yaml

from utils.data import load_prompts
from rollout.Qwen3.rollout_simulator import Qwen3RolloutSimulator
from utils.fp8_ops import Fp8Backend
from align.Qwen3.bf16_replay import Qwen3BF16Replayer
from join_metrics import join_and_compute, SequenceMetrics


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(cfg: dict):
    model_name = cfg["model"]["name"]
    seed = cfg["generation"]["seed"]
    max_new_tokens = cfg["generation"]["max_new_tokens"]
    temperature = cfg["generation"]["temperature"]
    num_prompts = cfg["dataset"]["num_prompts"]
    max_prompt_len = cfg["dataset"]["max_prompt_length"]
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[config] model={model_name}  config={cfg['experiment']['config']}")
    print(f"[config] prompts={num_prompts}  max_new_tokens={max_new_tokens}")
    print()

    # ── 1. Rollout simulator ──
    print("Loading rollout simulator (FP8 attention) ...")
    rollout_sim = Qwen3RolloutSimulator(
        model_name=model_name, seed=seed, fp8_backend=Fp8Backend.FLASHINFER,
    )

    # ── 2. Load prompts ──
    print("Loading prompts ...")
    prompts = load_prompts(
        num_prompts=num_prompts,
        tokenizer=rollout_sim.tokenizer,
        max_length=max_prompt_len,
        seed=seed,
    )
    print(f"  Loaded {len(prompts)} prompts\n")

    # ── 3. Generate rollout trajectories ──
    rollout_results = []
    for p in prompts:
        pid = p["prompt_id"]
        print(f"  Rollout prompt {pid} ({p['input_ids'].shape[1]} tokens) ...")
        result = rollout_sim.generate(
            prompt_id=pid,
            input_ids=p["input_ids"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        rollout_results.append(result)
        print(f"    → generated {len(result.generated_ids)} tokens")

    rollout_sim.cleanup()

    # Free rollout model before loading replay model
    del rollout_sim
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── 4. BF16 replay ──
    print("\nLoading BF16 replayer ...")
    replayer = Qwen3BF16Replayer(model_name=model_name)

    replay_results = []
    for rr in rollout_results:
        pid = rr.prompt_id
        prompt_ids = torch.tensor([rr.input_ids])
        gen_ids = torch.tensor([rr.generated_ids])
        print(f"  Replay prompt {pid} ...")
        replay = replayer.replay(
            prompt_id=pid,
            prompt_ids=prompt_ids,
            generated_ids=gen_ids,
        )
        replay_results.append(replay)

    replayer.cleanup()
    del replayer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── 5. Join & compute metrics ──
    print("\nComputing metrics ...")
    all_metrics: list[SequenceMetrics] = []
    for rr, rp in zip(rollout_results, replay_results):
        m = join_and_compute(rr, rp)
        all_metrics.append(m)
        print(f"  Prompt {m.prompt_id}: Delta_T={m.delta_T:.4f}  |Delta_T|={m.abs_delta_T:.4f}")

    # ── 6. Save summary ──
    summary = []
    step_records = []
    for m in all_metrics:
        entry = {
            "prompt_id": m.prompt_id,
            "delta_T": m.delta_T,
            "abs_delta_T": m.abs_delta_T,
            "logprob_deltas": m.logprob_deltas,
            "num_step_metrics": len(m.step_metrics),
        }
        # Aggregate hidden mismatch stats
        if m.step_metrics:
            entry["mean_hidden_l2"] = sum(s.hidden_l2 for s in m.step_metrics) / len(m.step_metrics)
            entry["mean_attn_entropy"] = sum(s.attn_entropy for s in m.step_metrics) / len(m.step_metrics)
            entry["mean_cosine_dist"] = sum(s.hidden_cosine_distance for s in m.step_metrics) / len(m.step_metrics)

            for s in m.step_metrics:
                step_records.append(
                    {
                        "prompt_id": s.prompt_id,
                        "step": s.step,
                        "layer": s.layer,
                        "attn_entropy": s.attn_entropy,
                        "top1_prob": s.top1_prob,
                        "tail_mass": s.tail_mass,
                        "hidden_l2": s.hidden_l2,
                        "hidden_rel_l2": s.hidden_rel_l2,
                        "hidden_cosine_distance": s.hidden_cosine_distance,
                    }
                )
        summary.append(entry)

    out_path = out_dir / "results_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")

    details_path = out_dir / "step_metrics.json"
    with open(details_path, "w") as f:
        json.dump(step_records, f, indent=2)
    print(f"Step metrics saved to {details_path}")


def main():
    parser = argparse.ArgumentParser(description="Run mismatch-test experiment")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
