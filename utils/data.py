"""Data loading utilities for the mismatch-test pipeline.

Loads prompts from HuggingFaceH4/ultrachat_200k (train_sft split)
and tokenises them for use with the rollout / replay modules.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from transformers import PreTrainedTokenizerBase


def load_prompts(
    num_prompts: int,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
    seed: int = 42,
    dataset_name: str = "HuggingFaceH4/ultrachat_200k",
    split: str = "train_sft",
) -> List[Dict]:
    """Load and tokenise user prompts from the ultrachat dataset.

    Each returned dict contains:
        prompt_id  : int
        text       : str  (raw user message)
        input_ids  : Tensor [1, seq_len]
        attention_mask : Tensor [1, seq_len]

    Args:
        num_prompts: Number of prompts to load.
        tokenizer: HuggingFace tokenizer (must match the target model).
        max_length: Maximum token length for truncation.
        seed: Random seed for shuffling.
        dataset_name: HuggingFace dataset identifier.
        split: Dataset split to use.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    ds = ds.shuffle(seed=seed)

    prompts: List[str] = []
    for example in ds:
        if len(prompts) >= num_prompts:
            break
        messages = example.get("messages", [])
        if messages and messages[0].get("role") == "user":
            prompts.append(messages[0]["content"])

    results: List[Dict] = []
    for i, text in enumerate(prompts):
        encoding = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        results.append(
            {
                "prompt_id": i,
                "text": text,
                "input_ids": encoding["input_ids"],           # [1, seq_len]
                "attention_mask": encoding["attention_mask"],  # [1, seq_len]
            }
        )

    return results
