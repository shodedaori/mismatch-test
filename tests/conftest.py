"""Shared pytest fixtures for mismatch-test."""

import pytest
import torch

MODEL_NAME = "Qwen/Qwen3-0.6B"

# Custom marker for tests that need a downloaded model
model = pytest.mark.model


@pytest.fixture(scope="session")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def model_name():
    return MODEL_NAME
