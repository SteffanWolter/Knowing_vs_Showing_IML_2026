from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class LoadedModel:
    model: object
    tokenizer: object
    model_name: str


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def load_causal_lm(
    model_name: str,
    *,
    device: str,
    load_in_4bit: bool,
) -> LoadedModel:
    """
    Load a Hugging Face causal LM with optional 4-bit quantization.

    This function intentionally uses `device_map="auto"` to support large models without manual placement.
    """
    tok = AutoTokenizer.from_pretrained(model_name, token=hf_token())
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = (
        torch.bfloat16 if torch.cuda.is_available() and device.startswith("cuda") else torch.float32
    )
    quantization_config = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token(),
        torch_dtype=dtype,
        device_map="auto",
        quantization_config=quantization_config,
    )

    return LoadedModel(model=model, tokenizer=tok, model_name=model_name)
