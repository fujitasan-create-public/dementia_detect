from __future__ import annotations

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def _resolve_dtype(name: str) -> torch.dtype:
    key = name.lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {name}")
    return mapping[key]


def load_tokenizer(model_name_or_path: str, trust_remote_code: bool = True) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_with_lora(model_cfg: dict, lora_cfg: dict) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        torch_dtype=_resolve_dtype(model_cfg.get("torch_dtype", "bfloat16")),
        device_map="auto",
    )

    task_type = TaskType[lora_cfg.get("task_type", "CAUSAL_LM")]
    peft_cfg = LoraConfig(
        r=lora_cfg.get("r", 8),
        lora_alpha=lora_cfg.get("lora_alpha", 16),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type=task_type,
        target_modules=lora_cfg.get("target_modules"),
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model
