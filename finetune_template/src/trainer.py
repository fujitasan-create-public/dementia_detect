from __future__ import annotations

import inspect

from trl import SFTConfig, SFTTrainer


def build_sft_config(training_cfg: dict) -> SFTConfig:
    return SFTConfig(
        output_dir=training_cfg["output_dir"],
        num_train_epochs=training_cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
        learning_rate=training_cfg.get("learning_rate", 2e-4),
        weight_decay=training_cfg.get("weight_decay", 0.0),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.0),
        logging_steps=training_cfg.get("logging_steps", 10),
        save_steps=training_cfg.get("save_steps", 200),
        save_total_limit=training_cfg.get("save_total_limit", 2),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        fp16=training_cfg.get("fp16", False),
        bf16=training_cfg.get("bf16", True),
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", True),
        report_to=training_cfg.get("report_to", "none"),
    )


def create_sft_trainer(
    model,
    tokenizer,
    train_dataset,
    data_collator,
    data_cfg: dict,
    training_cfg: dict,
) -> SFTTrainer:
    sft_config = build_sft_config(training_cfg)
    kwargs = {
        "model": model,
        "args": sft_config,
        "train_dataset": train_dataset,
        "dataset_text_field": data_cfg.get("text_field", "text"),
        "max_seq_length": data_cfg.get("max_seq_length", 1024),
        "data_collator": data_collator,
    }

    signature = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in signature.parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in signature.parameters:
        kwargs["tokenizer"] = tokenizer
    if "packing" in signature.parameters:
        kwargs["packing"] = False

    return SFTTrainer(**kwargs)
