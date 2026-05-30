from __future__ import annotations

import argparse
import logging
from pathlib import Path

from finetune_template.src.data import create_data_collator, load_training_dataset
from finetune_template.src.model import load_model_with_lora, load_tokenizer
from finetune_template.src.trainer import create_sft_trainer
from finetune_template.src.utils import load_config, seed_everything, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning template")
    parser.add_argument(
        "--config",
        type=str,
        default="finetune_template/config/config.yaml",
        help="Path to config yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    setup_logging(cfg.get("log_level", "INFO"))
    seed_everything(cfg.get("seed", 42))

    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    data_cfg = cfg["data"]
    training_cfg = cfg["training"]

    logger.info("Loading tokenizer and base model...")
    tokenizer = load_tokenizer(
        model_name_or_path=model_cfg["name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    model = load_model_with_lora(model_cfg, lora_cfg)

    logger.info("Loading dataset from %s", data_cfg["train_file"])
    train_dataset = load_training_dataset(
        train_file=data_cfg["train_file"],
        text_field=data_cfg.get("text_field", "text"),
    )
    data_collator = create_data_collator(tokenizer)

    trainer = create_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=data_collator,
        data_cfg=data_cfg,
        training_cfg=training_cfg,
    )

    logger.info("Starting training...")
    trainer.train()

    output_dir = Path(training_cfg["output_dir"]) / "final"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving LoRA adapter to %s", output_dir)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
