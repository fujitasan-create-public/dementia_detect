from __future__ import annotations

from datasets import Dataset, load_dataset
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase


def load_training_dataset(train_file: str, text_field: str) -> Dataset:
    dataset = load_dataset("json", data_files=train_file, split="train")
    if text_field not in dataset.column_names:
        cols = ", ".join(dataset.column_names)
        raise ValueError(f"Column '{text_field}' not found. Available columns: {cols}")
    return dataset


def create_data_collator(
    tokenizer: PreTrainedTokenizerBase,
) -> DataCollatorForLanguageModeling:
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
