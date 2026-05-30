# template-pytorch

LoRA を使った PyTorch fine-tuning テンプレートです。  
`uv` で依存関係を管理し、CUDA 版 PyTorch を使う前提です。

## Structure

```text
finetune_template/
├── config/
│   └── config.yaml
├── src/
│   ├── data.py
│   ├── model.py
│   ├── trainer.py
│   └── utils.py
└── train.py
```

## Setup

```bash
uv sync
```

## Run

1. `finetune_template/config/config.yaml` を調整
2. 学習データ (`jsonl`) を `train_file` に指定
3. 実行

```bash
uv run train-lora --config finetune_template/config/config.yaml
```


