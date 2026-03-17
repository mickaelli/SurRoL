#!/usr/bin/env python
"""SurRoL-VLA · Qwen2-VL LoRA fine-tuning script.

Usage
-----
    python vlm/trainer/train_vla.py --config vlm/config/train_config.yaml

The script:
  1. Loads config from YAML
  2. Initialises Qwen2-VL with optional QLoRA quantisation
  3. Injects LoRA adapters
  4. Loads the ShareGPT-format dataset
  5. Fine-tunes with HuggingFace Trainer
  6. Saves LoRA adapter weights
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
import torch
from transformers import TrainingArguments, Trainer

# Make sure project root is on sys.path so local imports work
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from vlm.model.qwen_vl_vla import load_model_and_processor, apply_lora, save_lora_weights
from vlm.dataset.vla_dataset import VLADataset, VLACollator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_vla")


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str, resume_from: str | None = None) -> None:
    cfg = load_config(config_path)
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    # ── 1. Model + Processor ──
    logger.info("=== Loading model ===")
    model, processor = load_model_and_processor(cfg)

    # ── 2. LoRA ──
    logger.info("=== Applying LoRA ===")
    model = apply_lora(model, cfg)

    # ── 3. Dataset ──
    logger.info("=== Loading dataset ===")
    data_path = _PROJECT_ROOT / data_cfg["training_data"]
    image_root = _PROJECT_ROOT / data_cfg["image_root"]
    dataset = VLADataset(data_path=data_path, image_root=image_root)
    logger.info("Dataset size: %d samples", len(dataset))

    # ── 4. Collator ──
    collator = VLACollator(
        processor=processor,
        max_length=data_cfg.get("max_length", 1024),
    )

    # ── 5. Training arguments ──
    output_dir = _PROJECT_ROOT / train_cfg["output_dir"]
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.get("num_train_epochs", 10),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=train_cfg.get("logging_steps", 5),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        remove_unused_columns=False,
        seed=train_cfg.get("seed", 42),
        report_to="none",  # disable wandb etc. by default
        # Gradient checkpointing kwargs for Qwen2-VL compatibility
        gradient_checkpointing_kwargs={"use_reentrant": False}
        if train_cfg.get("gradient_checkpointing", True)
        else None,
    )

    # ── 6. Trainer ──
    logger.info("=== Starting training ===")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train(resume_from_checkpoint=resume_from)

    # ── 7. Save LoRA weights ──
    lora_out = output_dir / "lora_weights"
    save_lora_weights(model, lora_out)

    # Also save the processor for convenience
    processor.save_pretrained(str(lora_out))
    logger.info("=== Training complete! ===")
    logger.info("LoRA weights saved to: %s", lora_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2-VL for VLA")
    parser.add_argument(
        "--config",
        type=str,
        default="vlm/config/train_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint dir to resume from",
    )
    args = parser.parse_args()
    main(config_path=args.config, resume_from=args.resume)
