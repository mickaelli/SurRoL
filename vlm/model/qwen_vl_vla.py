"""SurRoL-VLA · Qwen2-VL model utilities with LoRA/QLoRA support.

Provides helpers to:
  1. Load the base Qwen2-VL model + processor
  2. Apply LoRA adapters via PEFT
  3. Save / load LoRA weights
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
)

logger = logging.getLogger(__name__)


# ── public API ───────────────────────────────────────────────


def load_model_and_processor(
    cfg: Dict[str, Any],
) -> tuple[Qwen2VLForConditionalGeneration, AutoProcessor]:
    """Load base Qwen2-VL model and its processor from config dict.

    Parameters
    ----------
    cfg : dict
        Parsed YAML config (top-level keys: model, lora, data, training).

    Returns
    -------
    model, processor
    """
    model_cfg = cfg["model"]
    name_or_path: str = model_cfg["name_or_path"]
    dtype_str: str = model_cfg.get("torch_dtype", "bfloat16")
    torch_dtype = getattr(torch, dtype_str, torch.bfloat16)

    # ── quantization (QLoRA) ──
    quant_cfg = model_cfg.get("quantization", {})
    quantization_config = None
    if quant_cfg.get("enabled", False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=(quant_cfg.get("bits", 4) == 4),
            load_in_8bit=(quant_cfg.get("bits", 4) == 8),
            bnb_4bit_quant_type=quant_cfg.get("quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant_cfg.get("double_quant", True),
            bnb_4bit_compute_dtype=torch_dtype,
        )
        logger.info("QLoRA enabled: %d-bit %s quantization", quant_cfg["bits"], quant_cfg["quant_type"])

    # ── load model ──
    logger.info("Loading model: %s  dtype=%s", name_or_path, dtype_str)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        name_or_path,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    #attn_implementation="flash_attention_2",

    # Prepare for k-bit training when quantization is on
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg.get("training", {}).get("gradient_checkpointing", True),
        )
    if hasattr(model,"enable_input_require_grads"):
        model.enable_input_require_grads()

    # ── processor ──
    data_cfg = cfg.get("data", {})
    processor = AutoProcessor.from_pretrained(
        name_or_path,
        trust_remote_code=True,
        min_pixels=data_cfg.get("min_pixels", 256),
        max_pixels=data_cfg.get("max_pixels", 512000),
    )

    return model, processor


def apply_lora(
    model: Qwen2VLForConditionalGeneration,
    cfg: Dict[str, Any],
) -> Qwen2VLForConditionalGeneration:
    """Inject LoRA adapters into the model.

    Parameters
    ----------
    model : Qwen2VLForConditionalGeneration
        The base (or quantized) model.
    cfg : dict
        Parsed YAML config.

    Returns
    -------
    PeftModel wrapping the original model.
    """
    lora_cfg = cfg["lora"]
    task_type_str = lora_cfg.get("task_type", "CAUSAL_LM")
    task_type = getattr(TaskType, task_type_str, TaskType.CAUSAL_LM)

    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        bias=lora_cfg.get("bias", "none"),
        task_type=task_type,
    )

    model = get_peft_model(model, peft_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        "LoRA applied ▸ trainable=%s (%.2f%%)  total=%s",
        f"{trainable:,}",
        100 * trainable / total,
        f"{total:,}",
    )
    model.print_trainable_parameters()
    return model


def save_lora_weights(model: PeftModel, output_dir: str | Path) -> None:
    """Save only the LoRA adapter weights."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    logger.info("LoRA weights saved to %s", output_dir)


def load_model_for_inference(
    base_name_or_path: str,
    lora_path: str | Path,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
) -> tuple[Qwen2VLForConditionalGeneration, AutoProcessor]:
    """Load base model + merge LoRA weights for inference.

    Parameters
    ----------
    base_name_or_path : str
        HuggingFace model ID or local path.
    lora_path : str | Path
        Path to saved LoRA adapter.
    torch_dtype : torch.dtype
        Compute dtype.
    device_map : str
        Device placement strategy.

    Returns
    -------
    model, processor
    """
    logger.info("Loading base model: %s", base_name_or_path)
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    logger.info("Loading LoRA weights from: %s", lora_path)
    model = PeftModel.from_pretrained(base_model, str(lora_path))
    model = model.merge_and_unload()  # merge for faster inference
    model.eval()

    processor = AutoProcessor.from_pretrained(
        base_name_or_path,
        trust_remote_code=True,
    )

    return model, processor
