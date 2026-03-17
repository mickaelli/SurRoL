#!/usr/bin/env python
"""SurRoL-VLA · Qwen2-VL inference script.

Usage
-----
Single image:
    python vlm/trainer/infer_vla.py \\
        --model Qwen/Qwen2-VL-2B-Instruct \\
        --lora-path vlm/out/qwen2vl_vla_lora/lora_weights \\
        --image vlm/dataset/expert_active_track/frames/ep0000/t0000.jpg \\
        --instruction "Keep the red cube centered in the camera view."

Batch mode (directory of images):
    python vlm/trainer/infer_vla.py \\
        --model Qwen/Qwen2-VL-2B-Instruct \\
        --lora-path vlm/out/qwen2vl_vla_lora/lora_weights \\
        --image-dir vlm/dataset/expert_active_track/frames/ep0000 \\
        --instruction "Keep the red cube centered in the camera view."
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from vlm.model.qwen_vl_vla import load_model_for_inference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("infer_vla")


def parse_action_from_text(text: str) -> Optional[List[float]]:
    """Extract action list from model-generated text.

    Tries several patterns:
      1. JSON with "action" key:  {"action": [0.1, -0.2, 0.3]}
      2. Plain JSON list:         [0.1, -0.2, 0.3]
      3. Comma-separated numbers: 0.1, -0.2, 0.3
    """
    # Try JSON parse first
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, dict) and "action" in parsed:
            return [float(x) for x in parsed["action"]]
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Try regex for JSON-like action
    match = re.search(r'"action"\s*:\s*\[([^\]]+)\]', text)
    if match:
        try:
            return [float(x.strip()) for x in match.group(1).split(",")]
        except ValueError:
            pass

    # Try bare list
    match = re.search(r'\[([^\]]+)\]', text)
    if match:
        try:
            return [float(x.strip()) for x in match.group(1).split(",")]
        except ValueError:
            pass

    # Try comma-separated numbers without brackets
    match = re.search(
        r"(-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?){2,})",
        text,
    )
    if match:
        try:
            return [float(x.strip()) for x in match.group(1).split(",")]
        except ValueError:
            pass

    return None


def predict_action(
    model,
    processor,
    image: Image.Image,
    instruction: str,
    max_new_tokens: int = 128,
) -> dict:
    """Run one forward pass and return the predicted action.

    Returns
    -------
    dict with keys:
      - raw_text: str  (full generated text)
      - action: list[float] | None  (parsed action vector, or None if parse failed)
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for deterministic actions
        )

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    output_ids = generated_ids[:, input_len:]
    raw_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    action = parse_action_from_text(raw_text)

    return {"raw_text": raw_text, "action": action}


def main():
    parser = argparse.ArgumentParser(description="VLA inference with Qwen2-VL + LoRA")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Base model name or path")
    parser.add_argument("--lora-path", type=str, required=True, help="Path to LoRA adapter weights")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory of images for batch inference")
    parser.add_argument(
        "--instruction",
        type=str,
        default="请根据当前手术台环境，输出机械臂下一步的动作参数。",
        help="Text instruction",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    if args.image is None and args.image_dir is None:
        parser.error("Provide either --image or --image-dir")

    torch_dtype = getattr(torch, args.dtype, torch.bfloat16)

    # ── Load model ──
    model, processor = load_model_for_inference(
        base_name_or_path=args.model,
        lora_path=args.lora_path,
        torch_dtype=torch_dtype,
    )

    # ── Collect images ──
    image_paths: list[Path] = []
    if args.image:
        image_paths.append(Path(args.image))
    if args.image_dir:
        img_dir = Path(args.image_dir)
        image_paths.extend(sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")))

    if not image_paths:
        logger.error("No images found!")
        return

    logger.info("Running inference on %d image(s)...", len(image_paths))

    results = []
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        result = predict_action(
            model,
            processor,
            image,
            args.instruction,
            max_new_tokens=args.max_new_tokens,
        )
        result["image"] = str(img_path)
        results.append(result)

        action_str = json.dumps(result["action"]) if result["action"] else "PARSE_FAILED"
        logger.info("  %s  →  action=%s", img_path.name, action_str)
        if result["action"] is None:
            logger.warning("    raw output: %s", result["raw_text"])

    # ── Summary ──
    print("\n" + "=" * 60)
    print("Inference Results")
    print("=" * 60)
    for r in results:
        print(f"  {Path(r['image']).name}: {r['action']}")
    print("=" * 60)

    # Save results to JSON
    out_path = Path(args.lora_path).parent / "inference_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
