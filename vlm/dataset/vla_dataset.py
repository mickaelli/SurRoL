"""SurRoL-VLA · Qwen2-VL Dataset for VLA fine-tuning.

Reads ShareGPT-format JSON (produced by data_processing.py) and converts
each sample into tokenised model inputs suitable for Qwen2-VL SFT.

Key design choices
------------------
* Uses the Qwen2-VL chat template to build input_ids so the fine-tuned
  model retains instruction-following ability.
* Only the **assistant** tokens contribute to the loss (label masking).
* Images are loaded eagerly as PIL and passed through the processor's
  built-in dynamic-resolution handling.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class VLADataset(Dataset):
    """PyTorch dataset that yields Qwen2-VL-compatible training samples.

    Each item is a dict with keys expected by the Qwen2-VL processor:
      - messages: list[dict]  (chat-format messages with image references)
      - images:   list[PIL.Image]

    The actual tokenisation happens in the *collator* (see `VLACollator`)
    so that padding is handled at the batch level.
    """

    def __init__(
        self,
        data_path: str | Path,
        image_root: str | Path | None = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.image_root = Path(image_root) if image_root else self.data_path.parent

        with self.data_path.open("r", encoding="utf-8") as f:
            self.raw: List[Dict[str, Any]] = json.load(f)

        logger.info("Loaded %d samples from %s", len(self.raw), self.data_path)

    # ── Dataset interface ─────────────────────────────────────

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.raw[idx]
        conversations = item["conversations"]
        image_paths: List[str] = item.get("images", [])

        # ── load images ──
        images: List[Image.Image] = []
        for rel_path in image_paths:
            img_path = self.image_root / rel_path
            images.append(Image.open(img_path).convert("RGB"))

        # ── build Qwen2-VL message list ──
        # Qwen2-VL expects messages in this format:
        #   [{"role": "user", "content": [{"type":"image","image":...}, {"type":"text","text":...}]},
        #    {"role": "assistant", "content": [{"type":"text","text":...}]}]
        messages: list[dict] = []
        img_idx = 0
        for turn in conversations:
            role = turn["role"]
            raw_content: str = turn["content"]

            if role == "user":
                content_parts: list[dict] = []
                # Replace <image> placeholder(s) with image entries
                segments = raw_content.split("<image>")
                for i, seg in enumerate(segments):
                    if i > 0 and img_idx < len(images):
                        content_parts.append({"type": "image", "image": images[img_idx]})
                        img_idx += 1
                    if seg.strip():
                        content_parts.append({"type": "text", "text": seg.strip()})
                messages.append({"role": "user", "content": content_parts})
            else:
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": raw_content}],
                })

        return {
            "messages": messages,
            "images": images,
        }


class VLACollator:
    """Data collator that tokenises a batch of VLA samples via Qwen2-VL processor.

    For each sample the collator:
      1. Applies the chat template to get the full prompt (user + assistant).
      2. Tokenises with the processor (handles image resizing & token insertion).
      3. Builds labels where only assistant-reply tokens are unmasked.
      4. Pads the batch to the same length.
    """

    IGNORE_INDEX = -100

    def __init__(
        self,
        processor: Any,
        max_length: int = 1024,
    ) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts: list[str] = []
        all_images: list[list[Image.Image]] = []

        for sample in batch:
            messages = sample["messages"]
            # Apply chat template → full text with special tokens
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
            all_images.append(sample["images"])

        # Flatten images list for processor (it expects a flat list)
        flat_images: list[Image.Image] = []
        for imgs in all_images:
            flat_images.extend(imgs)

        # Tokenise the whole batch
        model_inputs = self.processor(
            text=texts,
            images=flat_images if flat_images else None,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # ── build labels with assistant-only masking ──
        input_ids = model_inputs["input_ids"]
        labels = input_ids.clone()

        # Mask padding tokens
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = self.IGNORE_INDEX

        # Mask everything before and including the assistant header per sample
        # Qwen2-VL chat template uses <|im_start|>assistant\n ... <|im_end|>
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        # Encode "assistant" as a token sequence for matching
        assistant_token_ids = self.tokenizer.encode("assistant", add_special_tokens=False)

        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()
            # Find the position right after "<|im_start|>assistant\n"
            mask_end = self._find_assistant_content_start(ids, im_start_id, assistant_token_ids)
            if mask_end > 0:
                labels[i, :mask_end] = self.IGNORE_INDEX

        model_inputs["labels"] = labels
        return model_inputs

    @staticmethod
    def _find_assistant_content_start(
        ids: list[int],
        im_start_id: int,
        assistant_token_ids: list[int],
    ) -> int:
        """Return the index of the first token of the assistant *content*.

        Scans for the pattern: <|im_start|> + assistant_tokens + newline.
        Returns the position right after the newline so everything before
        (system prompt, user message, assistant header) is masked out.
        """
        n = len(ids)
        ast_len = len(assistant_token_ids)
        for j in range(n):
            if ids[j] != im_start_id:
                continue
            # Check if followed by "assistant" tokens
            start = j + 1
            end = start + ast_len
            if end > n:
                continue
            if ids[start:end] == assistant_token_ids:
                # Skip past the newline token (Qwen2 template inserts \n after role)
                content_start = end
                # The template usually has a \n token right after "assistant"
                if content_start < n:
                    content_start += 1  # skip the \n
                return content_start
        # Fallback: don't mask anything (should not happen with valid data)
        return 0
