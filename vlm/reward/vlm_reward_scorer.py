"""SurRoL-VLA · VLM-based Reward Scorer (RLAIF).

Uses a multimodal LLM (Qwen2-VL) to score how close the current surgical
scene is to a goal state. The score (0-10) is used as a reward signal for
reinforcement learning, implementing RLAIF (Reinforcement Learning from AI
Feedback).

Design
------
The scorer is intentionally **decoupled from the gym env** so it can be:
  - Used as a standalone tool for offline scoring
  - Plugged into the VLMRewardWrapper (vlm_reward_wrapper.py)
  - Called from any custom training loop

Scoring prompt template
-----------------------
The model receives:
  1. Current observation image
  2. Goal/reference image (optional — can be text-only goal description)
  3. A scoring prompt asking for a 0-10 integer

The prompt is configurable so you can experiment with different scoring
criteria (e.g. distance to target, grasp quality, trajectory smoothness).
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ── Default prompts ──────────────────────────────────────────

DEFAULT_SCORE_PROMPT_WITH_GOAL = (
    "你是一个对手术机器人操作极其严厉且挑剔的评分专家。\n"
    "图 1 是当前手术台画面，图 2 是最终成功的参考画面。\n"
    "你的任务是根据图 1 中【红色方块】相对于【画面中心】的位置精度打分：\n"
    "1. 必须优先观察红色方块是否在画面中。如果完全看不到红色方块，或方块在画面边缘，必须直接给 0 分。\n"
    "2. 只有当红色方块位于画面正中央（误差极小）且距离相机极近时，才给 9.0-10.0 分。\n"
    "3. 如果能看到方块但不在中心，根据偏离程度严格给出 1.0-5.0 分之间的分值。\n"
    "评分要求：请给出极其精确的小数分数（如 2.3 或 7.8），以提供细腻的反馈。\n"
    "请先简要描述红色方块在图 1 中的位置（中心/左/右/上/下/看不见），然后输出 JSON：{\"score\": <分数>}"
)

DEFAULT_SCORE_PROMPT_NO_GOAL = (
    "你是一个对手术机器人操作极其严厉且挑剔的评分专家。\n"
    "这张图是当前手术台的实时画面。任务目标是：{task_description}\n"
    "请评估当前画面中任务的完成程度：\n"
    "1. 如果目标物体（红色方块）不在画面中或在边缘，直接给 0 分。\n"
    "2. 如果目标处于画面中心且已对准，给 9.0-10.0 分。\n"
    "3. 中间状态请给出极其精确的小数分值（如 4.5），不要只给整数。\n"
    "请先描述物体位置，然后输出 JSON：{\"score\": <分数>}"
)


# ── Score parser ─────────────────────────────────────────────


def parse_score(text: str) -> Optional[float]:
    """Extract a numeric score from model output.

    Tries:
      1. Full JSON: {"score": N}
      2. JSON snippet containing "score"
      3. Labeled pattern: score: N
      4. Fallback to any number in [0, 10] (prefers the last occurrence)
    Returns float in [0, 10] or None if parsing fails.
    """
    s = text.strip()

    def _clamp(val: object) -> Optional[float]:
        try:
            score = float(val)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(10.0, score))

    # 1) Full JSON
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict) and "score" in parsed:
            return _clamp(parsed["score"])
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # 2) JSON snippet(s) containing score
    for m in reversed(list(re.finditer(r"\{[^{}]*?(?:\"score\"|score)[^{}]*?\}", s, flags=re.DOTALL))):
        try:
            parsed = json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
        if isinstance(parsed, dict) and "score" in parsed:
            clamped = _clamp(parsed["score"])
            if clamped is not None:
                return clamped

    # 3) Labeled pattern: score: N / "score": N
    labeled = re.findall(r"(?:\"score\"|score)\s*[:=]\s*(-?\d+(?:\.\d+)?)", s)
    if labeled:
        clamped = _clamp(labeled[-1])
        if clamped is not None:
            return clamped

    # 4) Fallback: any number in [0, 10] (prefer last)
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    for raw in reversed(nums):
        clamped = _clamp(raw)
        if clamped is None:
            continue
        if 0.0 <= clamped <= 10.0:
            return clamped

    return None


# ── VLM Reward Scorer ────────────────────────────────────────


class VLMRewardScorer:
    """Scores observations using Qwen2-VL as a visual reward model.

    Parameters
    ----------
    model : PreTrainedModel
        Qwen2-VL model (can be base or LoRA-merged).
    processor : AutoProcessor
        Corresponding Qwen2-VL processor.
    goal_image : PIL.Image | np.ndarray | None
        Optional goal/reference image. If provided, two-image scoring
        is used (current + goal). Otherwise, text-only goal description.
    task_description : str
        Text description of the task goal (used when goal_image is None).
    score_prompt : str | None
        Custom scoring prompt. If None, uses the appropriate default.
    max_new_tokens : int
        Max generation tokens.
    score_range : tuple[float, float]
        Output range to normalise the 0-10 score to. Default (0, 1).
    cache_ttl : float
        Minimum seconds between consecutive VLM calls (throttling).
        Set to 0 to disable.
    """

    def __init__(
        self,
        model,
        processor,
        goal_image: Optional[Union[Image.Image, np.ndarray]] = None,
        task_description: str = "将视野中心移动到红色方块",
        score_prompt: Optional[str] = None,
        max_new_tokens: int = 64,
        score_range: tuple = (-1.0, 1.0),
        cache_ttl: float = 0.0,
    ) -> None:
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.score_range = score_range
        self.cache_ttl = cache_ttl

        # Goal image
        if goal_image is not None:
            if isinstance(goal_image, np.ndarray):
                goal_image = Image.fromarray(goal_image)
            self.goal_image: Optional[Image.Image] = goal_image.convert("RGB")
        else:
            self.goal_image = None

        # Prompt
        if score_prompt is not None:
            self.score_prompt = score_prompt
        elif self.goal_image is not None:
            self.score_prompt = DEFAULT_SCORE_PROMPT_WITH_GOAL
        else:
            self.score_prompt = DEFAULT_SCORE_PROMPT_NO_GOAL.format(
                task_description=task_description
            )

        # Throttle tracking
        self._last_call_time: float = 0.0
        self._last_score: float = (score_range[0] + score_range[1]) / 2

        # Stats
        self.total_calls: int = 0
        self.total_parse_fails: int = 0

    def score(
        self,
        current_image: Union[Image.Image, np.ndarray],
    ) -> float:
        """Score the current observation image.

        Parameters
        ----------
        current_image : PIL.Image or numpy array
            Current rendered frame from the environment.

        Returns
        -------
        float : Normalised reward in self.score_range.
        """
        # Throttle
        now = time.time()
        if self.cache_ttl > 0 and (now - self._last_call_time) < self.cache_ttl:
            return self._last_score

        if isinstance(current_image, np.ndarray):
            current_image = Image.fromarray(current_image)
        current_image = current_image.convert("RGB")

        # Build messages
        if self.goal_image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": current_image},
                        {"type": "image", "image": self.goal_image},
                        {"type": "text", "text": self.score_prompt},
                    ],
                }
            ]
            images_flat = [current_image, self.goal_image]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": current_image},
                        {"type": "text", "text": self.score_prompt},
                    ],
                }
            ]
            images_flat = [current_image]

        # Tokenise
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text_prompt],
            images=images_flat,
            return_tensors="pt",
            padding=True,
        )
        import torch
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, input_len:]
        raw_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        # Parse score
        raw_score = parse_score(raw_text)
        self.total_calls += 1

        if raw_score is None:
            self.total_parse_fails += 1
            logger.warning("Score parse failed: %s → using cached %.2f", raw_text[:80], self._last_score)
            return self._last_score

        # Normalise from [0, 10] to self.score_range
        lo, hi = self.score_range
        normalised = lo + (raw_score / 10.0) * (hi - lo)

        self._last_score = normalised
        self._last_call_time = now

        logger.debug("VLM score: raw=%.1f  normalised=%.3f", raw_score, normalised)
        return normalised

    def get_stats(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_parse_fails": self.total_parse_fails,
            "parse_fail_rate": (
                self.total_parse_fails / self.total_calls
                if self.total_calls > 0
                else 0.0
            ),
        }

# ── Remote VLM Reward Scorer ───────────────────────────────────

class RemoteVLMRewardScorer:
    """Drop-in replacement for VLMRewardScorer that targets the FastAPI server.
    
    This avoids loading the heavy Qwen2-VL model and its dependencies 
    (transformers, flash_attn) in the RL environment, solving dependency conflicts.
    """
    
    def __init__(
        self,
        url: str,
        goal_image: Optional[Union[Image.Image, np.ndarray]] = None,
        task_description: str = "将红色方框移动到视野中心",
        score_prompt: Optional[str] = None,
        score_range: tuple = (-1.0, 1.0),
        cache_ttl: float = 0.0,
    ) -> None:
        self.url = url
        self.task_description = task_description
        self.score_prompt = score_prompt
        self.score_range = score_range
        self.cache_ttl = cache_ttl

        if goal_image is not None:
            if isinstance(goal_image, np.ndarray):
                goal_image = Image.fromarray(goal_image)
            self.goal_image = goal_image.convert("RGB")
        else:
            self.goal_image = None

        self._last_call_time = 0.0
        self._last_score = (score_range[0] + score_range[1]) / 2.0
        self.total_calls = 0
        self.total_parse_fails = 0

    def _img_to_b64(self, img: Image.Image) -> str:
        import io, base64
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def score(self, current_image: Union[Image.Image, np.ndarray]) -> float:
        import time, requests
        now = time.time()
        if self.cache_ttl > 0 and (now - self._last_call_time) < self.cache_ttl:
            return self._last_score

        if isinstance(current_image, np.ndarray):
            current_image = Image.fromarray(current_image)
        current_image = current_image.convert("RGB")

        payload = {
            "current_image": self._img_to_b64(current_image),
            "task_description": self.task_description,
            "score_range": self.score_range,
        }
        if self.score_prompt is not None:
            payload["score_prompt"] = self.score_prompt
        if self.goal_image is not None:
            payload["goal_image"] = self._img_to_b64(self.goal_image)

        self.total_calls += 1
        try:
            resp = requests.post(self.url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            # Note: the normalized score lies in [score_range[0], score_range[1]] default which is fine
            normalized = data.get("score", self._last_score)
            
            if "stats" in data:
                server_fails = data["stats"].get("total_parse_fails", 0)
                # Keep local count updated if needed
                
            self._last_score = normalized
            self._last_call_time = now
            logger.debug("VLM score: normalised=%.3f", normalized)
            logger.debug("VLM score (remote): %.3f", normalized)
            return normalized
        except Exception as e:
            logger.error("API call to remote VLM server failed: %s", e)
            self.total_parse_fails += 1
            return self._last_score

    def get_stats(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_parse_fails": self.total_parse_fails,
            "parse_fail_rate": (
                self.total_parse_fails / self.total_calls
                if self.total_calls > 0
                else 0.0
            ),
        }
