"""SurRoL-VLA · Gym Reward Wrapper for RLAIF.

Wraps a SurRoL gym environment and replaces (or augments) the original
reward with a VLM-based score from the VLMRewardScorer.

Three reward modes
------------------
  "replace"   - Use only the VLM score as reward.
  "add"       - reward = env_reward + vlm_weight * vlm_score
  "multiply"  - reward = env_reward * (0.5 + vlm_score)

The wrapper calls the VLM scorer at a configurable frequency
(every *score_every* steps) to limit GPU overhead. Between calls
the last score is reused.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import gym
import numpy as np
from PIL import Image

from vlm.reward.vlm_reward_scorer import VLMRewardScorer

logger = logging.getLogger(__name__)


class VLMRewardWrapper(gym.Wrapper):
    """Gym wrapper that injects VLM-based rewards into a SurRoL env.

    Parameters
    ----------
    env : gym.Env
        The base SurRoL environment (must support render(mode='rgb_array')).
    scorer : VLMRewardScorer
        An initialised VLM reward scorer instance.
    reward_mode : str
        One of 'replace', 'add', 'multiply'.
    vlm_weight : float
        Weight factor for the VLM score when mode='add'.
    score_every : int
        Call the VLM scorer every N steps. Use 1 for every step (slow but
        most accurate), or higher values (e.g. 5-10) for efficiency.
    downsample : int
        Downsample factor for the rendered image before scoring.
    """

    def __init__(
        self,
        env: gym.Env,
        scorer: VLMRewardScorer,
        reward_mode: str = "replace",
        vlm_weight: float = 1.0,
        score_every: int = 1,
        downsample: int = 2,
    ) -> None:
        super().__init__(env)
        assert reward_mode in ("replace", "add", "multiply", "none", "dense_human"), \
            f"reward_mode must be 'replace', 'add', 'multiply', 'none', or 'dense_human', got '{reward_mode}'"
        self.scorer = scorer
        self.reward_mode = reward_mode
        self.vlm_weight = vlm_weight
        self.score_every = max(1, score_every)
        self.downsample = downsample

        self._step_count: int = 0
        self._last_vlm_score: float = (scorer.score_range[0] + scorer.score_range[1]) / 2.0

    def reset(self, **kwargs):
        self._step_count = 0
        result = self.env.reset(**kwargs)
        
        # 强制在第一步 (t=0) 就拿到真实图片的分数，避免前 `score_every` 步都在无目标地盲猜
        if self.reward_mode not in ("none", "dense_human"):
            image = self.env.render(mode="rgb_array")
            if self.downsample > 1:
                image = image[::self.downsample, ::self.downsample]
            self._last_vlm_score = self.scorer.score(image)
        else:
            self._last_vlm_score = (self.scorer.score_range[0] + self.scorer.score_range[1]) / 2.0
            
        return result

    def step(self, action):
        result = self.env.step(action)

        # Unpack (support both old gym 4-tuple and new 5-tuple)
        if len(result) == 5:
            obs, env_reward, terminated, truncated, info = result
        else:
            obs, env_reward, done, info = result
            terminated = done
            truncated = False

        self._step_count += 1

        # ── Bypass VLM entirely if mode is none or dense_human ──
        if self.reward_mode in ("none", "dense_human"):
            final_reward = env_reward
            if self.reward_mode == "dense_human":
                unwrapped = self.env.unwrapped
                if hasattr(unwrapped, "_get_obs") and hasattr(unwrapped, "goal"):
                    raw_obs = unwrapped._get_obs()
                    if isinstance(raw_obs, dict) and "achieved_goal" in raw_obs:
                        dist = np.linalg.norm(raw_obs["achieved_goal"] - unwrapped.goal)
                        # scale distance to a sensible dense reward, e.g. -distance * 10 
                        # so that perfectly reaching it gives 0, being 10cm away gives -1
                        final_reward = float(-dist * 10.0)

            if isinstance(info, dict):
                info["env_reward"] = env_reward
                info["combined_reward"] = final_reward
            if len(result) == 5:
                return obs, final_reward, terminated, truncated, info
            return obs, final_reward, terminated or truncated, info

        # ── Call VLM scorer at configured frequency ──
        if self._step_count % self.score_every == 0:
            image = self.env.render(mode="rgb_array")
            if self.downsample > 1:
                image = image[::self.downsample, ::self.downsample]
            self._last_vlm_score = self.scorer.score(image)

        vlm_score = self._last_vlm_score

        # ── Combine rewards ──
        if self.reward_mode == "replace":
            reward = vlm_score
        elif self.reward_mode == "add":
            reward = env_reward + self.vlm_weight * vlm_score
        elif self.reward_mode == "multiply":
            reward = env_reward * (0.5 + vlm_score)
        else:
            reward = env_reward

        # Attach VLM info for logging
        if isinstance(info, dict):
            info["vlm_score"] = vlm_score
            info["env_reward"] = env_reward
            info["combined_reward"] = reward

        if len(result) == 5:
            return obs, reward, terminated, truncated, info
        return obs, reward, terminated or truncated, info

    def get_scorer_stats(self) -> dict:
        return self.scorer.get_stats()
