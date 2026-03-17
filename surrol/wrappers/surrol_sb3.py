import gym
import numpy as np
from gym import spaces
from gym.wrappers import TimeLimit
from typing import Dict, Optional, Tuple

from surrol.gym.surrol_env import RENDER_HEIGHT, RENDER_WIDTH

TASK_NAME_TO_ID: Dict[str, str] = {
    "needle_reach": "NeedleReach-v0",
    "gauze_retrieve": "GauzeRetrieve-v0",
    "needle_pick": "NeedlePick-v0",
    "peg_transfer": "PegTransfer-v0",
    "needle_regrasp": "NeedleRegrasp-v0",
    "bipeg_transfer": "BiPegTransfer-v0",
    "ecm_reach": "ECMReach-v0",
    "mis_orient": "MisOrient-v0",
    "static_track": "StaticTrack-v0",
    "active_track": "ActiveTrack-v0",
}

DEFAULT_MAX_STEPS: Dict[str, int] = {
    "active_track": 500,
}


class SurrolWrapper(gym.Wrapper):
    """Lightweight wrapper to feed SurRoL tasks into Stable-Baselines3.

    Supports state, rgb, or combined observations without touching legacy baselines.
    """

    def __init__(
        self,
        task: str,
        obs_mode: str = "state",
        render_mode: str = None,
        image_size: Optional[Tuple[int, int]] = None,
        max_episode_steps: Optional[int] = None,
    ):
        task_key = task.lower()
        if task_key not in TASK_NAME_TO_ID:
            raise ValueError(f"Unknown SurRoL task '{task}'. Supported: {sorted(TASK_NAME_TO_ID.keys())}")
        obs_mode_norm = obs_mode.lower()
        if obs_mode_norm == "rgb+state":
            obs_mode_norm = "both"
        if obs_mode_norm not in {"state", "rgb", "both"}:
            raise ValueError("obs_mode must be one of: 'state', 'rgb', 'rgb+state', 'both'")
        self.obs_mode = obs_mode_norm

        env_id = TASK_NAME_TO_ID[task_key]
        base_env = gym.make(env_id, render_mode=render_mode)
        base_env = self._ensure_time_limit(base_env, task_key, max_episode_steps)

        self._image_shape = self._resolve_image_shape(image_size)
        super().__init__(base_env)

        self.observation_space = self._build_observation_space(self.env.observation_space)

        # Gymnasium-compatible reset/step outputs

    def reset(self, **kwargs):
        seed = kwargs.pop("seed", None)
        if seed is not None:
            self.env.seed(seed)
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        return self._convert_obs(obs), info

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, done, info = result
            truncated = info.pop("TimeLimit.truncated", False) if isinstance(info, dict) else False
            terminated = done and not truncated
        obs = self._convert_obs(obs)
        return obs, reward, terminated, truncated, info

    def _convert_obs(self, base_obs):
        if self.obs_mode == "state":
            return base_obs

        image = self.env.render(mode="rgb_array")
        if self.obs_mode == "rgb":
            return image

        if isinstance(base_obs, dict):
            merged = {"image": image}
            merged.update(base_obs)
            return merged
        return {"image": image, "state": base_obs}

    def _build_observation_space(self, base_space: spaces.Space) -> spaces.Space:
        if self.obs_mode == "state":
            return base_space
        h, w = self._image_shape
        image_space = spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)
        if self.obs_mode == "rgb":
            return image_space
        if isinstance(base_space, spaces.Dict):
            merged = {"image": image_space}
            merged.update(base_space.spaces)
            return spaces.Dict(merged)
        return spaces.Dict({"image": image_space, "state": base_space})

    def _resolve_image_shape(self, image_size: Optional[Tuple[int, int]]) -> Tuple[int, int]:
        if image_size is None:
            return RENDER_HEIGHT, RENDER_WIDTH
        if len(image_size) != 2:
            raise ValueError("image_size must be (height, width)")
        return image_size

    @staticmethod
    def _ensure_time_limit(env: gym.Env, task_key: str, override_steps: Optional[int]) -> gym.Env:
        if isinstance(env, TimeLimit):
            if override_steps:
                env._max_episode_steps = override_steps  # type: ignore[attr-defined]
            return env
        max_steps = override_steps or DEFAULT_MAX_STEPS.get(task_key, 50)
        return TimeLimit(env, max_episode_steps=max_steps)
