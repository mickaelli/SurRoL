"""Modern SB3 training entrypoint for SurRoL tasks.

Example:
    python train_sb3.py --task peg_transfer --obs-mode rgb --total-timesteps 100000
"""

import argparse
import os
from typing import Callable, Dict, Optional

import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper

from surrol.wrappers import SurrolWrapper, TASK_NAME_TO_ID


class VecTransposeImageDict(VecEnvWrapper):
    """Transpose image channel for Dict observations with an `image` key."""

    def __init__(self, venv, image_key: str = "image"):
        assert isinstance(venv.observation_space, spaces.Dict), "VecTransposeImageDict requires Dict observation space"
        assert image_key in venv.observation_space.spaces, "Missing image key in observation space"
        self.image_key = image_key
        space = venv.observation_space.spaces[image_key]
        assert len(space.shape) == 3 and space.shape[2] in (1, 3, 4), "Image space must be channel-last"
        transposed = spaces.Box(low=space.low.min(), high=space.high.max(),
                                 shape=(space.shape[2], space.shape[0], space.shape[1]), dtype=space.dtype)
        spaces_dict: Dict[str, spaces.Space] = dict(venv.observation_space.spaces)
        spaces_dict[image_key] = transposed
        super().__init__(venv, observation_space=spaces.Dict(spaces_dict))

    def reset(self):
        result = self.venv.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        obs[self.image_key] = obs[self.image_key].transpose(0, 3, 1, 2)
        return obs, info

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        obs[self.image_key] = obs[self.image_key].transpose(0, 3, 1, 2)
        return obs, rewards, dones, infos


def make_env(task: str, obs_mode: str, seed: int, max_episode_steps: Optional[int]) -> Callable[[], gym.Env]:
    def _init():
        env = SurrolWrapper(task=task, obs_mode=obs_mode, max_episode_steps=max_episode_steps)
        env = Monitor(env)
        env.seed(seed)
        return env

    return _init


def build_vec_env(args) -> DummyVecEnv:
    env_fns = [make_env(args.task, args.obs_mode, args.seed + i, args.max_episode_steps) for i in range(args.n_envs)]
    vec_env = DummyVecEnv(env_fns)
    obs_mode = "both" if args.obs_mode == "rgb+state" else args.obs_mode
    if obs_mode == "rgb":
        vec_env = VecTransposeImage(vec_env)
    elif obs_mode == "both":
        vec_env = VecTransposeImageDict(vec_env)
    return vec_env


def choose_policy(obs_space: spaces.Space, obs_mode: str) -> str:
    obs_mode = "both" if obs_mode == "rgb+state" else obs_mode
    if obs_mode == "rgb":
        return "CnnPolicy"
    if isinstance(obs_space, spaces.Dict):
        return "MultiInputPolicy"
    return "MlpPolicy"


PRESETS: Dict[str, Dict[str, Dict[str, float]]] = {
    "mid": {
        "ppo": {
            "state": {"learning_rate": 3e-4, "n_steps": 1024, "batch_size": 256},
            "rgb": {"learning_rate": 1e-4, "n_steps": 512, "batch_size": 128},
            "both": {"learning_rate": 2e-4, "n_steps": 512, "batch_size": 128},
            "rgb+state": {"learning_rate": 2e-4, "n_steps": 512, "batch_size": 128},
        }
    }
}


def apply_presets(args):
    preset = PRESETS.get(args.preset, {})
    per_algo = preset.get("ppo", {})
    per_mode = per_algo.get(args.obs_mode, {})
    for key, value in per_mode.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    # fallbacks if still unset
    if args.learning_rate is None:
        args.learning_rate = 3e-4
    if args.n_steps is None:
        args.n_steps = 1024
    if args.batch_size is None:
        args.batch_size = 256


def main():
    parser = argparse.ArgumentParser(description="Train SurRoL tasks with Stable-Baselines3 PPO")
    parser.add_argument("--task", choices=sorted(TASK_NAME_TO_ID.keys()), default="peg_transfer")
    parser.add_argument("--obs-mode", choices=["state", "rgb", "both", "rgb+state"], default="rgb")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None, help="Rollout horizon per environment")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--preset", choices=["none", "mid"], default="mid", help="Preset hyperparameters")
    args = parser.parse_args()

    set_random_seed(args.seed)

    apply_presets(args)

    log_dir = args.log_dir or os.path.join("logs", "sb3", f"{args.task}-{args.obs_mode}")
    os.makedirs(log_dir, exist_ok=True)

    train_env = build_vec_env(args)
    policy = choose_policy(train_env.observation_space, args.obs_mode)

    model = PPO(
        policy,
        train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        verbose=1,
        tensorboard_log=log_dir,
        device=args.device,
    )

    callback = None
    if args.eval_freq and args.eval_freq > 0:
        eval_env = build_vec_env(args)
        best_path = os.path.join(log_dir, "best_model")
        os.makedirs(best_path, exist_ok=True)
        callback = EvalCallback(
            eval_env,
            best_model_save_path=best_path,
            log_path=log_dir,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
        )

    model.learn(total_timesteps=args.total_timesteps, callback=callback)
    model.save(os.path.join(log_dir, "ppo_surrol_final"))

    train_env.close()
    if callback is not None:
        callback.eval_env.close()


if __name__ == "__main__":
    main()
