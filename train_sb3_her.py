"""HER + TD3 training entrypoint for SurRoL goal-based tasks.

Example:
    python train_sb3_her.py --task peg_transfer --total-timesteps 100000
"""

import argparse
import os
from typing import Callable, Dict, Optional

import gym
from gym import spaces
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.her import HerReplayBuffer

from surrol.wrappers import SurrolWrapper, TASK_NAME_TO_ID


def make_env(task: str, obs_mode: str, seed: int, max_episode_steps: Optional[int]) -> Callable[[], gym.Env]:
    def _init():
        env = SurrolWrapper(task=task, obs_mode=obs_mode, max_episode_steps=max_episode_steps)
        env = Monitor(env)
        env.seed(seed)
        return env

    return _init


def build_vec_env(args) -> DummyVecEnv:
    env_fns = [make_env(args.task, args.obs_mode, args.seed + i, args.max_episode_steps) for i in range(args.n_envs)]
    return DummyVecEnv(env_fns)


def choose_policy(obs_space: spaces.Space) -> str:
    if isinstance(obs_space, spaces.Dict):
        return "MultiInputPolicy"
    return "MlpPolicy"


def main():
    parser = argparse.ArgumentParser(description="Train SurRoL tasks with HER + TD3 (Stable-Baselines3)")
    parser.add_argument("--task", choices=sorted(TASK_NAME_TO_ID.keys()), default="peg_transfer")
    parser.add_argument("--obs-mode", choices=["state", "both"], default="state", help="HER expects goal dict")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=50)
    parser.add_argument("--n-sampled-goals", type=int, default=4, help="HER: goals sampled per transition")
    args = parser.parse_args()

    set_random_seed(args.seed)

    log_dir = args.log_dir or os.path.join("logs", "sb3_her", f"{args.task}-{args.obs_mode}")
    os.makedirs(log_dir, exist_ok=True)

    train_env = build_vec_env(args)
    policy = choose_policy(train_env.observation_space)

    replay_buffer_kwargs: Dict[str, object] = dict(
        n_sampled_goal=args.n_sampled_goals,
        goal_selection_strategy="future",
        online_sampling=True,
        max_episode_length=args.max_episode_steps,
    )

    model = TD3(
        policy,
        train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=replay_buffer_kwargs,
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
    model.save(os.path.join(log_dir, "td3_her_surrol_final"))

    train_env.close()
    if callback is not None:
        callback.eval_env.close()


if __name__ == "__main__":
    main()
