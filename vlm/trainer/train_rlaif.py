#!/usr/bin/env python
"""SurRoL-VLA · RLAIF Training Script.

Train an RL agent (SB3 PPO) using VLM-generated rewards from Qwen2-VL.

Flow
----
  1. Load Qwen2-VL (base or with LoRA) as a reward model
  2. Optionally capture a "goal image" from a successful oracle rollout
  3. Wrap the SurRoL env with VLMRewardWrapper
  4. Train PPO using the VLM scores as reward signal

Usage
-----
    python vlm/trainer/train_rlaif.py \\
        --vlm-model Qwen/Qwen2-VL-2B-Instruct \\
        --lora-path vlm/out/qwen2vl_vla_lora/lora_weights \\
        --task static_track \\
        --reward-mode replace \\
        --score-every 5 \\
        --total-timesteps 50000

    # With a goal image:
    python vlm/trainer/train_rlaif.py \\
        --vlm-model Qwen/Qwen2-VL-2B-Instruct \\
        --goal-image vlm/dataset/expert_static_track/frames/ep0000/t0018.jpg \\
        --task static_track \\
        --reward-mode add --vlm-weight 0.5

# 直接用 Qwen2-VL 原始模型做 reward（无需微调）
python vlm/trainer/train_rlaif.py \
    --vlm-model Qwen/Qwen2-VL-2B-Instruct \
    --task static_track --reward-mode replace --score-every 5

# 或用微调后的 LoRA 模型做 reward（更精准）
python vlm/trainer/train_rlaif.py \
    --vlm-model Qwen/Qwen2-VL-2B-Instruct \
    --lora-path vlm/out/qwen2vl_vla_lora/lora_weights \
    --task static_track --reward-mode add --vlm-weight 0.5

"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import gym
import numpy as np
from PIL import Image

# Removed global PyTorch and SB3 imports because the client environment might not have them installed yet.
# These will be imported inside main() or their respective specific files.

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from surrol.wrappers import SurrolWrapper, TASK_NAME_TO_ID
# Moved load_model_for_inference inside if-block to avoid transformers import on client
from vlm.reward.vlm_reward_scorer import VLMRewardScorer

from vlm.reward.vlm_reward_wrapper import VLMRewardWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_rlaif")


# ── Goal image capture ────────────────────────────────────────


def capture_goal_image(
    task: str,
    seed: int = 42,
    oracle_steps: int = 200,
    downsample: int = 2,
) -> np.ndarray:
    """Run the oracle policy and capture the frame with highest reward
    as the 'goal image' for VLM scoring.

    Returns
    -------
    np.ndarray : RGB image (downsampled).
    """
    logger.info("Capturing goal image via oracle rollout on '%s'...", task)
    env_id = TASK_NAME_TO_ID[task]
    env = gym.make(env_id, render_mode="rgb_array")
    env.seed(seed)

    obs = env.reset()
    if isinstance(obs, tuple):
        obs, _ = obs

    best_reward = -float("inf")
    best_frame: Optional[np.ndarray] = None

    for t in range(oracle_steps):
        action = env.get_oracle_action(obs)
        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            obs, reward, done, info = step_out

        if reward > best_reward:
            best_reward = reward
            frame = env.render(mode="rgb_array")
            if downsample > 1:
                frame = frame[::downsample, ::downsample]
            best_frame = frame

        if done:
            break

    env.close()
    logger.info("Goal image captured at reward=%.3f", best_reward)
    return best_frame


# ── Callbacks ───────────────────────────────────────────────


try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    class BaseCallback:
        def __init__(self, verbose=0):
            self.verbose = verbose
            self.num_timesteps = 0
            self.n_calls = 0

class VLMScoreLogCallback(BaseCallback):
    """SB3 callback that logs VLM scorer stats periodically."""

    def __init__(self, wrapper: VLMRewardWrapper, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.wrapper = wrapper
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            stats = self.wrapper.get_scorer_stats()
            logger.info(
                "VLM scorer stats @ step %d: calls=%d  parse_fails=%d (%.1f%%)",
                self.num_timesteps,
                stats["total_calls"],
                stats["total_parse_fails"],
                stats["parse_fail_rate"] * 100,
            )
        return True


class ExperimentMetricsCallback(BaseCallback):
    """Logs true environment rewards and success rates to a CSV for fair comparison."""
    
    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = Path(log_dir) / "experiment_metrics.csv"
        self.ep_env_reward = 0.0
        self.ep_success = False
        
        # Initialize CSV
        with self.log_path.open("w", encoding="utf-8") as f:
            f.write("step,ep_env_reward,success\n")

    def _on_step(self) -> bool:
        info_list = self.locals.get("infos", [])
        if not info_list:
            return True
            
        info = info_list[0]
        # Track episode metrics
        self.ep_env_reward += info.get("env_reward", 0.0)
        
        if info.get("is_success", False):
            self.ep_success = True
            
        # Log when episode finishes
        dones = self.locals.get("dones", [False])
        if dones[0]:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(f"{self.num_timesteps},{self.ep_env_reward},{int(self.ep_success)}\n")
            # Reset counters
            self.ep_env_reward = 0.0
            self.ep_success = False
            
        return True


class SafeFlattenWrapper(gym.ObservationWrapper):
    """A robust flattening wrapper for older Gym versions.
    
    Correctly handles 'reset()' returning (obs, info) tuples and avoids 
    TypeErrors when flattening Dict spaces.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env = env
        from gym import spaces
        
        # Cache check to avoid import/version mismatches during steps
        self.is_dict = isinstance(env.observation_space, spaces.Dict)

        if self.is_dict:
            try:
                from gym.spaces.utils import flatten_space, flatten
                self._flatten = flatten
            except ImportError:
                # Fallback for older gym versions
                def flatten_space(space):
                    dim = sum(np.prod(s.shape) for s in space.spaces.values())
                    return spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)
                
                def flatten(space, x):
                    return np.concatenate([np.asarray(x[key]).flatten() for key in space.spaces.keys()])
                self._flatten = flatten

            self.observation_space = flatten_space(env.observation_space)

    def observation(self, observation):
        if self.is_dict:
            return self._flatten(self.env.observation_space, observation).astype(np.float32)
        return observation

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            return self.observation(obs)
        return self.observation(result)

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            return self.observation(obs), reward, done, info
        else:
            obs, reward, done, info = result
            return self.observation(obs), reward, done, info


# ── Main ──────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="RLAIF training with VLM reward")

    # VLM model
    parser.add_argument("--vlm-model", type=str, default="Qwen/Qwen2-VL-2B-Instruct",
                        help="VLM base model name or path")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="Optional LoRA adapter path for the VLM")
    parser.add_argument("--vlm-dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--vlm-url", type=str, default=None,
                        help="Optional URL for remote VLM FastAPI server. Setting this runs the script in client mode.")

    # Goal image
    parser.add_argument("--goal-image", type=str, default=None,
                        help="Path to a goal image. If not provided, one is captured via oracle.")
    parser.add_argument("--task-description", type=str,
                        default="将目标物体移动到指定位置",
                        help="Text goal description (used when no goal image)")
    parser.add_argument("--no-goal-image", action="store_true",
                        help="Skip goal image and use text-only prompt")

    # Reward config
    parser.add_argument("--reward-mode", type=str, default="replace",
                        choices=["replace", "add", "multiply", "none", "dense_human"])
    parser.add_argument("--vlm-weight", type=float, default=1.0,
                        help="Weight for VLM score in 'add' mode")
    parser.add_argument("--score-every", type=int, default=5,
                        help="Only call VLM every N steps (optimization)")
    parser.add_argument("--record-video", action="store_true",
                        help="Record training progress episodes as MP4 videos")
    parser.add_argument("--score-prompt", type=str, default=None,
                        help="Custom scoring prompt. Uses default if not set.")

    # Environment
    parser.add_argument("--task", type=str, default="active_track",
                        choices=sorted(TASK_NAME_TO_ID.keys()))
    parser.add_argument("--obs-mode", type=str, default="state",
                        choices=["state", "rgb", "both", "rgb+state"])
    parser.add_argument("--max-episode-steps", type=int, default=None)

    # RL training
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    # Output
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--downsample", type=int, default=2)

    args = parser.parse_args()

    # Local Stable Baselines 3 imports inside main
    from stable_baselines3.common.utils import set_random_seed
    set_random_seed(args.seed)

    log_dir = args.log_dir or os.path.join("logs", "rlaif", f"{args.task}-{args.reward_mode}")
    os.makedirs(log_dir, exist_ok=True)

    # ── 1. Load VLM for scoring ──
    if args.vlm_url is None:
        import torch
        logger.info("=== Loading VLM reward model ===")
        torch_dtype = getattr(torch, args.vlm_dtype, torch.bfloat16)

        if args.lora_path:
            from vlm.model.qwen_vl_vla import load_model_for_inference
            model, processor = load_model_for_inference(
                base_name_or_path=args.vlm_model,
                lora_path=args.lora_path,
                torch_dtype=torch_dtype,
            )
        else:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                args.vlm_model, torch_dtype=torch_dtype,
                device_map="auto", trust_remote_code=True,
            )
            model.eval()
            processor = AutoProcessor.from_pretrained(
                args.vlm_model, trust_remote_code=True,
            )
    else:
        logger.info("=== Running in client mode with remote VLM @ %s ===", args.vlm_url)
        model = None
        processor = None

    # ── 2. Goal image ──
    goal_image = None
    if args.goal_image:
        goal_image = Image.open(args.goal_image).convert("RGB")
        logger.info("Using provided goal image: %s", args.goal_image)
    elif not args.no_goal_image:
        goal_np = capture_goal_image(
            args.task, seed=args.seed, downsample=args.downsample,
        )
        if goal_np is not None:
            goal_image = Image.fromarray(goal_np)

    # ── 3. Create scorer ──
    if args.vlm_url is None:
        scorer = VLMRewardScorer(
            model=model,
            processor=processor,
            goal_image=goal_image,
            task_description=args.task_description,
            score_prompt=args.score_prompt,
            score_range=(-1.0, 1.0),
        )
    else:
        from vlm.reward.vlm_reward_scorer import RemoteVLMRewardScorer
        scorer = RemoteVLMRewardScorer(
            url=args.vlm_url,
            goal_image=goal_image,
            task_description=args.task_description,
            score_prompt=args.score_prompt,
            score_range=(-1.0, 1.0),
        )

    # ── 4. Create wrapped environment ──
    logger.info("=== Creating environment: %s ===", args.task)
    base_env = SurrolWrapper(
        task=args.task,
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        max_episode_steps=args.max_episode_steps,
    )
    wrapped_env = VLMRewardWrapper(
        env=base_env,
        scorer=scorer,
        reward_mode=args.reward_mode,
        vlm_weight=args.vlm_weight,
        score_every=args.score_every,
        downsample=args.downsample,
    )
    from stable_baselines3.common.monitor import Monitor
    # Monitor records episode rewards/lengths to progress.csv
    monitor_path = os.path.join(log_dir, "monitor.csv")
    wrapped_env = Monitor(wrapped_env, filename=monitor_path)

    # Flatten Dict observation spaces for compatibility with older SB3 versions
    if isinstance(wrapped_env.observation_space, gym.spaces.Dict):
        logger.info("Flattening Dict observation space for SB3 compatibility using SafeFlattenWrapper")
        wrapped_env = SafeFlattenWrapper(wrapped_env)

    # Wrap in DummyVecEnv
    from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
    vec_env = DummyVecEnv([lambda: wrapped_env])

    if args.record_video:
        video_folder = os.path.join(log_dir, "videos")
        os.makedirs(video_folder, exist_ok=True)
        # Record one episode every 10,000 steps
        record_freq = 10000
        vec_env = VecVideoRecorder(vec_env, video_folder,
                                   record_video_trigger=lambda step: step % record_freq == 0,
                                   video_length=args.max_episode_steps)

    # ── 5. Train PPO ──
    logger.info("=== Starting RLAIF PPO training ===")
    logger.info("  reward_mode=%s  score_every=%d  vlm_weight=%.2f",
                args.reward_mode, args.score_every, args.vlm_weight)

    # Determine Policy
    # FlattenObservation converts Dict to Box, so we use MlpPolicy/CnnPolicy
    policy = "MlpPolicy" if args.obs_mode == "state" else "CnnPolicy"

    from stable_baselines3 import PPO

    # Tensorboard disabled due to version conflicts in simulation environment
    tb_log = None
    # try:
    #     import tensorboard
    # except ImportError:
    #     logger.warning("Tensorboard not found. Disabling tensorboard logging.")
    #     tb_log = None

    ppo_model = PPO(
        policy,
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        verbose=1,
        tensorboard_log=None,
        device=args.device,
    )

    # Configure SB3 Logger to save to CSV since Tensorboard is disabled
    from stable_baselines3.common.logger import configure
    logger.info("Configuring SB3 logger to save metrics to %s/progress.csv", log_dir)
    sb3_logger = configure(log_dir, ["stdout", "csv"])
    ppo_model.set_logger(sb3_logger)

    vlm_callback = VLMScoreLogCallback(wrapped_env, log_freq=500)
    exp_callback = ExperimentMetricsCallback(log_dir=log_dir)
    
    from stable_baselines3.common.callbacks import CallbackList
    ppo_model.learn(total_timesteps=args.total_timesteps, 
                    callback=CallbackList([vlm_callback, exp_callback]))

    # ── 6. Save ──
    save_path = os.path.join(log_dir, "ppo_rlaif_final")
    ppo_model.save(save_path)
    logger.info("Model saved to %s", save_path)

    # Log final VLM stats
    stats = scorer.get_stats()
    logger.info("=== Training complete! ===")
    logger.info("  VLM calls: %d  parse_fails: %d (%.1f%%)",
                stats["total_calls"], stats["total_parse_fails"],
                stats["parse_fail_rate"] * 100)

    vec_env.close()


if __name__ == "__main__":
    main()
