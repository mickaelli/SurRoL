#!/usr/bin/env python
"""SurRoL-VLA · RLAIF Zero-Shot Generalization Evaluator.

Loads a trained PPO policy (from Baseline, Dense Human, or RLAIF) 
and evaluates it on a potentially unseen task (Zero-Shot Transfer).

Usage:
  python vlm/eval/eval_rlaif_zeroshot.py \\
      --model-path logs/experiment/rlaif/ppo_rlaif_final.zip \\
      --eval-task GauzeRetrieve-v0 \\
      --episodes 50 \\
      --save-video
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np

from stable_baselines3 import PPO

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from surrol.wrappers import SurrolWrapper, TASK_NAME_TO_ID
import imageio.v2 as iio

def main():
    parser = argparse.ArgumentParser(description="Evaluate RLAIF PPO Zero-Shot Transfer")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained PPO .zip file")
    parser.add_argument("--eval-task", type=str, required=True,
                        help="Target SurRoL gym id to evaluate on (e.g., GauzeRetrieve-v0)")
    parser.add_argument("--obs-mode", type=str, default="state",
                        help="Must match the obs-mode used during training")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save-video", action="store_true",
                        help="Record a video of the first 3 evaluation episodes")
    parser.add_argument("--out-dir", type=str, default="logs/zeroshot_eval")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Zero-Shot Transfer Evaluation")
    print(f"  Model: {args.model_path}")
    print(f"  Task:  {args.eval_task}")
    print(f"{'='*60}")

    # 1. Load Model
    print("Loading PPO Policy...")
    model = PPO.load(args.model_path)

    # 2. Create Evaluation Environment
    # We use snake_case for SurrolWrapper task
    task_snake = args.eval_task.replace("-v0", "").lower()
    
    # Simple dictionary to fix special cases if any, map Camelcase to snake_case
    env = SurrolWrapper(
        task=task_snake if task_snake in TASK_NAME_TO_ID else args.eval_task,
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        max_episode_steps=args.max_steps,
    )

    success_count = 0
    total_rewards = []
    
    video_frames = []
    episodes_to_record = 3 if args.save_video else 0

    # 3. Evaluate loop
    for ep in range(args.episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs, _ = obs

        ep_reward = 0.0
        success = False
        record_this_ep = ep < episodes_to_record

        for step in range(args.max_steps):
            action, _states = model.predict(obs, deterministic=True)
            
            # Record frame BEFORE step to see initial state
            if record_this_ep:
                frame = env.render(mode="rgb_array")
                video_frames.append(frame)

            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                obs, reward, done, info = step_out

            ep_reward += float(reward)
            if info.get("is_success", False):
                success = True

            if done:
                break

        # Record final frame
        if record_this_ep:
            frame = env.render(mode="rgb_array")
            video_frames.append(frame)

        if success:
            success_count += 1
        total_rewards.append(ep_reward)

        print(f"  Episode {ep+1:02d} | Reward: {ep_reward:7.2f} | Success: {success}")

    # 4. Save video
    if args.save_video and video_frames:
        model_name = Path(args.model_path).parent.name
        vid_path = os.path.join(args.out_dir, f"zeroshot_{model_name}_on_{args.eval_task}.mp4")
        print(f"\nSaving video to {vid_path}...")
        iio.mimsave(vid_path, video_frames, fps=15)

    # 5. Summary
    success_rate = success_count / args.episodes
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)

    print(f"\n{'='*60}")
    print(f"  Evaluation Results: {args.eval_task}")
    print(f"  Success Rate: {success_rate * 100:.1f}% ({success_count}/{args.episodes})")
    print(f"  Mean Reward:  {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
