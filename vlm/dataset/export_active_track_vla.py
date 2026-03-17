import argparse
import json
from pathlib import Path
from typing import IO, Optional

import gym
import numpy as np
from surrol.gym.surrol_env import SurRoLEnv

_warned_imageio = False

# Minimal collector that rolls out the scripted oracle for ActiveTrack
# and dumps per-step samples (image, obs, action delta, instruction, reward, done)


def _to_serializable(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


def make_env(render: bool):
    render_mode = "rgb_array" if render else None
    env = gym.make("ActiveTrack-v0", render_mode=render_mode)
    return env


def _save_jpg(image: np.ndarray, path: Path):
    """Best-effort JPEG save for debug mode."""
    global _warned_imageio
    try:
        import imageio.v2 as iio
        # print(image.shape)
        iio.imwrite(path, image, format="jpeg")
    except Exception as exc:  # noqa: BLE001
        if not _warned_imageio:
            _warned_imageio = True
            print(f"[debug] Could not write JPEGs: {exc}")


def collect_episode(
    env,
    episode_id: int,
    max_steps: int,
    instruction: str,
    out_dir: Path,
    frames_dir: Path,
    manifest: IO[str],
):
    obs = env.reset()
    if isinstance(obs, tuple):
        obs, _ = obs
    samples = 0
    ep_frame_dir = frames_dir / f"ep{episode_id:04d}"
    ep_frame_dir.mkdir(parents=True, exist_ok=True)
    for t in range(max_steps):
        action = env.get_oracle_action(obs)
        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            obs, reward, done, info = step_out
        image = env.render(mode="rgb_array")
        image = image[::2, ::2]  # downsample by 2 to reduce resolution and size
        jpg_path = ep_frame_dir / f"t{t:04d}.jpg"
        _save_jpg(image, jpg_path)

        record = {
            "episode": episode_id,
            "t": t,
            "instruction": instruction,
            "image": str(jpg_path.relative_to(out_dir)),
            "obs": np.asarray(obs, dtype=np.float32).tolist(),
            "action": np.asarray(action, dtype=np.float32).tolist(),
            "reward": float(reward),
            "done": bool(done),
            "info": json.loads(json.dumps(info, default=_to_serializable)),
        }
        manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
        samples += 1
        if done:
            break
    return samples


def main(
    num_episodes: int,
    max_steps: int,
    output_dir: Path,
    seed: Optional[int],
    instruction: str,
    render: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    manifest = manifest_path.open("w", encoding="utf-8")

    env = make_env(render)
    if seed is not None:
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    total_samples = 0
    try:
        for ep in range(num_episodes):
            total_samples += collect_episode(
                env,
                ep,
                max_steps,
                instruction,
                output_dir,
                frames_dir,
                manifest,
            )
    finally:
        env.close()
        manifest.close()
    print(f"Wrote {total_samples} samples to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ActiveTrack expert rollouts for VLA finetuning")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to roll out")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--out", type=Path, default=Path("./vlm/dataset/expert_active_track"), help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--instruction",
        type=str,
        default="Keep the red cube centered in the camera view using the endoscope.Left image is the global view, right image is the endoscope view.",
        help="Text instruction stored with each sample",
    )
    parser.add_argument("--render", action="store_true", help="Render to rgb_array for saving frames")

    args = parser.parse_args()
    main(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        output_dir=args.out,
        seed=args.seed,
        instruction=args.instruction,
        render=args.render,
    )
