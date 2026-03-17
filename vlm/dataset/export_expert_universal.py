#!/usr/bin/env python
"""SurRoL-VLA · Universal expert trajectory exporter for ALL SurRoL tasks.

Supports both ECM tasks (ActiveTrack, StaticTrack, etc.) and PSM tasks
(NeedlePick, GauzeRetrieve, PegTransfer, etc.).

Key differences handled automatically
--------------------------------------
  ECM tasks: flat obs (np.ndarray), action_dim=3, no goal
  PSM tasks: dict obs {'observation', 'achieved_goal', 'desired_goal'},
             action_dim=5 (delta_pos[3] + delta_yaw[1] + gripper[1])

Usage
-----
    # StaticTrack (ECM)
    python vlm/dataset/export_expert_universal.py \\
        --task StaticTrack-v0 --episodes 100 --render

    # NeedlePick (PSM)
    python vlm/dataset/export_expert_universal.py \\
        --task NeedlePick-v0 --episodes 100 --render

    # GauzeRetrieve (PSM)
    python vlm/dataset/export_expert_universal.py \\
        --task GauzeRetrieve-v0 --episodes 100 --render

    # All three in one go
    python vlm/dataset/export_expert_universal.py \\
        --task StaticTrack-v0 NeedlePick-v0 GauzeRetrieve-v0 \\
        --episodes 50 --render
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import IO, Dict, List, Optional

import gym
import numpy as np
from surrol.gym.surrol_env import SurRoLEnv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Task metadata ────────────────────────────────────────────

TASK_INSTRUCTIONS: Dict[str, str] = {
    "ActiveTrack-v0": (
        "Keep the red cube centered in the camera view using the endoscope. "
        "Left image is the global view, right image is the endoscope view."
    ),
    "StaticTrack-v0": (
        "Keep the camera steady and track the static target. "
        "Adjust the endoscope to maintain the target at the center of the view."
    ),
    "ECMReach-v0": (
        "Move the endoscope camera to reach the target position in 3D space."
    ),
    "MisOrient-v0": (
        "Adjust the endoscope orientation to align with the target orientation."
    ),
    "NeedlePick-v0": (
        "Use the surgical gripper to pick up the needle from the tray "
        "and lift it to the target position above the workspace."
    ),
    "GauzeRetrieve-v0": (
        "Use the surgical gripper to grasp the gauze pad from the tray "
        "and retrieve it to the target position."
    ),
    "PegTransfer-v0": (
        "Pick up the block from one peg and transfer it to the target peg."
    ),
    "NeedleReach-v0": (
        "Move the surgical instrument tip to reach the target position in 3D space."
    ),
    "NeedleRegrasp-v0": (
        "Use both arms to regrasp the needle: pick it up with one arm "
        "and hand it over to the other arm."
    ),
    "BiPegTransfer-v0": (
        "Use both arms to pick up the block and transfer it between pegs."
    ),
}


# ── Helpers ───────────────────────────────────────────────────

_warned_imageio = False


def _to_serializable(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _save_jpg(image: np.ndarray, path: Path):
    global _warned_imageio
    try:
        import imageio.v2 as iio
        iio.imwrite(path, image, format="jpeg")
    except Exception as exc:
        if not _warned_imageio:
            _warned_imageio = True
            print(f"[debug] Could not write JPEGs: {exc}")


def _flatten_obs(obs) -> np.ndarray:
    """Convert observation to flat numpy array, handling both dict and array obs."""
    if isinstance(obs, dict):
        # GoalEnv dict obs → concatenate all parts
        parts = []
        for key in ("observation", "achieved_goal", "desired_goal"):
            if key in obs:
                parts.append(np.asarray(obs[key], dtype=np.float32).ravel())
        return np.concatenate(parts) if parts else np.array([], dtype=np.float32)
    return np.asarray(obs, dtype=np.float32).ravel()


# ── Core collection ──────────────────────────────────────────


def collect_episode(
    env,
    episode_id: int,
    max_steps: int,
    instruction: str,
    out_dir: Path,
    frames_dir: Path,
    downsample: int = 2,
) -> tuple[dict, list[dict]]:
    """Collect one episode using the oracle policy.

    Returns
    -------
    (stats, records)
    """
    obs = env.reset()
    if isinstance(obs, tuple):
        obs, _ = obs

    ep_frame_dir = frames_dir / f"ep{episode_id:04d}"
    ep_frame_dir.mkdir(parents=True, exist_ok=True)

    samples = 0
    total_reward = 0.0
    success = False

    records = []
    for t in range(max_steps):
        # Oracle action
        action = env.get_oracle_action(obs)

        # Step
        step_out = env.step(action)
        if len(step_out) == 5:
            next_obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            next_obs, reward, done, info = step_out

        # Render + save image
        image = env.render(mode="rgb_array")
        if downsample > 1:
            image = image[::downsample, ::downsample]
        jpg_path = ep_frame_dir / f"t{t:04d}.jpg"
        _save_jpg(image, jpg_path)

        # Build record
        obs_flat = _flatten_obs(obs)
        action_flat = np.asarray(action, dtype=np.float32)

        record = {
            "episode": episode_id,
            "t": t,
            "instruction": instruction,
            "image": str(jpg_path.relative_to(out_dir)),
            "obs": obs_flat.tolist(),
            "action": action_flat.tolist(),
            "reward": float(reward),
            "done": bool(done),
            "info": json.loads(json.dumps(info, default=_to_serializable)),
        }

        # Add goal info for GoalEnv tasks
        if isinstance(obs, dict):
            if "achieved_goal" in obs:
                record["achieved_goal"] = np.asarray(obs["achieved_goal"], dtype=np.float32).tolist()
            if "desired_goal" in obs:
                record["desired_goal"] = np.asarray(obs["desired_goal"], dtype=np.float32).tolist()

        records.append(record)
        samples += 1
        total_reward += float(reward)

        if info.get("is_success", False):
            success = True

        obs = next_obs
        if done:
            break

    stats = {"steps": samples, "total_reward": total_reward, "success": success}
    return stats, records


def _worker_fn(
    task_id: str,
    ep_indices: list[int],
    max_steps: int,
    instruction: str,
    out_dir: Path,
    frames_dir: Path,
    seed: Optional[int],
    downsample: int,
) -> list[tuple[dict, list[dict]]]:
    """Worker process: creates its own env and runs a chunk of episodes."""
    # Each worker must import locally to be safe
    import gym
    import surrol.gym.surrol_env

    env = gym.make(task_id, render_mode="rgb_array")
    
    results = []
    for ep_idx in ep_indices:
        # Seed per episode to ensure variety and reproducibility
        if seed is not None:
            ep_seed = seed + ep_idx
            env.seed(ep_seed)
            env.action_space.seed(ep_seed)
            env.observation_space.seed(ep_seed)
            # Extra safety: seed global random states as well
            import random
            random.seed(ep_seed)
            np.random.seed(ep_seed)
        
        stats, records = collect_episode(
            env, ep_idx, max_steps, instruction,
            out_dir, frames_dir, downsample=downsample
        )
        results.append((stats, records))
    
    env.close()
    return results

def _worker_fn_unpacked(args):
    """Helper for imap_unordered which passes a single tuple."""
    return _worker_fn(*args)


def export_task(
    task_id: str,
    num_episodes: int,
    max_steps: int,
    output_base: Path,
    seed: Optional[int],
    instruction: Optional[str],
    downsample: int,
    jobs: int = 1,
):
    """Export expert demonstrations for a single task."""
    # Resolve instruction
    if instruction is None:
        instruction = TASK_INSTRUCTIONS.get(task_id, f"Complete the {task_id} task.")

    # Task-specific output dir
    task_name = task_id.replace("-v0", "").lower()
    # Prefer snake_case directory names (e.g. expert_static_track) for consistency
    # with SurrolWrapper TASK_NAME_TO_ID. Fall back to legacy camel-lower names
    # (e.g. expert_statictrack) if mapping is unavailable.
    try:
        from surrol.wrappers.surrol_sb3 import TASK_NAME_TO_ID  # type: ignore
        _id_to_key = {v: k for k, v in TASK_NAME_TO_ID.items()}
        task_name = _id_to_key.get(task_id, task_name)
    except Exception:
        pass
    out_dir = output_base / f"expert_{task_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.jsonl"
    manifest = manifest_path.open("w", encoding="utf-8")

    num_jobs = min(num_episodes, jobs)
    print(f"\n{'='*60}")
    print(f"  Exporting: {task_id}")
    print(f"  Instruction: {instruction[:80]}...")
    print(f"  Episodes: {num_episodes}, Max steps: {max_steps}")
    print(f"  Parallel jobs: {num_jobs}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    # Chunk episodes for workers
    all_ep_indices = list(range(num_episodes))
    chunks = np.array_split(all_ep_indices, num_jobs)
    chunks = [c.tolist() for c in chunks if len(c) > 0]

    import multiprocessing as mp
    total_samples = 0
    successes = 0

    with mp.Pool(processes=num_jobs) as pool:
        # Prepare arguments
        worker_args = [
            (task_id, chunk, max_steps, instruction, out_dir, frames_dir, seed, downsample)
            for chunk in chunks
        ]
        
        # Use imap_unordered for streaming results as they complete
        # We need a flat list of arguments for imap
        # Each flat arg is a tuple for a SINGLE episode to allow fine-grained progress
        flat_args = [
            (task_id, [ep], max_steps, instruction, out_dir, frames_dir, seed, downsample)
            for ep in range(num_episodes)
        ]
        
        # We use a progress counter
        completed = 0
        for worker_result in pool.imap_unordered(_worker_fn_unpacked, flat_args):
            for stats, records in worker_result:
                total_samples += stats["steps"]
                if stats["success"]:
                    successes += 1
                
                # Write records to manifest IMMEDIATELY
                for record in records:
                    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                # Flush to ensure it's written to disk immediately
                manifest.flush()
                
                completed += 1
                ep_id = records[0]["episode"] if records else -1
                
                # Optional: periodic progress (main process)
                if completed % 10 == 0 or completed == 1 or completed == num_episodes:
                    print(f"  Completed Episode {ep_id} [{completed}/{num_episodes}]: "
                          f"steps={stats['steps']}  reward={stats['total_reward']:.3f}  "
                          f"success={stats['success']}")

    manifest.close()

    print(f"\n  Summary for {task_id}:")
    print(f"    Total samples: {total_samples}")
    print(f"    Success rate:  {successes}/{num_episodes} ({successes/max(1,num_episodes)*100:.0f}%)")
    print(f"    Manifest:      {manifest_path}")

    return {
        "task": task_id,
        "total_samples": total_samples,
        "success_rate": successes / max(1, num_episodes),
        "manifest": str(manifest_path),
        "out_dir": str(out_dir),
        "instruction": instruction,
    }


# ── Main ──────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Export expert rollouts for VLA fine-tuning (multi-task)"
    )
    parser.add_argument(
        "--task", type=str, nargs="+",
        default=["StaticTrack-v0"],
        help="One or more Gym env IDs to export",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("./vlm/dataset"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instruction", type=str, default=None,
                        help="Override instruction (applies to all tasks)")
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=4,
                        help="Number of parallel processes for collection")
    parser.add_argument("--render", action="store_true",
                        help="(kept for CLI compat, always renders rgb_array)")
    args = parser.parse_args()

    all_results: List[dict] = []
    for task_id in args.task:
        result = export_task(
            task_id=task_id,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            output_base=args.out,
            seed=args.seed,
            instruction=args.instruction,
            downsample=args.downsample,
            jobs=args.jobs,
        )
        all_results.append(result)

    # ── Instructions for next step ──
    print(f"\n  Next Step: Run data processing in your VLM environment")
    print(f"    python vlm/dataset/convert_manifests.py --out \"{args.out}\"")
    print(f"{'='*60}")

    # Final summary
    print(f"\n{'='*60}")
    print("  Export Complete!")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  {r['task']}: {r['total_samples']} samples, "
              f"success={r['success_rate']*100:.0f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
