#!/usr/bin/env python
"""SurRoL-VLA · Closed-loop evaluation: VLA model ↔ SurRoL environment.

The script runs the fine-tuned Qwen2-VL VLA model inside SurRoL in a
closed loop:

    ┌─────────────────────────────────────────────────────────┐
    │  reset env  →  render image  →  VLA predicts action     │
    │     ↑                               │                   │
    │     └───── env.step(action) ← ──────┘                   │
    └─────────────────────────────────────────────────────────┘

Metrics collected per episode
-----------------------------
- total_reward        cumulative reward
- ep_length           number of steps
- success             whether the task was solved (if available)
- avg_reward_per_step mean reward
- action_parse_fails  how many times the model output could not be parsed

Usage
-----
    # 1. Local Inference (Requires torch/transformers)
    python vlm/eval/eval_closed_loop.py \
        --model Qwen/Qwen2-VL-2B-Instruct \
        --lora-path vlm/out/qwen2vl_vla_lora/lora_weights \
        --task StaticTrack-v0 \
        --episodes 10 \
        --max-steps 200 \
        --save-video

    # 2. Client-Server Inference (Bypasses torch, requires vla_server.py running)
    python vlm/eval/eval_closed_loop.py \
        --task StaticTrack-v0 \
        --server-url http://127.0.0.1:8000/predict \
        --episodes 10 \
        --save-video
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import gym
import numpy as np
from PIL import Image
import urllib.request
from surrol.gym.surrol_env import SurRoLEnv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Note: Torch and VLM imports are now lazy to support pure-client execution
# without requiring Torch or Transformers in the current environment.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_closed_loop")


# ── Episode result container ─────────────────────────────────


@dataclass
class EpisodeResult:
    episode_id: int = 0
    total_reward: float = 0.0
    ep_length: int = 0
    success: bool = False
    avg_reward_per_step: float = 0.0
    action_parse_fails: int = 0
    rewards: List[float] = field(default_factory=list)

    def finalise(self):
        self.avg_reward_per_step = (
            self.total_reward / self.ep_length if self.ep_length > 0 else 0.0
        )


# ── Video recorder helper ────────────────────────────────────


class FrameRecorder:
    """Accumulates RGB frames and writes an MP4 at the end."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.frames: List[np.ndarray] = []

    def add(self, frame: np.ndarray):
        if self.enabled:
            self.frames.append(frame)

    def save(self, path: Path, fps: int = 15):
        if not self.enabled or not self.frames:
            return
        try:
            import imageio.v2 as iio
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = iio.get_writer(str(path), fps=fps, codec="libx264")
            for f in self.frames:
                writer.append_data(f)
            writer.close()
            logger.info("Video saved: %s (%d frames)", path, len(self.frames))
        except Exception as exc:
            logger.warning("Could not write video: %s", exc)

    def reset(self):
        self.frames.clear()


# ── Oracle baseline (for comparison) ─────────────────────────


def get_oracle_action_safe(env, obs) -> Optional[np.ndarray]:
    """Try calling env.get_oracle_action; return None if unavailable."""
    try:
        return env.get_oracle_action(obs)
    except (AttributeError, NotImplementedError):
        return None


# ── Core evaluation loop ─────────────────────────────────────


def evaluate_episode(
    env,
    model,
    processor,
    episode_id: int,
    instruction: str,
    max_steps: int,
    max_new_tokens: int,
    fallback_action_dim: int,
    recorder: FrameRecorder,
    compare_oracle: bool = False,
    downsample: int = 2,
    server_url: Optional[str] = None,
) -> EpisodeResult:
    """Run one episode with the VLA model deciding actions.

    Parameters
    ----------
    env : gym.Env
        SurRoL environment instance.
    model, processor
        Fine-tuned Qwen2-VL model and its processor (None if server_url is used).
    episode_id : int
        Numeric id for logging.
    instruction : str
        Text instruction sent to the model each step.
    max_steps : int
        Maximum number of steps per episode.
    max_new_tokens : int
        Maximum generation length for the VLA model.
    fallback_action_dim : int
        Action dimension to use when parsing fails (zero action).
    recorder : FrameRecorder
        Optional video recorder.
    compare_oracle : bool
        If True, also compute oracle actions for logging comparison.
    downsample : int
        Image downsample factor before sending to VLA (must match training).
    server_url : str
        Optional HTTP server URL to send inference requests to (client mode).
    """
    result = EpisodeResult(episode_id=episode_id)
    recorder.reset()

    obs = env.reset()
    if isinstance(obs, tuple):
        obs, _ = obs

    for t in range(max_steps):
        # ── 1. Render image ──
        image_np = env.render(mode="rgb_array")
        # Downsample to match training resolution
        if downsample > 1:
            image_np = image_np[::downsample, ::downsample]
        recorder.add(image_np)

        pil_image = Image.fromarray(image_np)

        # ── 2. VLA predicts action ──
        if server_url:
            # Client-Server Mode
            import base64
            import io
            
            # Convert PIL to base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            payload = json.dumps({
                "image": img_b64,
                "instruction": instruction
            }).encode("utf-8")
            
            req = urllib.request.Request(
                server_url, 
                data=payload, 
                headers={'Content-Type': 'application/json'}
            )
            
            try:
                with urllib.request.urlopen(req) as response:
                    res_body = response.read().decode("utf-8")
                    res_json = json.loads(res_body)
                    action = res_json.get("action")
                    pred_raw_text = res_json.get("raw_text", "")
            except Exception as e:
                logger.error(f"HTTP Server request failed: {e}")
                action = None
                pred_raw_text = "HTTP ERROR"
        else:
            # Local Inference Mode
            from vlm.trainer.infer_vla import predict_action
            pred = predict_action(
                model, processor, pil_image, instruction,
                max_new_tokens=max_new_tokens,
            )
            action = pred["action"]
            pred_raw_text = pred.get("raw_text", "")

        if action is not None and len(action) != fallback_action_dim:
            result.action_parse_fails += 1
            logger.warning(
                "  ep%d t%d: wrong action dim=%d (expected %d), raw=%s → using zero action",
                episode_id, t, len(action), fallback_action_dim, pred_raw_text[:120],
            )
            action = None

        if action is None:
            result.action_parse_fails += 1
            logger.warning(
                "  ep%d t%d: parse failed, raw=%s → using zero action",
                episode_id, t, pred_raw_text[:120],
            )
            action = [0.0] * fallback_action_dim

        action_np = np.array(action, dtype=np.float32)

        # Optional oracle comparison
        if compare_oracle:
            oracle = get_oracle_action_safe(env, obs)
            if oracle is not None:
                delta = np.linalg.norm(action_np - oracle)
                if t % 20 == 0:
                    logger.info(
                        "  ep%d t%d: VLA=%s  oracle=%s  Δ=%.3f",
                        episode_id, t,
                        np.round(action_np, 2).tolist(),
                        np.round(oracle, 2).tolist(),
                        delta,
                    )

        # ── 3. Step environment ──
        step_out = env.step(action_np)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            obs, reward, done, info = step_out

        result.total_reward += float(reward)
        result.rewards.append(float(reward))
        result.ep_length += 1

        # Check success
        if info.get("is_success", False):
            result.success = True

        if done:
            break

    result.finalise()
    return result


# ── Main ──────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Closed-loop VLA evaluation in SurRoL"
    )
    # Model
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--lora-path", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    # Environment
    parser.add_argument("--task", type=str, default="StaticTrack-v0",
                        help="Gym env id (ActiveTrack-v0, StaticTrack-v0, NeedlePick-v0, ...)")
    parser.add_argument("--server-url", type=str, default="",
                        help="If provided, send image to this VLM HTTP API (e.g. http://127.0.0.1:8000/predict). Bypasses local model loading.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    # VLA
    parser.add_argument(
        "--instruction", type=str,
        default=None,
        help="Text instruction (if not provided, will auto-detect from task)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--downsample", type=int, default=2,
                        help="Image downsample factor (must match training)")
    parser.add_argument("--fallback-action-dim", type=int, default=0,
                        help="Action dim for zero-fallback when parse fails (0=auto from env.action_space)")
    # Output
    parser.add_argument("--out-dir", type=str, default="vlm/eval/results")
    parser.add_argument("--save-video", action="store_true",
                        help="Save episode videos as MP4")
    parser.add_argument("--compare-oracle", action="store_true",
                        help="Log oracle vs VLA action comparison")
    args = parser.parse_args()

    out_dir = _PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model (or setup client) ──
    model, processor = None, None
    if args.server_url:
        logger.info(f"Client Mode: Routing VLA inference to {args.server_url}")
    else:
        logger.info("Local Mode: Loading VLA model: %s + LoRA from %s", args.model, args.lora_path)
        import torch
        from vlm.model.qwen_vl_vla import load_model_for_inference
        torch_dtype = getattr(torch, args.dtype, torch.bfloat16)
        model, processor = load_model_for_inference(
            base_name_or_path=args.model,
            lora_path=args.lora_path,
            torch_dtype=torch_dtype,
        )

    # ── Create environment ──
    logger.info("Creating environment: %s", args.task)
    env = gym.make(args.task, render_mode="rgb_array")
    if args.seed is not None:
        env.seed(args.seed)
        env.action_space.seed(args.seed)
        env.observation_space.seed(args.seed)

    if args.fallback_action_dim <= 0:
        try:
            args.fallback_action_dim = int(env.action_space.shape[0])
        except Exception:
            args.fallback_action_dim = 0

    # ── Resolve Instruction ──
    instruction = args.instruction
    if not instruction:
        try:
            from vlm.dataset.export_expert_universal import TASK_INSTRUCTIONS
            instruction = TASK_INSTRUCTIONS.get(args.task, f"Complete the {args.task} task.")
        except ImportError:
            instruction = "Keep the camera steady and track the static target. Adjust the endoscope to maintain the target at the center of the view."

    # ── Evaluate ──
    all_results: List[EpisodeResult] = []
    recorder = FrameRecorder(enabled=args.save_video)

    logger.info("=" * 60)
    logger.info("Starting closed-loop evaluation: %d episodes, max %d steps",
                args.episodes, args.max_steps)
    logger.info("Instruction used: %s", instruction)
    logger.info("=" * 60)

    for ep in range(args.episodes):
        t0 = time.time()
        result = evaluate_episode(
            env=env,
            model=model,
            processor=processor,
            episode_id=ep,
            instruction=instruction,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            fallback_action_dim=args.fallback_action_dim,
            recorder=recorder,
            compare_oracle=args.compare_oracle,
            downsample=args.downsample,
            server_url=args.server_url if args.server_url else None,
        )
        elapsed = time.time() - t0
        all_results.append(result)

        logger.info(
            "Episode %d/%d ▸ reward=%.3f  length=%d  success=%s  "
            "parse_fails=%d  time=%.1fs",
            ep + 1, args.episodes,
            result.total_reward, result.ep_length, result.success,
            result.action_parse_fails, elapsed,
        )

        if args.save_video:
            video_path = out_dir / f"episode_{ep:03d}.mp4"
            recorder.save(video_path)

    env.close()

    # ── Aggregate metrics ──
    total_rewards = [r.total_reward for r in all_results]
    ep_lengths = [r.ep_length for r in all_results]
    successes = [r.success for r in all_results]
    parse_fails = [r.action_parse_fails for r in all_results]

    summary = {
        "task": args.task,
        "model": args.model,
        "lora_path": args.lora_path,
        "num_episodes": args.episodes,
        "max_steps": args.max_steps,
        "instruction": args.instruction,
        "metrics": {
            "mean_reward": float(np.mean(total_rewards)),
            "std_reward": float(np.std(total_rewards)),
            "min_reward": float(np.min(total_rewards)),
            "max_reward": float(np.max(total_rewards)),
            "mean_ep_length": float(np.mean(ep_lengths)),
            "success_rate": float(np.mean(successes)),
            "total_parse_fails": int(np.sum(parse_fails)),
        },
        "episodes": [asdict(r) for r in all_results],
    }

    # ── Print summary ──
    print("\n" + "=" * 60)
    print("  Closed-Loop Evaluation Summary")
    print("=" * 60)
    m = summary["metrics"]
    print(f"  Task:           {args.task}")
    print(f"  Episodes:       {args.episodes}")
    print(f"  Mean Reward:    {m['mean_reward']:.3f} ± {m['std_reward']:.3f}")
    print(f"  Mean Length:    {m['mean_ep_length']:.1f}")
    print(f"  Success Rate:   {m['success_rate'] * 100:.1f}%")
    print(f"  Parse Fails:    {m['total_parse_fails']}")
    print("=" * 60 + "\n")

    # ── Save results ──
    results_path = out_dir / "eval_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
