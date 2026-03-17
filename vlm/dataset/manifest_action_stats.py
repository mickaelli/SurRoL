#!/usr/bin/env python
"""Manifest action statistics (streaming).

This script scans one or more `manifest.jsonl` files produced by the expert
exporters and prints basic action distribution stats, with special focus on
PSM gripper actions (action[4] < 0 means CLOSE in SurRoL).

Examples
--------
python vlm/dataset/manifest_action_stats.py ^
  --manifest vlm/dataset/expert_needle_pick/manifest.jsonl ^
  --manifest vlm/dataset/expert_gauze_retrieve/manifest.jsonl ^
  --progress-every 200000
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class OnlineStats:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2
        if x < self.min:
            self.min = x
        if x > self.max:
            self.max = x

    def std(self) -> float:
        if self.n < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.n - 1))


def _is_finite_list(vals: object) -> bool:
    if not isinstance(vals, list) or not vals:
        return False
    for v in vals:
        if not isinstance(v, (int, float)):
            return False
        if not math.isfinite(float(v)):
            return False
    return True


def scan_manifest(path: Path, max_lines: Optional[int], progress_every: int) -> None:
    total_lines = 0
    valid_actions = 0
    dim: Optional[int] = None
    dim_mismatch = 0

    per_dim: list[OnlineStats] = []

    # For PSM-style actions (dim>=5): jaw close rate + when it happens
    jaw_close = 0
    jaw_total = 0
    t_total: list[int] = []
    t_close: list[int] = []

    # Movement magnitude summaries (mostly for debugging "stuck after reach")
    dpos_norm = OnlineStats()
    dz = OnlineStats()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if max_lines is not None and total_lines >= max_lines:
                break
            total_lines += 1
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            action = item.get("action")
            if not _is_finite_list(action):
                continue

            action_f = [float(x) for x in action]
            if dim is None:
                dim = len(action_f)
                per_dim = [OnlineStats() for _ in range(dim)]
            if len(action_f) != dim:
                dim_mismatch += 1
                continue

            valid_actions += 1
            for i, x in enumerate(action_f):
                per_dim[i].update(x)

            # Movement magnitude
            if dim >= 3:
                norm = math.sqrt(action_f[0] ** 2 + action_f[1] ** 2 + action_f[2] ** 2)
                dpos_norm.update(norm)
                dz.update(action_f[2])

            # Jaw close stats (PSM)
            if dim >= 5:
                jaw_total += 1
                is_close = action_f[4] < 0
                if is_close:
                    jaw_close += 1

                t = item.get("t")
                if isinstance(t, int) and t >= 0:
                    while len(t_total) <= t:
                        t_total.append(0)
                        t_close.append(0)
                    t_total[t] += 1
                    if is_close:
                        t_close[t] += 1

            if progress_every > 0 and total_lines % progress_every == 0:
                print(f"  ... {path.name}: scanned {total_lines:,} lines")

    print("\n" + "=" * 72)
    print(f"Manifest: {path}")
    print(f"Total lines:       {total_lines:,}")
    print(f"Valid actions:     {valid_actions:,}")
    if dim is not None:
        print(f"Action dim:        {dim}")
    if dim_mismatch:
        print(f"Dim mismatches:    {dim_mismatch:,} (skipped)")

    if per_dim:
        print("\nPer-dimension stats (min / max / mean ± std):")
        for i, st in enumerate(per_dim):
            print(f"  a[{i}]: {st.min:+.4f} / {st.max:+.4f} / {st.mean:+.4f} ± {st.std():.4f}")

    if dpos_norm.n:
        print("\nDelta-pos magnitude:")
        print(f"  ||dpos|| mean ± std: {dpos_norm.mean:.4f} ± {dpos_norm.std():.4f}")
        print(f"  dz mean ± std:       {dz.mean:.4f} ± {dz.std():.4f}")

    if jaw_total:
        close_ratio = jaw_close / jaw_total
        print("\nJaw close ratio (action[4] < 0):")
        print(f"  close: {jaw_close:,} / {jaw_total:,} = {close_ratio * 100:.2f}%")
        if t_total:
            print("  close% by timestep (top 10 timesteps by close%):")
            pairs = []
            for t, tot in enumerate(t_total):
                if tot <= 0:
                    continue
                pairs.append((t_close[t] / tot, t, t_close[t], tot))
            pairs.sort(reverse=True)
            for frac, t, c, tot in pairs[:10]:
                print(f"    t={t:03d}: {c}/{tot} = {frac * 100:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming action stats for SurRoL manifest.jsonl")
    parser.add_argument("--manifest", type=Path, action="append", required=True, help="Path to manifest.jsonl")
    parser.add_argument("--max-lines", type=int, default=None, help="Optional max lines per file (for quick sampling)")
    parser.add_argument("--progress-every", type=int, default=0, help="Print progress every N lines (0=off)")
    args = parser.parse_args()

    for p in args.manifest:
        scan_manifest(p, max_lines=args.max_lines, progress_every=args.progress_every)


if __name__ == "__main__":
    main()

