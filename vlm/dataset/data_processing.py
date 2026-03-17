"""Minimal dataset + preprocessing utilities for ActiveTrack VLA data.

Reads a manifest.jsonl with fields:
- image: relative path to JPEG
- instruction: text prompt
- obs: list of floats
- action: list of floats
- reward, done, info: optional

Capabilities:
- filter NaN/Inf in obs/action, track action min/max
- normalize actions to [-1, 1]
- build vector-quantized (VQ) action codebooks (k-means; optional faiss)
- export ShareGPT-style multimodal JSON for LLaMA-Factory/other SFT pipelines

from pathlib import Path
import torch
from vlm.dataset.data_processing import (
    compute_action_min_max,
    build_action_codebook_faiss,
    convert_manifest_to_sharegpt,
)

src = Path("expert_active_track/manifest.jsonl")

# 连续值模式 (推荐用于闭环控制)
convert_manifest_to_sharegpt(
    manifest_path=src,
    output_path=Path("expert_active_track/training_data.json"),
    user_prompt="请根据当前手术台环境，输出机械臂下一步的动作参数。",
    mode="continuous",
    precision=3,
)

# 归一化模式
act_min, act_max = compute_action_min_max(src)

# VQ 离散模式
actions = torch.tensor([json.loads(l)["action"] for l in src.open() if l.strip()], dtype=torch.float32)
codebook = build_action_codebook_faiss(actions, num_codes=256, num_iters=25, seed=0)
convert_manifest_to_sharegpt(
    manifest_path=src,
    output_path=Path("expert_active_track/training_data_vq.json"),
    user_prompt="请根据当前手术台环境，输出机械臂下一步的动作参数。",
    mode="vq",
    codebook=codebook,
    precision=2,
)
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


def _is_valid_numeric(seq: List[float]) -> bool:
	tensor = torch.tensor(seq, dtype=torch.float32)
	return torch.isfinite(tensor).all().item()


def compute_action_min_max(manifest_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Scan manifest.jsonl to get per-dimension min and max of actions."""
	mins: Optional[torch.Tensor] = None
	maxs: Optional[torch.Tensor] = None
	with Path(manifest_path).open("r", encoding="utf-8") as f:
		for line in f:
			if not line.strip():
				continue
			item: Dict[str, object] = json.loads(line)
			act = item.get("action")
			if act is None or not _is_valid_numeric(act):
				continue
			vec = torch.tensor(act, dtype=torch.float32)
			mins = vec if mins is None else torch.minimum(mins, vec)
			maxs = vec if maxs is None else torch.maximum(maxs, vec)
	if mins is None or maxs is None:
		return torch.tensor([], dtype=torch.float32), torch.tensor([], dtype=torch.float32)
	return mins, maxs


def normalize_action(action: torch.Tensor, act_min: torch.Tensor, act_max: torch.Tensor) -> torch.Tensor:
	"""Linearly map action to [-1, 1]; handles degenerate ranges by leaving as zero."""
	if act_min.numel() == 0 or act_max.numel() == 0:
		return action
	denom = act_max - act_min
	denom = torch.where(denom == 0, torch.ones_like(denom), denom)
	return 2 * (action - act_min) / denom - 1


def build_action_codebook(
	actions: torch.Tensor,
	num_codes: int = 256,
	num_iters: int = 20,
	seed: int = 0,
) -> torch.Tensor:
	"""Simple k-means to build a VQ codebook for continuous actions.

	Args:
	    actions: [N, D] float tensor of actions.
	    num_codes: codebook size.
	    num_iters: k-means iterations.
	    seed: RNG seed for init.

	Returns:
	    codebook: [num_codes, D] centroids.
	"""
	assert actions.ndim == 2, "actions must be [N, D]"
	gen = torch.Generator().manual_seed(seed)
	N = actions.shape[0]
	idx = torch.randperm(N, generator=gen)[:num_codes]
	codebook = actions[idx].clone()
	for _ in range(num_iters):
		# Assign
		dists = (
			actions.pow(2).sum(dim=1, keepdim=True)
			- 2 * actions @ codebook.t()
			+ codebook.pow(2).sum(dim=1)
		)
		assign = dists.argmin(dim=1)
		# Update
		for k in range(num_codes):
			mask = assign == k
			if mask.any():
				codebook[k] = actions[mask].mean(dim=0)
			else:
				# Reinitialize empty cluster to a random action
				ridx = torch.randint(0, N, (1,), generator=gen).item()
				codebook[k] = actions[ridx]
	return codebook


def build_action_codebook_faiss(
	actions: torch.Tensor,
	num_codes: int = 256,
	num_iters: int = 20,
	seed: int = 0,
) -> torch.Tensor:
	"""FAISS-based k-means for codebook (uses L2).

	Requires faiss-gpu or faiss-cpu installed. Falls back to torch k-means if import fails.
	"""
	try:
		import faiss  # type: ignore
	except Exception:
		return build_action_codebook(actions, num_codes=num_codes, num_iters=num_iters, seed=seed)

	assert actions.ndim == 2, "actions must be [N, D]"
	N, D = actions.shape
	# faiss expects float32 contiguous
	arr = actions.detach().cpu().numpy().astype("float32", copy=False)
	clu = faiss.Clustering(D, num_codes)
	clu.niter = num_iters
	clu.seed = seed
	index = faiss.IndexFlatL2(D)
	clu.train(arr, index)
	centroids = faiss.vector_to_array(clu.centroids).reshape(num_codes, D)
	return torch.from_numpy(centroids).to(actions)


def encode_action_to_token(action: torch.Tensor, codebook: torch.Tensor) -> int:
	"""Map a single action to nearest code index."""
	with torch.no_grad():
		dists = ((codebook - action) ** 2).sum(dim=1)
		return int(dists.argmin().item())


def decode_token_to_action(token: int, codebook: torch.Tensor) -> torch.Tensor:
	"""Lookup code vector for a token."""
	return codebook[int(token)]



class ManifestDataset(torch.utils.data.Dataset):
	def __init__(
		self,
		manifest_path: Path,
		image_root: Optional[Path] = None,
		transform: Optional[Callable] = None,
		normalize_actions: bool = False,
		action_min: Optional[torch.Tensor] = None,
		action_max: Optional[torch.Tensor] = None,
	) -> None:
		self.manifest_path = Path(manifest_path)
		self.image_root = image_root or self.manifest_path.parent
		self.transform = transform
		self.normalize_actions = normalize_actions

		self.samples: List[Dict] = []
		action_vals: List[torch.Tensor] = []

		with self.manifest_path.open("r", encoding="utf-8") as f:
			for line in f:
				if not line.strip():
					continue
				item: Dict[str, object] = json.loads(line)
				obs = item.get("obs")
				act = item.get("action")
				if obs is None or act is None:
					continue
				if not (_is_valid_numeric(obs) and _is_valid_numeric(act)):
					continue
				self.samples.append(item)
				action_vals.append(torch.tensor(act, dtype=torch.float32))

		if action_vals:
			stacked = torch.stack(action_vals)
			self.action_min = stacked.min(dim=0).values if action_min is None else action_min
			self.action_max = stacked.max(dim=0).values if action_max is None else action_max
		else:
			self.action_min = torch.tensor([], dtype=torch.float32)
			self.action_max = torch.tensor([], dtype=torch.float32)

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int) -> Dict[str, object]:
		item = self.samples[idx]
		img_path = self.image_root / item["image"]
		image = Image.open(img_path).convert("RGB")
		if self.transform:
			image = self.transform(image)
		else:
			arr = np.array(image, dtype=np.uint8)
			image = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

		obs = torch.tensor(item["obs"], dtype=torch.float32)
		action = torch.tensor(item["action"], dtype=torch.float32)
		action_norm = normalize_action(action, self.action_min, self.action_max) if self.normalize_actions else action

		return {
			"image": image,
			"instruction": item.get("instruction", ""),
			"obs": obs,
			"action": action,
			"action_norm": action_norm,
			"reward": item.get("reward", None),
			"done": item.get("done", None),
			"info": item.get("info", None),
		}


def build_dataloader(
	manifest_path: Path,
	batch_size: int = 8,
	shuffle: bool = True,
	num_workers: int = 4,
	image_root: Optional[Path] = None,
	transform: Optional[Callable] = None,
	normalize_actions: bool = False,
	action_min: Optional[torch.Tensor] = None,
	action_max: Optional[torch.Tensor] = None,
) -> Tuple[ManifestDataset, torch.utils.data.DataLoader]:
	dataset = ManifestDataset(
		manifest_path,
		image_root=image_root,
		transform=transform,
		normalize_actions=normalize_actions,
		action_min=action_min,
		action_max=action_max,
	)
	loader = torch.utils.data.DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		pin_memory=True,
		drop_last=False,
	)
	return dataset, loader


def load_actions(manifest_path: Path) -> torch.Tensor:
	"""Load all actions from manifest into a tensor [N, D]."""
	acts: List[torch.Tensor] = []
	with Path(manifest_path).open("r", encoding="utf-8") as f:
		for line in f:
			if not line.strip():
				continue
			item: Dict[str, object] = json.loads(line)
			act = item.get("action")
			if act is None or not _is_valid_numeric(act):
				continue
			acts.append(torch.tensor(act, dtype=torch.float32))
	if not acts:
		return torch.empty((0, 0), dtype=torch.float32)
	return torch.stack(acts)


def convert_manifest_to_sharegpt(
	manifest_path: Path,
	output_path: Path,
	user_prompt: str,
	mode: str = "normalized",
	action_min: Optional[torch.Tensor] = None,
	action_max: Optional[torch.Tensor] = None,
	codebook: Optional[torch.Tensor] = None,
	precision: int = 2,
	max_samples: Optional[int] = None,
) -> None:
	"""Convert SurRoL manifest.jsonl to ShareGPT-style JSON for multimodal SFT.

	mode: "continuous" (raw), "normalized" ([-1,1] with provided min/max), or "vq" (discrete token via codebook).
	"""
	assert mode in {"continuous", "normalized", "vq"}
	man_path = Path(manifest_path)
	out_path = Path(output_path)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	rows: List[Dict[str, object]] = []
	with man_path.open("r", encoding="utf-8") as fin:
		for idx, line in enumerate(fin):
			if max_samples is not None and idx >= max_samples:
				break
			if not line.strip():
				continue
			item: Dict[str, object] = json.loads(line)
			act = item.get("action")
			img = item.get("image")
			if act is None or img is None or not _is_valid_numeric(act):
				continue
			action_tensor = torch.tensor(act, dtype=torch.float32)
			if mode == "normalized":
				if action_min is None or action_max is None:
					raise ValueError("action_min/action_max required for normalized mode")
				payload = normalize_action(action_tensor, action_min, action_max).tolist()
			elif mode == "vq":
				if codebook is None:
					raise ValueError("codebook required for vq mode")
				token = encode_action_to_token(action_tensor, codebook)
				action_dec = decode_token_to_action(token, codebook).tolist()
				payload = {"token": token, "action_decoded": [round(float(x), precision) for x in action_dec]}
			else:
				payload = action_tensor.tolist()
			if mode != "vq":
				payload = [round(float(x), precision) for x in payload]  # type: ignore[assignment]
			messages = [
				{"role": "user", "content": f"<image>{user_prompt}"},
				{"role": "assistant", "content": json.dumps({"action": payload}, ensure_ascii=False) if mode != "vq" else json.dumps(payload, ensure_ascii=False)},
			]
			rows.append({
				"id": f"sample-{idx}",
				"conversations": messages,
				"images": [str(img)],
			})
	with out_path.open("w", encoding="utf-8") as fout:
		json.dump(rows, fout, ensure_ascii=False, indent=2)


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Convert SurRoL manifest to ShareGPT JSON for SFT")
	parser.add_argument("--manifest", type=Path, default=Path("./vlm/dataset/expert_active_track/manifest.jsonl"), help="Input manifest.jsonl path")
	parser.add_argument("--output", type=Path, default=Path("./vlm/dataset/expert_active_track/training_data.json"), help="Output JSON path")
	parser.add_argument("--user-prompt", type=str, default="请根据当前手术台环境，输出机械臂下一步的动作参数。", help="User prompt for each sample")
	parser.add_argument("--mode", type=str, choices=["continuous", "normalized", "vq"], default="continuous", help="Conversion mode")
	parser.add_argument("--precision", type=int, default=2, help="Decimal precision for action values")
	parser.add_argument("--num-codes", type=int, default=256, help="Codebook size for vq mode")
	parser.add_argument("--kmeans-iters", type=int, default=25, help="K-means iterations for vq mode")
	args = parser.parse_args()

	act_min, act_max = compute_action_min_max(args.manifest)
	codebook = None
	if args.mode == "vq":
		actions = load_actions(args.manifest)
		codebook = build_action_codebook_faiss(actions, num_codes=args.num_codes, num_iters=args.kmeans_iters, seed=0)

	convert_manifest_to_sharegpt(
		manifest_path=args.manifest,
		output_path=args.output,
		user_prompt=args.user_prompt,
		mode=args.mode,
		action_min=act_min,
		action_max=act_max,
		codebook=codebook,
		precision=args.precision,
	)