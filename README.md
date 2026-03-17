# SurRoL-VLM

This repository is a secondary development project based on [SurRoL](https://github.com/med-air/SurRoL). It keeps the SurRoL simulation foundation and extends it with VLM/VLA/RLAIF-related code for surgical robot control experiments.

The repository inherits SurRoL code under the original MIT license. See [LICENSE](LICENSE) for the license text.

## Overview

- Base simulator: SurRoL + PyBullet + dVRK-compatible task abstractions
- Main extension: `vlm/` pipeline for dataset export, VLA fine-tuning, inference service, VLM reward scoring, and RLAIF training
- Current practical focus: static/PSM task experiments such as `StaticTrack`, `NeedlePick`, and `GauzeRetrieve`

## Repository Layout

- `surrol/`: core simulator, robot models, tasks, wrappers, assets
- `vlm/`: VLM-related code
- `vlm/dataset/`: expert export, manifest conversion, dataset utilities
- `vlm/trainer/`: VLA and RLAIF training/inference entrypoints
- `vlm/eval/`: closed-loop evaluation, zero-shot evaluation, plotting, VLA server
- `tests/`: notebooks and basic environment checks
- `run/`: legacy experiment scripts

## Quick Start

### 1. Install

This codebase mixes legacy SurRoL components and newer PyTorch/VLM components. In practice, they are often run in separate environments or on separate machines.

Basic local editable install:

```bash
pip install -e .
```

For modern SB3-based training:

```bash
pip install torch>=2.0 gymnasium "stable-baselines3[extra]>=2.0.0"
```

### 2. Run SB3 baseline training

```bash
python train_sb3.py --task peg_transfer --obs-mode rgb --total-timesteps 100000
```

### 3. VLM workflow

Typical VLM workflow:

1. Export expert data with `vlm/dataset/export_expert_universal.py`
2. Convert `manifest.jsonl` into training data with `vlm/dataset/convert_manifests.py`
3. Fine-tune a VLA model with `vlm/trainer/train_vla.py`
4. Serve inference with `vlm/eval/vla_server.py`
5. Run closed-loop evaluation with `vlm/eval/eval_closed_loop.py`

Example config file:

- [train_config.yaml](E:/LLM/SurRoL-main/vlm/config/train_config.yaml)

## Notes

- `baselines/` is intentionally not tracked in this repository. If you need OpenAI Baselines-related legacy experiments, prepare that dependency separately.
- The VLM and simulation environments may be split across different machines. The repository supports client/server inference and remote reward scoring for that reason.
- Some scripts still reflect mixed legacy and experimental workflows. Treat `vlm/` as the main entrypoint for current multimodal work.

## Upstream Attribution

This project is derived from SurRoL:

- Project page: <https://med-air.github.io/SurRoL/>
- Paper: [SurRoL: An Open-source Reinforcement Learning Centered and dVRK Compatible Platform for Surgical Robot Learning](https://arxiv.org/abs/2108.13035)

If you use this repository in research, cite the original SurRoL work as appropriate.

```bibtex
@inproceedings{xu2021surrol,
  title={SurRoL: An Open-source Reinforcement Learning Centered and dVRK Compatible Platform for Surgical Robot Learning},
  author={Xu, Jiaqi and Li, Bin and Lu, Bo and Liu, Yun-Hui and Dou, Qi and Heng, Pheng-Ann},
  booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2021},
  organization={IEEE}
}
```

## License

This repository includes code derived from SurRoL and is distributed under the MIT license. See [LICENSE](LICENSE).
