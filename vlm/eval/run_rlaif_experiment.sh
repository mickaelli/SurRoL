#!/usr/bin/env bash
# SurRoL-VLA · RLAIF vs. Baseline RL Experiment Runner
#
# This script runs a comparative experiment to measure sample efficiency
# and success rate:
#   1. Baseline RL: Pure PPO with sparse environment rewards.
#   2. RLAIF: PPO augmented with dense VLM-generated visual rewards.
#
# Usage:
#   bash vlm/eval/run_rlaif_experiment.sh

set -e

# Configuration
TASK="static_track"
TIMESTEPS=50000
VLM_MODEL="Qwen/Qwen2-VL-2B-Instruct"
SCORE_EVERY=5

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================================${NC}"
echo -e "${GREEN}  Starting RLAIF Comparative Experiment  ${NC}"
echo -e "${BLUE}=======================================================${NC}"
echo "Task: $TASK"
echo "Total Timesteps per run: $TIMESTEPS"
echo "VLM Model: $VLM_MODEL"
echo ""

# ── 1. Run Baseline (Sparse Reward) ──
echo -e "${BLUE}[1/3] Running Baseline (Sparse Reward RL)...${NC}"
python vlm/trainer/train_rlaif.py \
    --vlm-model "$VLM_MODEL" \
    --task "$TASK" \
    --reward-mode "none" \
    --total-timesteps $TIMESTEPS \
    --log-dir "logs/experiment/baseline"

# ── 2. Run RLAIF (Dense VLM Reward) ──
echo -e "${BLUE}[2/3] Running RLAIF (Dense Visual Reward)...${NC}"
python vlm/trainer/train_rlaif.py \
    --vlm-model "$VLM_MODEL" \
    --task "$TASK" \
    --reward-mode "replace" \
    --score-every $SCORE_EVERY \
    --total-timesteps $TIMESTEPS \
    --log-dir "logs/experiment/rlaif"

# ── 3. Run Dense Human Reward (Heuristic) ──
echo -e "${BLUE}[3/3] Running Dense Human Reward (Heuristic Distance)...${NC}"
python vlm/trainer/train_rlaif.py \
    --vlm-model "$VLM_MODEL" \
    --task "$TASK" \
    --reward-mode "dense_human" \
    --total-timesteps $TIMESTEPS \
    --log-dir "logs/experiment/dense_human"

# ── 4. Plot Results ──
echo -e "${BLUE}[4/4] Generating Comparative Plots...${NC}"
python vlm/eval/plot_learning_curves.py \
    --baseline-dir "logs/experiment/baseline" \
    --dense-human-dir "logs/experiment/dense_human" \
    --rlaif-dir "logs/experiment/rlaif" \
    --output "logs/experiment/rlaif_ablation.png"

echo -e "${GREEN}Experiment complete! Results saved to logs/experiment/rlaif_ablation.png${NC}"
