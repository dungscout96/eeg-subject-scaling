#!/usr/bin/env bash
# Regenerate every figure and number in the paper from the shipped results/.
# CPU only, about a minute. Run from the repository root:  bash scripts/reproduce_from_results.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/4  checkpoint selection from the held-out epoch curves (dry run: checks only)"
uv run python scripts/select_checkpoints.py > /dev/null
uv run python scripts/select_checkpoints.py --metric probe_mean_r > /dev/null

echo "== 2/4  FLOPs per step / epoch (Appendix C.3)"
uv run python analysis/count_train_flops.py > /dev/null

echo "== 3/4  figures -> figures/"
uv run python analysis/plot_method_figure.py
uv run python analysis/plot_fig1_scaling.py
uv run python analysis/plot_optimum_compute.py > /dev/null && echo "wrote figures/optimum_compute_e2v1_time.png / .pdf"

echo "== 4/4  every reported number, checked against the paper -> results/paper_tables.txt"
uv run python analysis/paper_numbers.py | tail -n 1
