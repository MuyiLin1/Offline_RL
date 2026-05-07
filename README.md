# Hybrid Offline-Online Follower Manipulation in General-Sum Stackelberg Games

Synthetic experiments for **hybrid offline–online follower manipulation** in Stackelberg bandits (NeurIPS 2026 submission).

---

## Repository Structure

```
paper/            LaTeX source and final figures
  latex.tex       Main paper
  figures/        All figures referenced in the paper

code/             Source code
  run_experiments.py   Main experiment runner (exp1, exp2, exp3)
  hybrid_fmucb.py      Core FMUCB algorithm implementation
  aggregate.py         Aggregates per-game results into summary plots
  requirements.txt     Python dependencies

results/          Raw experiment results (20 games × 24 seeds each)
  4x4/           4×4 game results (exp1 + exp2 + exp3)
  6x6/           6×6 game results (exp1 + exp2)
  8x8/           8×8 game results (exp1 + exp2)
```

---

## Setup

```bash
cd code
pip install -r requirements.txt
```

## Running Experiments

```bash
cd code

# 4×4 games: Experiment 1 (N_off sweep) + Experiment 2 (coverage quality)
python run_experiments.py --exp1-only --exp2-only \
  --n-a 4 --n-b 4 --horizon 5000 --seeds 24 \
  --game-seeds "737,1482,1603,3043,3197,4098,4356,5003,5466,5621,5954,7184,7466,7657,8001,8400,8671,9011,9134,9153" \
  --exp1-n-off-grid "0,200,500,1000,2000,3000,5000,8000" \
  --exp1-data-quality good --exp2-n-off 1000 \
  --out-dir ../results/4x4 --no-progress

# 6×6 games
python run_experiments.py --exp1-only --exp2-only \
  --n-a 6 --n-b 6 --horizon 8000 --seeds 24 \
  --game-seeds "737,1482,1603,3043,3197,4098,4356,5003,5466,5621,5954,7184,7466,7657,8001,8400,8671,9011,9134,9153" \
  --exp1-n-off-grid "0,500,1000,2000,4000,6000,10000,15000" \
  --exp1-data-quality good --exp2-n-off 2000 \
  --out-dir ../results/6x6 --no-progress

# 8×8 games
python run_experiments.py --exp1-only --exp2-only \
  --n-a 8 --n-b 8 --horizon 12000 --seeds 24 \
  --game-seeds "737,1482,1603,3043,3197,4098,4356,5003,5466,5621,5954,7184,7466,7657,8001,8400,8671,9011,9134,9153" \
  --exp1-n-off-grid "0,1000,2000,4000,8000,12000,20000,30000" \
  --exp1-data-quality good --exp2-n-off 4000 \
  --out-dir ../results/8x8 --no-progress

# Experiment 3 (contextual generalization, 4×4 only)
python run_experiments.py --exp3 \
  --n-a 4 --n-b 4 --horizon 20000 --seeds 24 \
  --exp3-n-off 1000 --exp3-n-x-list "5,10,20,30" --d-ctx 3 \
  --out-dir ../results/4x4 --no-progress
```

## Aggregating Results into Figures

After experiments complete, generate paper-quality plots:

```bash
cd code
python aggregate.py ../results/4x4 ../results/6x6 ../results/8x8
```

## Key Parameters

| Parameter | 4×4 | 6×6 | 8×8 |
|-----------|-----|-----|-----|
| Horizon T | 5000 | 8000 | 12000 |
| N_off grid (exp1) | 0–8000 | 0–15000 | 0–30000 |
| N_off fixed (exp2) | 1000 | 2000 | 4000 |
| Game seeds | 20 | 20 | 20 |
| Random seeds | 24 | 24 | 24 |
| EXP3 γ | 0.01 | 0.01 | 0.01 |
| Confidence δ | 0.05 | 0.05 | 0.05 |
