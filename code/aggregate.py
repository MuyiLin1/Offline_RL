"""Aggregate per-game exp1/exp2 summaries into combined summary + plots without re-running.

Plotting style matches run_experiments.py exactly.
"""
import json, os, sys, numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Style constants (from run_experiments.py)
# --------------------------------------------------------------------------
COLORS = {
    "baseline": "#C73E1D", "hybrid": "#2E86AB", "gated": "#9B59B6",
    "good": "#3A7D44", "poor": "#A23B72", "neutral": "#7D8590",
    "tabular": "#6B4E71", "contextual": "#2A9D8F",
}
ETC_COLOR = "#F4A261"


def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def ci_mean(data, axis=0, z=1.96):
    data = np.array(data, dtype=float)
    m = np.nanmean(data, axis=axis)
    n = np.sum(np.isfinite(data), axis=axis).astype(float)
    n = np.maximum(n, 1.0)
    s = np.nanstd(data, axis=axis, ddof=1)
    s = np.where(np.isfinite(s), s, 0.0)
    return m, z * s / np.sqrt(n)


# --------------------------------------------------------------------------
# Experiment 1
# --------------------------------------------------------------------------
def aggregate_exp1(out_dir, game_seeds, data_quality="good"):
    all_results = []
    for gs in game_seeds:
        path = os.path.join(out_dir, f"game_{gs}", "exp1_summary.json")
        if not os.path.exists(path):
            print(f"WARNING: missing {path}, skipping", file=sys.stderr)
            continue
        with open(path) as f:
            all_results.append(json.load(f))
    if not all_results:
        print("No exp1 results found", file=sys.stderr)
        return

    avg = {"n_off": all_results[0]["n_off"], "game_seeds": list(game_seeds),
           "data_quality": data_quality, "num_games": len(all_results)}
    for key in ["tfw_baseline_mean", "tfw_hybrid_mean", "tfw_gated_mean", "tfw_etc_mean",
                "convergence_baseline_mean", "convergence_hybrid_mean",
                "theorem1_offline_transfer_rate_gated_mean",
                "offline_m0_nonempty_rate_gated_mean"]:
        vals = [r[key] for r in all_results if key in r]
        if vals:
            avg[key] = np.nanmean(vals, axis=0).tolist()
            avg[key + "_per_game"] = [v if isinstance(v, list) else [v] for v in vals]

    with open(os.path.join(out_dir, "exp1_summary.json"), "w") as f:
        json.dump(avg, f, indent=2)
    print(f"Wrote {out_dir}/exp1_summary.json ({len(all_results)} games)")

    # --- Plot (matching run_experiments.py style) ---
    n_off = avg["n_off"]
    num_games = len(all_results)
    x_labels = [str(n) for n in n_off]
    # Sqrt-proportional spacing: compromise between linear (too tight at left)
    # and log (too spread at right). Gives proportional feel but readable.
    raw = np.array(n_off, dtype=float)
    x_plot = np.sqrt(raw)  # sqrt transform for spacing
    # Normalize to [0, 10] range for nice axis
    if x_plot[-1] > 0:
        x_plot = x_plot / x_plot[-1] * 10.0

    # Gather per-game arrays: shape [num_games, n_points]
    tfw_base = np.array(avg.get("tfw_baseline_mean_per_game", []))
    tfw_hyb = np.array(avg.get("tfw_hybrid_mean_per_game", []))
    tfw_gated = np.array(avg.get("tfw_gated_mean_per_game", []))
    tfw_etc = np.array(avg.get("tfw_etc_mean_per_game", []))

    mb, eb = ci_mean(tfw_base)
    mh, eh = ci_mean(tfw_hyb)
    mg, eg = ci_mean(tfw_gated)
    metc, eetc = ci_mean(tfw_etc)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.grid(True, alpha=0.3)

    ax.plot(x_plot, mb, "o-", color=COLORS["baseline"], lw=3, ms=8, label="FMUCB (online only)")
    ax.fill_between(x_plot, mb - eb, mb + eb, color=COLORS["baseline"], alpha=0.18)
    ax.plot(x_plot, mg, "^--", color=COLORS["gated"], lw=2.5, ms=7, label="Gated-FMUCB")
    ax.fill_between(x_plot, mg - eg, mg + eg, color=COLORS["gated"], alpha=0.15)
    ax.plot(x_plot, metc, "v:", color=ETC_COLOR, lw=2.5, ms=7, label="Explore-then-commit")
    ax.fill_between(x_plot, metc - eetc, metc + eetc, color=ETC_COLOR, alpha=0.15)
    ax.plot(x_plot, mh, "s-", color=COLORS["hybrid"], lw=3, ms=8, label="Hybrid-FMUCB")
    ax.fill_between(x_plot, mh - eh, mh + eh, color=COLORS["hybrid"], alpha=0.18)

    ax.set_xticks(x_plot)
    ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.set_xlabel(r"Offline dataset size $N_{\mathrm{off}}$", fontsize=16)
    ax.set_ylabel(r"$T_{f,w}$ (mistakes vs.\ true best manipulation $F^{fm}$)", fontsize=16)
    ax.set_title(rf"Experiment 1: Offline data size vs $T_{{f,w}}$ (good-coverage data, {num_games} games avg)")
    ax.legend(frameon=True, fancybox=True, shadow=True)
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp1_noff_vs_tfw.png"), bbox_inches="tight")
    # Also save with size-suffixed name
    dirname = os.path.basename(out_dir.rstrip("/"))
    if "6x6" in dirname:
        fig.savefig(os.path.join(out_dir, "exp1_noff_vs_tfw_6x6.png"), bbox_inches="tight")
    elif "8x8" in dirname:
        fig.savefig(os.path.join(out_dir, "exp1_noff_vs_tfw_8x8.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir}/exp1_noff_vs_tfw.png")


# --------------------------------------------------------------------------
# Experiment 2
# --------------------------------------------------------------------------
def aggregate_exp2(out_dir, game_seeds):
    all_results = []
    for gs in game_seeds:
        path = os.path.join(out_dir, f"game_{gs}", "exp2_summary.json")
        if not os.path.exists(path):
            print(f"WARNING: missing {path}, skipping", file=sys.stderr)
            continue
        with open(path) as f:
            all_results.append(json.load(f))
    if not all_results:
        print("No exp2 results found", file=sys.stderr)
        return

    methods = ["hybrid", "gated", "etc", "baseline"]
    kinds = ["good", "neutral", "adversarial"]
    num_games = len(all_results)

    # Build averaged summary
    avg = {"num_games": num_games, "game_seeds": list(game_seeds)}
    for key in all_results[0]:
        if key in ("game_seed", "game_seeds", "num_games"):
            continue
        vals = [r[key] for r in all_results if key in r]
        if vals:
            try:
                avg[key] = np.nanmean(vals, axis=0).tolist()
                avg[key + "_per_game"] = vals
            except Exception:
                avg[key] = vals[0]

    # Get n_off_fixed from first result
    n_off_fixed = all_results[0].get("n_off_fixed", "?")

    with open(os.path.join(out_dir, "exp2_summary.json"), "w") as f:
        json.dump(avg, f, indent=2)
    print(f"Wrote {out_dir}/exp2_summary.json ({num_games} games)")

    # --- Plot (matching run_experiments.py style) ---
    method_labels = ["Hybrid-\nFMUCB", "Gated-\nFMUCB", "Explore-then-\ncommit", "FMUCB\n(no offline)"]
    method_colors = [COLORS["hybrid"], COLORS["gated"], ETC_COLOR, COLORS["baseline"]]
    kind_labels = ["Good (manip. relevant)", "Neutral (uniform)", "Adversarial (sparse manip.)"]

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.5))
    n_methods = len(methods)
    bar_width = 0.2
    xpos = np.arange(n_methods)

    for ax, kind, kind_label in zip(axes, kinds, kind_labels):
        # Gather per-game means for this coverage type
        game_means = []
        for r in all_results:
            game_means.append([r.get(f"tfw_{m}_{kind}_mean", np.nan) for m in methods])
        game_means = np.array(game_means)  # [num_games, n_methods]
        means = np.nanmean(game_means, axis=0)
        if num_games > 1:
            errs = 1.96 * np.nanstd(game_means, axis=0, ddof=1) / np.sqrt(num_games)
        else:
            errs = np.zeros(n_methods)

        bars = ax.bar(xpos, means, width=bar_width * 3.5, yerr=errs,
                      color=method_colors, edgecolor="white",
                      linewidth=1.2, capsize=6, error_kw={"linewidth": 1.5})
        # Place numbers above confidence interval whiskers with clear padding
        y_max = max(m + e for m, e in zip(means, errs) if np.isfinite(m))
        pad = y_max * 0.02  # 2% of max height as padding
        for rect, m_val, e_val in zip(bars, means, errs):
            if np.isfinite(m_val):
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + e_val + pad,
                        f"{m_val:.0f}", ha="center", va="bottom",
                        fontsize=13, fontweight="bold")
        ax.set_xticks(xpos)
        ax.set_xticklabels(method_labels, fontsize=16)
        ax.set_ylabel(r"$T_{f,w}$ (mistakes)", fontsize=16)
        ax.set_title(kind_label)
        _style_axis(ax)

    fig.suptitle(
        rf"Experiment 2: All methods $\times$ data quality ($N_{{\mathrm{{off}}}}={n_off_fixed}$, {num_games} games avg)",
        fontsize=22, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exp2_coverage_bars.png"), bbox_inches="tight")
    # Also save with size-suffixed name for the paper
    dirname = os.path.basename(out_dir.rstrip("/"))
    if "6x6" in dirname:
        fig.savefig(os.path.join(out_dir, "exp2_coverage_bars_6x6.png"), bbox_inches="tight")
    elif "8x8" in dirname:
        fig.savefig(os.path.join(out_dir, "exp2_coverage_bars_8x8.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir}/exp2_coverage_bars.png")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    game_seeds = [737,1482,1603,3043,3197,4098,4356,5003,5466,5621,
                  5954,7184,7466,7657,8001,8400,8671,9011,9134,9153]

    for out_dir in sys.argv[1:]:
        print(f"\n=== Aggregating {out_dir} ===")
        aggregate_exp1(out_dir, game_seeds)
        aggregate_exp2(out_dir, game_seeds)
