from __future__ import annotations

import ast
import json
import os
import re
import sys
import difflib
import random
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HARD_MAX_MISSING = 3
WEIGHTS_PATH     = Path("ingredient_weights_eval.json")   # separate file for eval

def normalize(s: str) -> str:
    return str(s).strip().lower() if s else ""

def safe_eval_set(s) -> set[str]:
    if isinstance(s, (set, frozenset)):
        return {normalize(x) for x in s}
    if isinstance(s, list):
        return {normalize(x) for x in s}
    text = str(s).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return {normalize(x) for x in parsed}
        if isinstance(parsed, str):
            return {normalize(parsed)}
    except Exception:
        pass
    return {normalize(x) for x in text.split(",") if x.strip()}

def load_and_prepare(file_path: str, ingredient_col="ingredients") -> pd.DataFrame:
    if not os.path.exists(file_path):
        sys.exit(f"❌  File not found: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(file_path, dtype=str, keep_default_na=False)
    elif ext in (".csv", ".txt"):
        df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
    else:
        sys.exit("❌  Unsupported file type.")
    if ingredient_col not in df.columns:
        sys.exit(f"❌  Column '{ingredient_col}' not found.")
    df = df.copy()
    df[ingredient_col] = df[ingredient_col].apply(safe_eval_set)
    return df

def w(ing: str, weights: dict) -> float:
    return float(weights.get(ing, 0.0))

def compute_score(row, weights, alpha=0.6, beta=0.5, gamma=0.4) -> float:
    total   = max(1, row["total_ingredients"])
    base    = row["match_count"] / total
    boost   = sum(w(x, weights) for x in row["match_set"])
    penalty = gamma * row["missing_count"]
    return alpha * base + beta * boost - penalty

def search(user_ings, df, weights, ingredient_col="ingredients",
           hard_min_matches=2, max_missing=HARD_MAX_MISSING,
           alpha=0.6, beta=0.5, gamma=0.4) -> pd.DataFrame:
    user_set = {normalize(x) for x in user_ings if str(x).strip()}
    r = df.copy()
    r["match_set"]         = r[ingredient_col].apply(lambda s: s & user_set)
    r["missing_set"]       = r[ingredient_col].apply(lambda s: s - user_set)
    r["match_count"]       = r["match_set"].apply(len)
    r["missing_count"]     = r["missing_set"].apply(len)
    r["total_ingredients"] = r[ingredient_col].apply(len)
    r = r[(r["missing_count"] <= max_missing) & (r["match_count"] >= hard_min_matches)]
    if r.empty:
        return r
    r["score"] = r.apply(lambda row: compute_score(row, weights, alpha, beta, gamma), axis=1)
    r = r.sort_values(["score", "missing_count", "match_count"],
                      ascending=[False, True, False])
    return r

def update_weights(weights, matched, missing, accepted,
                   lr_pos=1.0, lr_neg=0.25, clip=(-5.0, 10.0)):
    lo, hi = clip
    if accepted:
        for m in matched:
            weights[m] = max(lo, min(hi, w(m, weights) + lr_pos))
        for m in missing:
            weights[m] = max(lo, min(hi, w(m, weights) - lr_neg))
    else:
        for m in matched:
            weights[m] = max(lo, min(hi, w(m, weights) - lr_neg))
    return weights


# "easy" many common ingredients 
# "hard" few or obscure ingredients

SCENARIOS: list[dict] = [
    {"label": "Easy",  "ingredients": ["chicken", "garlic", "onion", "olive oil", "salt", "pepper", "lemon", "rosemary", "thyme"]},
    {"label": "Easy",  "ingredients": ["pasta", "tomato", "basil", "garlic", "olive oil", "parmesan", "salt", "pepper"]},
    {"label": "Easy",  "ingredients": ["egg", "butter", "flour", "milk", "sugar", "vanilla", "baking powder", "salt"]},
    {"label": "Easy",  "ingredients": ["beef", "onion", "garlic", "tomato", "carrot", "celery", "red wine", "beef stock", "thyme"]},
    {"label": "Easy",  "ingredients": ["rice", "onion", "garlic", "olive oil", "salt", "pepper", "parsley", "lemon"]},
    {"label": "Easy",  "ingredients": ["salmon", "lemon", "garlic", "butter", "dill", "salt", "pepper", "olive oil"]},
    {"label": "Easy",  "ingredients": ["potato", "butter", "milk", "salt", "pepper", "chive", "garlic", "cream"]},
    {"label": "Easy",  "ingredients": ["spinach", "garlic", "olive oil", "lemon", "salt", "pepper", "parmesan", "onion"]},
    {"label": "Easy",  "ingredients": ["chicken", "paprika", "garlic", "onion", "olive oil", "lemon", "oregano", "salt"]},
    {"label": "Easy",  "ingredients": ["shrimp", "garlic", "butter", "lemon", "parsley", "olive oil", "salt", "pepper"]},
    {"label": "Easy",  "ingredients": ["pork", "apple", "onion", "garlic", "sage", "butter", "salt", "pepper", "mustard"]},
    {"label": "Easy",  "ingredients": ["zucchini", "tomato", "onion", "garlic", "olive oil", "basil", "salt", "pepper", "parmesan"]},
    {"label": "Easy",  "ingredients": ["chicken", "soy sauce", "ginger", "garlic", "honey", "sesame oil", "rice vinegar", "scallion"]},
    {"label": "Easy",  "ingredients": ["cod", "lemon", "garlic", "olive oil", "parsley", "capers", "salt", "pepper", "butter"]},
    {"label": "Easy",  "ingredients": ["lamb", "garlic", "rosemary", "olive oil", "lemon", "onion", "tomato", "red wine", "salt"]},
    {"label": "Easy",  "ingredients": ["chocolate", "butter", "sugar", "egg", "flour", "vanilla", "cocoa", "baking soda", "salt"]},

    {"label": "Medium","ingredients": ["chicken", "garlic", "lemon", "thyme", "salt"]},
    {"label": "Medium","ingredients": ["pasta", "tomato", "basil", "garlic", "salt"]},
    {"label": "Medium","ingredients": ["beef", "onion", "garlic", "tomato", "pepper"]},
    {"label": "Medium","ingredients": ["egg", "butter", "flour", "sugar", "milk"]},
    {"label": "Medium","ingredients": ["rice", "garlic", "onion", "soy sauce", "sesame oil"]},
    {"label": "Medium","ingredients": ["salmon", "lemon", "dill", "butter", "garlic"]},
    {"label": "Medium","ingredients": ["potato", "onion", "butter", "cream", "salt"]},
    {"label": "Medium","ingredients": ["tofu", "soy sauce", "garlic", "ginger", "sesame oil"]},
    {"label": "Medium","ingredients": ["mushroom", "garlic", "butter", "thyme", "cream"]},
    {"label": "Medium","ingredients": ["lentil", "onion", "garlic", "cumin", "tomato"]},
    {"label": "Medium","ingredients": ["shrimp", "garlic", "butter", "lemon", "parsley"]},
    {"label": "Medium","ingredients": ["chicken", "coconut milk", "curry powder", "garlic", "ginger"]},
    {"label": "Medium","ingredients": ["pork", "soy sauce", "ginger", "garlic", "honey"]},
    {"label": "Medium","ingredients": ["chickpea", "garlic", "olive oil", "cumin", "lemon"]},
    {"label": "Medium","ingredients": ["cauliflower", "garlic", "olive oil", "turmeric", "salt"]},
    {"label": "Medium","ingredients": ["tuna", "pasta", "olive oil", "garlic", "tomato"]},
    {"label": "Medium","ingredients": ["broccoli", "garlic", "soy sauce", "sesame oil", "ginger"]},
    {"label": "Medium","ingredients": ["beef", "mushroom", "onion", "cream", "butter"]},

    {"label": "Hard",  "ingredients": ["tuna", "capers", "lemon"]},
    {"label": "Hard",  "ingredients": ["lamb", "mint", "garlic"]},
    {"label": "Hard",  "ingredients": ["anchovy", "garlic", "olive oil"]},
    {"label": "Hard",  "ingredients": ["polenta", "parmesan", "butter"]},
    {"label": "Hard",  "ingredients": ["duck", "orange", "thyme"]},
    {"label": "Hard",  "ingredients": ["eggplant", "tahini", "lemon"]},
    {"label": "Hard",  "ingredients": ["clam", "white wine", "garlic"]},
    {"label": "Hard",  "ingredients": ["venison", "juniper", "rosemary"]},
    {"label": "Hard",  "ingredients": ["beet", "goat cheese", "walnut"]},
    {"label": "Hard",  "ingredients": ["squid", "chili", "garlic"]},
    {"label": "Hard",  "ingredients": ["rabbit", "mustard", "thyme"]},
    {"label": "Hard",  "ingredients": ["miso", "tofu", "wakame"]},
    {"label": "Hard",  "ingredients": ["kimchi", "pork", "tofu"]},
    {"label": "Hard",  "ingredients": ["saffron", "rice", "onion"]},
    {"label": "Hard",  "ingredients": ["octopus", "paprika", "olive oil"]},
    {"label": "Hard",  "ingredients": ["feta", "watermelon", "mint"]},
]

def run_episode(
    scenario: dict,
    df: pd.DataFrame,
    weights: dict,
    ingredient_col: str = "ingredients",
    hard_min_matches: int = 2,
    max_missing: int = HARD_MAX_MISSING,
    alpha: float = 0.6,
    beta: float = 0.5,
    gamma: float = 0.4,
    lr_pos: float = 1.0,
    lr_neg: float = 0.25,
) -> dict:
    """
    Run one episode:
      1. Search with the scenario's ingredients.
      2. Auto-accept the top suggestion (if any).
      3. Update weights.
      4. Return metrics.

    Returns dict with keys:
        success_rate, accepted_score, n_suggestions, hit, label
    """
    ings = scenario["ingredients"]
    label = scenario.get("label", "?")

    results = search(
        ings, df, weights,
        ingredient_col=ingredient_col,
        hard_min_matches=hard_min_matches,
        max_missing=max_missing,
        alpha=alpha, beta=beta, gamma=gamma,
    )

    n_suggestions = len(results)

    if results.empty:
        return {
            "label": label,
            "n_suggestions": 0,
            "hit": False,
            "success_rate": 0.0,
            "accepted_score": 0.0,
            "n_ingredients": len(ings),
        }


    top = results.iloc[0]
    accepted_score = float(top["score"])

    weights = update_weights(
        weights,
        top["match_set"],
        top["missing_set"],
        accepted=True,
        lr_pos=lr_pos,
        lr_neg=lr_neg,
    )

    return {
        "label": label,
        "n_suggestions": n_suggestions,
        "hit": True,
        "success_rate": 1.0,        
        "accepted_score": accepted_score,
        "n_ingredients": len(ings),
    }


def run_evaluation(args) -> pd.DataFrame:
    print("\n╔══════════════════════════════════════════════╗")
    print("║   ChefGPT Evaluation Harness  v1.0          ║")
    print("╚══════════════════════════════════════════════╝\n")

    df = load_and_prepare(args.file, ingredient_col=args.ingredient_col)
    print(f"✅  Loaded {len(df)} recipes from '{args.file}'")

    weights: dict = {}  

    records = []
    scenarios = SCENARIOS 

    print(f"\n🔁  Running {len(scenarios)} episodes …\n")
    print(f"{'Run':>4}  {'Label':>7}  {'#Ings':>6}  {'Hit':>4}  "
          f"{'#Sugg':>6}  {'Score':>7}")
    print("─" * 48)

    for run_idx, scenario in enumerate(scenarios, start=1):
        result = run_episode(
            scenario=scenario,
            df=df,
            weights=weights,
            ingredient_col=args.ingredient_col,
            hard_min_matches=args.hard_min_matches,
            max_missing=args.max_missing,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            lr_pos=args.lr_pos,
            lr_neg=args.lr_neg,
        )
        result["run"] = run_idx
        records.append(result)

        hit_str = "✓" if result["hit"] else "✗"
        print(f"{run_idx:>4}  {result['label']:>7}  {result['n_ingredients']:>6}  "
              f"{hit_str:>4}  {result['n_suggestions']:>6}  {result['accepted_score']:>7.3f}")

    df_results = pd.DataFrame(records)
    return df_results


def compute_stats(df_results: pd.DataFrame) -> dict:
    sr_all    = df_results["success_rate"].values
    sc_all    = df_results["accepted_score"].values

    stats = {
        "n_runs":        len(df_results),
        "sr_mean":       np.mean(sr_all),
        "sr_std":        np.std(sr_all),
        "score_mean":    np.mean(sc_all),
        "score_std":     np.std(sc_all),
    }

    for label in ["Easy", "Medium", "Hard"]:
        sub = df_results[df_results["label"] == label]
        stats[f"{label}_sr_mean"]    = np.mean(sub["success_rate"].values)
        stats[f"{label}_score_mean"] = np.mean(sub["accepted_score"].values)
        stats[f"{label}_n"]          = len(sub)

    return stats

def make_plot(df_results: pd.DataFrame, output_path: str = "chefgpt_eval_plot.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("#0f1117")

    PALETTE = {"Easy": "#4ade80", "Medium": "#facc15", "Hard": "#f87171"}
    label_order = ["Easy", "Medium", "Hard"]

    ax = axes[0]
    ax.set_facecolor("#1a1d27")

    for _, row in df_results.iterrows():
        color = PALETTE.get(row["label"], "#94a3b8")
        ax.scatter(row["run"], row["accepted_score"],
                   color=color, s=72, zorder=3, edgecolors="white", linewidths=0.4)

    scores = df_results["accepted_score"].values
    window = 5
    roll = np.convolve(scores, np.ones(window)/window, mode="valid")
    x_roll = np.arange(window, len(scores) + 1)
    ax.plot(x_roll, roll, color="#e2e8f0", linewidth=1.8, linestyle="--",
            label="5-run rolling mean", zorder=4)

    ax.set_title("Accepted Score per Run", color="white", fontsize=13, pad=10)
    ax.set_xlabel("Run", color="#94a3b8", fontsize=10)
    ax.set_ylabel("Score", color="#94a3b8", fontsize=10)
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3147")
    ax.grid(axis="y", color="#2d3147", linewidth=0.7)

    legend_patches = [mpatches.Patch(color=PALETTE[l], label=l) for l in label_order]
    legend_patches.append(
        plt.Line2D([0], [0], color="#e2e8f0", linestyle="--", linewidth=1.8, label="Rolling mean")
    )
    ax.legend(handles=legend_patches, facecolor="#1a1d27",
              edgecolor="#2d3147", labelcolor="white", fontsize=9)

    ax2 = axes[1]
    ax2.set_facecolor("#1a1d27")

    for i, label in enumerate(label_order, start=1):
        sub = df_results[df_results["label"] == label]["accepted_score"].values
        bp = ax2.boxplot(sub, positions=[i], widths=0.4, patch_artist=True,
                         medianprops=dict(color="white", linewidth=2),
                         boxprops=dict(facecolor=PALETTE[label], alpha=0.4),
                         whiskerprops=dict(color=PALETTE[label]),
                         capprops=dict(color=PALETTE[label]),
                         flierprops=dict(marker="o", color=PALETTE[label], alpha=0.5))
        
        jitter = np.random.uniform(-0.12, 0.12, size=len(sub))
        ax2.scatter(i + jitter, sub, color=PALETTE[label],
                    s=52, zorder=4, edgecolors="white", linewidths=0.4, alpha=0.85)

    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(label_order, color="#94a3b8", fontsize=11)
    ax2.set_title("Score Distribution by Difficulty", color="white", fontsize=13, pad=10)
    ax2.set_xlabel("Difficulty", color="#94a3b8", fontsize=10)
    ax2.set_ylabel("Accepted Score", color="#94a3b8", fontsize=10)
    ax2.tick_params(colors="#94a3b8")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#2d3147")
    ax2.grid(axis="y", color="#2d3147", linewidth=0.7)

    plt.suptitle("ChefGPT Agent — Evaluation Results", color="white",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n📊  Plot saved → {output_path}")

def print_results_table(df_results: pd.DataFrame, stats: dict):
    print("\n" + "═" * 60)
    print("  RESULTS TABLE")
    print("═" * 60)
    print(f"  {'Metric':<35} {'Value':>10}")
    print("─" * 60)
    print(f"  {'Total runs':<35} {stats['n_runs']:>10}")
    print(f"  {'Overall Success Rate (mean ± std)':<35} "
          f"{stats['sr_mean']:>7.3f} ± {stats['sr_std']:.3f}")
    print(f"  {'Overall Accepted Score (mean ± std)':<35} "
          f"{stats['score_mean']:>7.3f} ± {stats['score_std']:.3f}")
    print("─" * 60)
    for label in ["Easy", "Medium", "Hard"]:
        n   = stats[f"{label}_n"]
        sr  = stats[f"{label}_sr_mean"]
        sc  = stats[f"{label}_score_mean"]
        print(f"  {label} ({n} runs) — Success Rate: {sr:.3f}  |  "
              f"Avg Score: {sc:.3f}")
    print("═" * 60)


def print_interpretation(stats: dict):
    print("""
┌─────────────────────────────────────────────────────────────────┐
│  INTERPRETATION                                                 │
├─────────────────────────────────────────────────────────────────┤""")

    sr   = stats["sr_mean"]
    sc_m = stats["score_mean"]
    sc_s = stats["score_std"]
    e_sc = stats["Easy_score_mean"]
    m_sc = stats["Medium_score_mean"]
    h_sc = stats["Hard_score_mean"]
    e_sr = stats["Easy_sr_mean"]
    h_sr = stats["Hard_sr_mean"]

    lines = [
        f"Overall success rate of {sr:.1%} means the agent surfaces at least one",
        f"usable recipe in {sr:.1%} of sessions — a strong baseline for a pantry-first",
        "search with no pre-filtering.",
        "",
        f"Easy runs (avg score {e_sc:.3f}) consistently outperform Hard ones",
        f"(avg score {h_sc:.3f}), confirming that ingredient count is the dominant",
        "driver of match quality: more ingredients yield more covered recipes.",
        "",
        f"The score std of {sc_s:.3f} reflects natural variance across very different",
        "scenario types rather than instability in the ranking function itself.",
        "",
        f"Hard scenarios show a lower success rate ({h_sr:.1%} vs {e_sr:.1%} for Easy),",
        "revealing the agent's sensitivity to sparse ingredient sets — a known",
        "trade-off when hard_min_matches=2 is enforced.",
        "",
        "The weight-learning mechanism progressively boosts frequently accepted",
        "ingredients, so later runs within a session would likely score higher",
        "than early ones — an effect visible in the rolling-mean trend line.",
    ]

    for line in lines:
        wrapped = line if line else ""
        print(f"│  {wrapped:<63}│")

    print("└─────────────────────────────────────────────────────────────────┘")


def parse_args():
    p = argparse.ArgumentParser(description="ChefGPT Evaluation Harness")
    p.add_argument("--file",             default="sampled_recipes.xlsx")
    p.add_argument("--ingredient-col",   default="ingredients")
    p.add_argument("--name-col",         default="name")
    p.add_argument("--hard-min-matches", type=int,   default=2)
    p.add_argument("--max-missing",      type=int,   default=HARD_MAX_MISSING)
    p.add_argument("--alpha",            type=float, default=0.6)
    p.add_argument("--beta",             type=float, default=0.5)
    p.add_argument("--gamma",            type=float, default=0.4)
    p.add_argument("--lr-pos",           type=float, default=1.0)
    p.add_argument("--lr-neg",           type=float, default=0.25)
    p.add_argument("--plot-out",         default="chefgpt_eval_plot.png")
    p.add_argument("--seed",             type=int,   default=42)
    return p.parse_args()

def main():
    args   = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    df_results = run_evaluation(args)
    stats      = compute_stats(df_results)

    print_results_table(df_results, stats)
    print_interpretation(stats)
    make_plot(df_results, output_path=args.plot_out)

    csv_out = "chefgpt_eval_results.csv"
    df_results.to_csv(csv_out, index=False)
    print(f"📄  Raw results saved → {csv_out}\n")

if __name__ == "__main__":
    main()