#!/usr/bin/env python3
"""
Track B evaluation for ChefGPT using:
- N = 50 episodes
- 5 random seeds
- 2 settings: Easy and Hard
- Metrics:
    1. Success Rate
    2. Average Score
    3. Failure Rate

"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import random
import re
import statistics
import sys
from pathlib import Path

import pandas as pd

SEEDS = [11, 22, 33, 44, 55]
EPISODES_PER_SEED_PER_SETTING = 5
TOTAL_EPISODES = len(SEEDS) * 2 * EPISODES_PER_SEED_PER_SETTING  # 50


def safe_eval_list(value):
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    return [x.strip() for x in str(value).split(",") if x.strip()]


def load_chefgpt_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("chefgpt2_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["chefgpt2_module"] = module
    spec.loader.exec_module(module)
    return module


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def same_recipe(row, source_id: str, source_name: str) -> bool:
    row_id = normalize_text(row.get("id", ""))
    row_name = normalize_text(row.get("name", ""))
    source_id = normalize_text(source_id)
    source_name = normalize_text(source_name)

    if source_id and row_id and row_id == source_id:
        return True
    if source_name and row_name and row_name == source_name:
        return True
    return False


def build_query(ingredients, hard=False, rng=None):
    """
    Easy:
        Keep most ingredients so the source recipe has a fair chance to pass max_missing=2.
    Hard:
        Create sparse / noisy / vague queries.
    """
    rng = rng or random
    ings = [str(i).strip() for i in ingredients if str(i).strip()]
    if not ings:
        return ""

    if not hard:
        if len(ings) <= 4:
            chosen = ings[:]
        else:
            drop_n = rng.randint(0, min(2, len(ings) - 2))
            keep_n = max(2, len(ings) - drop_n)
            chosen = rng.sample(ings, keep_n)
        return ", ".join(chosen)

    keep_n = min(len(ings), rng.randint(2, min(4, max(2, len(ings)))))
    chosen = rng.sample(ings, keep_n)
    mode = rng.choice(["plain", "typo", "filter", "sparse", "vague"])

    query = ", ".join(chosen)

    if mode == "typo":
        def typo_word(word: str) -> str:
            word = word.strip()
            if len(word) > 5:
                return word[:-1]
            return word
        query = ", ".join(typo_word(x) for x in chosen)

    elif mode == "filter":
        prefix = rng.choice(["vegan", "spicy", "italian", "indian", "quick"])
        query = f"{prefix}, {query}"

    elif mode == "sparse":
        query = ", ".join(chosen[:2])

    elif mode == "vague":
        query = "something with " + ", ".join(chosen[:2])

    return query


def generate_episodes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    episode_no = 1

    for seed in SEEDS:
        easy_rng = random.Random(seed)
        hard_rng = random.Random(seed + 1000)

        easy_df = df.sample(n=EPISODES_PER_SEED_PER_SETTING, random_state=seed).reset_index(drop=True)
        hard_df = df.sample(n=EPISODES_PER_SEED_PER_SETTING, random_state=seed + 500).reset_index(drop=True)

        for _, row in easy_df.iterrows():
            ingredients = safe_eval_list(row["ingredients"])
            rows.append({
                "Episode": episode_no,
                "Seed": seed,
                "Setting": "Easy",
                "Source Recipe ID": str(row.get("id", "")),
                "Source Recipe Name": str(row.get("name", "")),
                "Query": build_query(ingredients, hard=False, rng=easy_rng),
                "max_missing": 2,
            })
            episode_no += 1

        for _, row in hard_df.iterrows():
            ingredients = safe_eval_list(row["ingredients"])
            rows.append({
                "Episode": episode_no,
                "Seed": seed,
                "Setting": "Hard",
                "Source Recipe ID": str(row.get("id", "")),
                "Source Recipe Name": str(row.get("name", "")),
                "Query": build_query(ingredients, hard=True, rng=hard_rng),
                "max_missing": 2,
            })
            episode_no += 1

    return pd.DataFrame(rows)


def evaluate_episode(mod, df, dataset_phrases, spell, weights, episode_row, hard_min_matches=2, snap_cutoff=0.86):
    query_text = str(episode_row["Query"])
    source_recipe_id = str(episode_row["Source Recipe ID"])
    source_recipe_name = str(episode_row["Source Recipe Name"])
    max_missing = int(episode_row["max_missing"])

    intent = mod.extract_intent(query_text)
    resolved_ings, corrections = mod.resolve_ingredients(
        intent["raw_ingredients"], spell, dataset_phrases, snap_cutoff
    )

    results = mod.search(
        resolved_ings,
        df,
        weights,
        ingredient_col="ingredients",
        hard_min_matches=hard_min_matches,
        max_missing=max_missing,
        dietary=intent["dietary"],
        cuisine=intent["cuisine"],
        alpha=0.6,
        beta=0.5,
        gamma=0.4,
    ).reset_index(drop=True)

    success = 0
    accepted_score = None
    no_result = 1 if results.empty else 0
    failure = no_result
    matched_recipe = ""

    if not results.empty:
        for _, row in results.iterrows():
            if same_recipe(row, source_recipe_id, source_recipe_name):
                success = 1
                accepted_score = float(row["score"])
                matched_recipe = str(row.get("name", ""))
                break

    return {
        "Episode": int(episode_row["Episode"]),
        "Seed": int(episode_row["Seed"]),
        "Setting": str(episode_row["Setting"]),
        "Source Recipe ID": source_recipe_id,
        "Source Recipe Name": source_recipe_name,
        "Query": query_text,
        "Resolved Ingredients": ", ".join(resolved_ings),
        "Corrections": " | ".join(corrections),
        "Success": int(success),
        "Accepted Score": accepted_score,
        "Failure": int(failure),
        "No Result": int(no_result),
        "Returned Results": int(len(results)),
        "Matched Recipe": matched_recipe,
    }


def mean_std(values):
    clean = [float(v) for v in values if v is not None and not pd.isna(v)]
    if not clean:
        return 0.0, 0.0
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)


def summarise_group(df_group: pd.DataFrame):
    per_seed = []

    for seed in sorted(df_group["Seed"].unique()):
        seed_df = df_group[df_group["Seed"] == seed].copy()

        success_rate = float(seed_df["Success"].mean()) if len(seed_df) else 0.0
        failure_rate = float(seed_df["Failure"].mean()) if len(seed_df) else 0.0

        score_values = seed_df.loc[seed_df["Success"] == 1, "Accepted Score"].dropna().tolist()
        avg_score = float(sum(score_values) / len(score_values)) if score_values else 0.0

        per_seed.append({
            "Seed": seed,
            "Success Rate": success_rate,
            "Average Score": avg_score,
            "Failure Rate": failure_rate,
            "Episodes": len(seed_df),
        })

    per_seed_df = pd.DataFrame(per_seed)

    sr_mean, sr_std = mean_std(per_seed_df["Success Rate"].tolist())
    score_mean, score_std = mean_std(per_seed_df["Average Score"].tolist())
    fr_mean, fr_std = mean_std(per_seed_df["Failure Rate"].tolist())

    return {
        "episodes": int(len(df_group)),
        "n_seeds": int(per_seed_df["Seed"].nunique()),
        "success_mean": sr_mean,
        "success_std": sr_std,
        "score_mean": score_mean,
        "score_std": score_std,
        "failure_mean": fr_mean,
        "failure_std": fr_std,
        "per_seed_df": per_seed_df,
    }


def print_summary(title: str, summary: dict):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Episodes         : {summary['episodes']}")
    print(f"Seeds            : {summary['n_seeds']}")
    print(
        f"Success Rate     : {summary['success_mean']:.4f} ± {summary['success_std']:.4f} "
        f"({summary['success_mean']*100:.2f}% ± {summary['success_std']*100:.2f}%)"
    )
    print(f"Average Score    : {summary['score_mean']:.4f} ± {summary['score_std']:.4f}")
    print(
        f"Failure Rate     : {summary['failure_mean']:.4f} ± {summary['failure_std']:.4f} "
        f"({summary['failure_mean']*100:.2f}% ± {summary['failure_std']*100:.2f}%)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run N=50 Track B evaluation for ChefGPT2.0 using Success Rate, Average Score, and Failure Rate."
    )
    parser.add_argument("--module", default="ChefGPT2.0.py")
    parser.add_argument("--recipes", default="sampled_recipes.xlsx")
    parser.add_argument("--dict-path", default="trained_dict.txt")
    parser.add_argument("--weights", default="ingredient_weights.json")
    parser.add_argument("--snap-cutoff", type=float, default=0.86)
    parser.add_argument("--hard-min-matches", type=int, default=2)
    parser.add_argument("--show-episodes", action="store_true", help="Print episode-level results")
    parser.add_argument("--show-per-seed", action="store_true", help="Print per-seed summaries")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    module_path = base / args.module
    recipes_path = base / args.recipes
    dict_path = base / args.dict_path
    weights_path = base / args.weights

    if not module_path.exists():
        raise FileNotFoundError(f"Missing file: {module_path}")
    if not recipes_path.exists():
        raise FileNotFoundError(f"Missing file: {recipes_path}")
    if not dict_path.exists():
        raise FileNotFoundError(f"Missing file: {dict_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing file: {weights_path}")

    mod = load_chefgpt_module(module_path)

    prepared_df = mod.load_and_prepare(str(recipes_path), ingredient_col="ingredients")
    raw_df = pd.read_excel(recipes_path, dtype=str, keep_default_na=False)

    dataset_phrases, _ = mod.build_dataset_vocab(prepared_df, "ingredients")
    dataset_words = mod.load_dataset_words(dict_path)
    spell = mod.init_spell(dataset_words)
    weights = mod.load_weights(weights_path)

    episodes_df = generate_episodes(raw_df)

    if len(episodes_df) != TOTAL_EPISODES:
        raise RuntimeError(f"Expected {TOTAL_EPISODES} episodes, got {len(episodes_df)}")

    results = []
    for _, ep in episodes_df.iterrows():
        result = evaluate_episode(
            mod=mod,
            df=prepared_df,
            dataset_phrases=dataset_phrases,
            spell=spell,
            weights=weights,
            episode_row=ep,
            hard_min_matches=args.hard_min_matches,
            snap_cutoff=args.snap_cutoff,
        )
        results.append(result)

    results_df = pd.DataFrame(results)

    if args.show_episodes:
        print("\nEPISODE RESULTS")
        print("---------------")
        for _, row in results_df.iterrows():
            score_display = "-" if pd.isna(row["Accepted Score"]) else f"{float(row['Accepted Score']):.2f}"
            print(
                f"Ep {int(row['Episode']):02d} | "
                f"Seed={int(row['Seed'])} | "
                f"{row['Setting']:<4} | "
                f"Success={int(row['Success'])} | "
                f"Score={score_display} | "
                f"Failure={int(row['Failure'])} | "
                f"Query={row['Query']}"
            )

    overall_summary = summarise_group(results_df)
    easy_summary = summarise_group(results_df[results_df["Setting"] == "Easy"].copy())
    hard_summary = summarise_group(results_df[results_df["Setting"] == "Hard"].copy())

    print("\nTRACK B EVALUATION SUMMARY")
    print("==========================")
    print(f"Total episodes: {TOTAL_EPISODES}")
    print(f"Seeds used    : {SEEDS}")
    print(f"Settings      : Easy, Hard")
    print(f"Episodes/seed : {EPISODES_PER_SEED_PER_SETTING} per setting")

    print_summary("OVERALL SUMMARY", overall_summary)
    print_summary("EASY SUMMARY", easy_summary)
    print_summary("HARD SUMMARY", hard_summary)

    if args.show_per_seed:
        print("\nPER-SEED OVERALL METRICS")
        print("------------------------")
        overall_per_seed = overall_summary["per_seed_df"].copy()
        for _, row in overall_per_seed.iterrows():
            print(
                f"Seed {int(row['Seed'])}: "
                f"Success Rate={float(row['Success Rate']):.4f} ({float(row['Success Rate'])*100:.2f}%), "
                f"Average Score={float(row['Average Score']):.4f}, "
                f"Failure Rate={float(row['Failure Rate']):.4f} ({float(row['Failure Rate'])*100:.2f}%)"
            )

    print("\nMetric definitions")
    print("------------------")
    print("Success Rate = proportion of episodes where the source recipe was retrieved")
    print("Average Score = mean score of successful retrieved source recipes")
    print("Failure Rate = proportion of episodes where no result was returned")
    print("\nReporting style")
    print("---------------")
    print("Mean ± std is computed across the 5 seeds.")


if __name__ == "__main__":
    main()