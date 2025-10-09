# chef_gpt_greedy_learning.py

import argparse
import ast
import json
import os
from collections import defaultdict, Counter
from typing import Iterable, Dict, Set, Tuple, List

import pandas as pd

DEFAULT_WEIGHTS_PATH = "ingredient_weights.json"
DEFAULT_LOG_PATH = "feedback_log.jsonl"
MAX_MISSING_ALLOWED = 3  # hard-coded

def safe_eval_list(s):
    if s is None or s == "":
        return []
    if isinstance(s, list):
        return s
    try:
        return ast.literal_eval(s)
    except Exception:
        return [item.strip().strip('"\'') for item in str(s).split(",") if item.strip()]

def normalize(s):
    return str(s).strip().lower()

def load_weights(path: str) -> Dict[str, float]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_weights(weights: Dict[str, float], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False, indent=2)

def log_feedback(user_ingredients: Iterable[str], suggested_recipe_id: int, accepted: bool, path: str):
    entry = {
        "user_ingredients": [normalize(x) for x in user_ingredients],
        "suggested_recipe_id": int(suggested_recipe_id),
        "accepted": bool(accepted)
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def load_recipes(file_path: str, ingredient_col: str = "ingredients") -> Tuple[pd.DataFrame, Dict[str, Set[int]]]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path, dtype=str, keep_default_na=False)
    else:
        df = pd.read_csv(file_path, dtype=str, keep_default_na=False)

    if ingredient_col not in df.columns:
        raise ValueError(f"Column '{ingredient_col}' not found. Available: {list(df.columns)}")

    df[ingredient_col] = df[ingredient_col].apply(lambda s: {normalize(x) for x in safe_eval_list(s)})

    inv_index: Dict[str, Set[int]] = defaultdict(set)
    for i, ings in df[ingredient_col].items():
        for ing in ings:
            inv_index[ing].add(int(i))
    return df, inv_index

def score_recipe(recipe_ings: Set[str], user_set: Set[str], weights: Dict[str, float], missing_penalty: float = 0.5) -> float:
    matched = recipe_ings & user_set
    missing = recipe_ings - user_set
    base = len(matched)
    wsum = sum(weights.get(ing, 0.0) for ing in matched)
    return base + wsum - missing_penalty * len(missing)

def get_candidate_recipe_ids(user_set: Set[str], inv_index: Dict[str, Set[int]], max_candidates: int = 5000) -> Set[int]:
    counts = Counter()
    for ing in user_set:
        for rid in inv_index.get(ing, []):
            counts[rid] += 1
    if not counts:
        return set()
    return {rid for rid, _ in counts.most_common(max_candidates)}

def greedy_search_weighted(
    user_ingredients: Iterable[str],
    df: pd.DataFrame,
    inv_index: Dict[str, Set[int]],
    weights: Dict[str, float],
    top_n: int = 5,
    ingredient_col: str = "ingredients",
    missing_penalty: float = 0.5
) -> List[Dict]:
    """
    Rank candidates by:
      1. Fewest missing ingredients first
      2. Highest weighted greedy score next

    Filters out recipes with > MAX_MISSING_ALLOWED missing ingredients.
    """
    user_set = {normalize(x) for x in user_ingredients}
    cand_ids = get_candidate_recipe_ids(user_set, inv_index, max_candidates=5000)
    if not cand_ids:
        cand_ids = set(df.index.tolist())  # fallback if no overlap

    scored: List[Tuple[int, float, int]] = []  # (missing_count, score, recipe_id)
    for rid in cand_ids:
        recipe_ings = df.at[int(rid), ingredient_col]
        missing = recipe_ings - user_set
        if len(missing) > MAX_MISSING_ALLOWED:
            continue
        sc = score_recipe(recipe_ings, user_set, weights, missing_penalty=missing_penalty)
        scored.append((len(missing), sc, rid))

    # Sort: fewer missing ingredients first, then higher score
    scored.sort(key=lambda x: (x[0], -x[1]))

    results = []
    for missing_count, sc, rid in scored[:top_n]:
        row = df.loc[int(rid)]
        recipe_ings = row[ingredient_col]
        results.append({
            "recipe_id": int(rid),
            "name": row.get("name", ""),
            "score": float(sc),
            "missing_count": missing_count,
            "matched": sorted(list(recipe_ings & user_set)),
            "missing": sorted(list(recipe_ings - user_set))
        })
    return results

MIN_WEIGHT = -5.0
MAX_WEIGHT = 10.0
POSITIVE_STEP = 1.0
NEGATIVE_STEP = 0.5

def update_weights_online(
    weights: Dict[str, float],
    df: pd.DataFrame,
    recipe_id: int,
    user_ingredients: Iterable[str],
    accepted: bool,
    ingredient_col: str = "ingredients",
    pos_step: float = POSITIVE_STEP,
    neg_step: float = NEGATIVE_STEP
) -> Dict[str, float]:
    user_set = {normalize(x) for x in user_ingredients}
    recipe_ings = df.at[int(recipe_id), ingredient_col]
    matched = recipe_ings & user_set

    for ing in matched:
        old = weights.get(ing, 0.0)
        new = min(MAX_WEIGHT, old + pos_step) if accepted else max(MIN_WEIGHT, old - neg_step)
        weights[ing] = new
    return weights

def parse_args():
    p = argparse.ArgumentParser(description="ChefGPT Greedy Weighted Search with Online Learning")
    p.add_argument("--file", required=True, help="Path to dataset file (.xlsx or .csv)")
    p.add_argument("--ings", required=False, default="", help="Comma-separated user ingredients (optional; will prompt if empty)")
    p.add_argument("--top", type=int, default=5, help="Number of top results to show")
    p.add_argument("--missing-penalty", type=float, default=0.5, help="Penalty per missing ingredient")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS_PATH, help="Path to weights JSON file")
    p.add_argument("--log", default=DEFAULT_LOG_PATH, help="Path to feedback log (.jsonl)")
    p.add_argument("--accept", type=int, help="Accept the suggestion at this rank (1-based)")
    p.add_argument("--reject", type=int, help="Reject the suggestion at this rank (1-based)")
    return p.parse_args()

def get_user_ingredients_from_terminal(cli_ings: str) -> List[str]:
    raw = cli_ings.strip()
    if not raw:
        raw = input("Enter ingredients (comma-separated): ").strip()
    if not raw:
        raise ValueError("No ingredients provided.")
    # allow either comma-separated or list-like input
    items = safe_eval_list(raw)
    if not items:
        items = [x.strip() for x in raw.split(",") if x.strip()]
    return [normalize(x) for x in items]

def main():
    args = parse_args()

    df, inv = load_recipes(args.file, ingredient_col="ingredients")
    weights = load_weights(args.weights)

    user_ingredients = get_user_ingredients_from_terminal(args.ings)

    print(f"Loaded {len(df)} recipes. Learned weights: {len(weights)} keys.")
    print(f"User ingredients (terminal): {', '.join(user_ingredients)}")
    print(f"Hard limit: at most {MAX_MISSING_ALLOWED} missing ingredients allowed per recipe.")

    results = greedy_search_weighted(
        user_ingredients=user_ingredients,
        df=df,
        inv_index=inv,
        weights=weights,
        top_n=args.top,
        ingredient_col="ingredients",
        missing_penalty=args.missing_penalty
    )

    if not results:
        print("\nNo recipes found within the allowed missing-ingredient threshold.")
        return

    print("\nTop results:")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['name']}  (id={r['recipe_id']}, score={r['score']:.2f})")
        print(f"   ✅ matched: {', '.join(r['matched']) if r['matched'] else '—'}")
        print(f"   ❌ missing: {', '.join(r['missing']) if r['missing'] else 'None'}")

    rank = None
    accepted_flag = None
    if args.accept is not None and args.reject is not None:
        print("\nPlease provide only one of --accept or --reject, not both.")
        return
    elif args.accept is not None:
        rank = args.accept
        accepted_flag = True
    elif args.reject is not None:
        rank = args.reject
        accepted_flag = False

    if accepted_flag is not None:
        if rank < 1 or rank > len(results):
            print(f"\nInvalid rank {rank}. Choose a number between 1 and {len(results)}.")
            return
        chosen = results[rank - 1]
        rid = chosen["recipe_id"]
        print(f"\nRecording feedback: {'ACCEPT' if accepted_flag else 'REJECT'} — {chosen['name']} (id={rid})")

        log_feedback(user_ingredients, rid, accepted_flag, path=args.log)
        weights = update_weights_online(weights, df, rid, user_ingredients, accepted_flag, ingredient_col="ingredients")
        save_weights(weights, path=args.weights)
        print(f"Feedback logged → {args.log}")
        print(f"Weights updated → {args.weights}")

        results2 = greedy_search_weighted(
            user_ingredients=user_ingredients,
            df=df,
            inv_index=inv,
            weights=weights,
            top_n=args.top,
            ingredient_col="ingredients",
            missing_penalty=args.missing_penalty
        )
        print("\nTop results AFTER weight update:")
        for i, r in enumerate(results2, 1):
            print(f"{i}. {r['name']}  (id={r['recipe_id']}, score={r['score']:.2f})")
            print(f"   ✅ matched: {', '.join(r['matched']) if r['matched'] else '—'}")
            print(f"   ❌ missing: {', '.join(r['missing']) if r['missing'] else 'None'}")


main()
