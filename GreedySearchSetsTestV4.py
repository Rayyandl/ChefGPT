import argparse
import ast
import os
import sys
import pandas as pd

# -----------------------
# Helper functions
# -----------------------
def normalize_ingredient(ing):
    if ing is None:
        return ""
    return str(ing).strip().lower()

def safe_eval_set(s):
    """Convert strings like "['butter','flour']" or 'butter, flour' to a set of normalized strings."""
    if s is None:
        return set()
    if isinstance(s, set):
        return {normalize_ingredient(x) for x in s}
    if isinstance(s, list):
        return {normalize_ingredient(x) for x in s}
    text = str(s).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return {normalize_ingredient(x) for x in parsed}
        if isinstance(parsed, str):
            return {normalize_ingredient(parsed)}
    except Exception:
        pass
    return {normalize_ingredient(x) for x in text.split(",") if x.strip()}

# -----------------------
# Load dataset
# -----------------------
def load_and_prepare(file_path, ingredient_col="ingredients"):
    if not os.path.exists(file_path):
        sys.exit(f"❌ File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(file_path, dtype=str, keep_default_na=False)
    elif ext in (".csv", ".txt"):
        df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
    else:
        sys.exit("❌ Unsupported file type. Use .csv or .xlsx")

    if ingredient_col not in df.columns:
        sys.exit(f"❌ Column '{ingredient_col}' not found. Available: {list(df.columns)}")

    df = df.copy()
    df[ingredient_col] = df[ingredient_col].apply(safe_eval_set)
    return df

# -----------------------
# Greedy search
# -----------------------
def greedy_search(user_ingredients, df, ingredient_col="ingredients"):
    """
    Rank by missing_count ASC, then match_count DESC, then total_ingredients ASC.
    Hard filter: missing_count <= 3 and match_count >= 2.
    """
    user_set = {normalize_ingredient(x) for x in user_ingredients if str(x).strip()}
    df = df.copy()

    df["match_set"] = df[ingredient_col].apply(lambda s: s & user_set)
    df["missing_set"] = df[ingredient_col].apply(lambda s: s - user_set)
    df["match_count"] = df["match_set"].apply(len)
    df["missing_count"] = df["missing_set"].apply(len)
    df["total_ingredients"] = df[ingredient_col].apply(len)

    # Hard filters
    df = df[(df["missing_count"] <= 3) & (df["match_count"] >= 2)]

    # Sort order
    df = df.sort_values(
        by=["missing_count", "match_count", "total_ingredients"],
        ascending=[True, False, True]
    )
    return df

# -----------------------
# Display results
# -----------------------
def print_results(results, name_col="name"):
    if results.empty:
        print("\n😕 No recipes within the missing-ingredients limit (≤ 3) and at least 2 matches.")
        return

    current_bucket = None
    for _, row in results.iterrows():
        if row["missing_count"] != current_bucket:
            current_bucket = row["missing_count"]
            print(f"\n=== Missing {current_bucket} ingredient(s) ===")
        name = row.get(name_col, "Unknown")
        matched = ", ".join(sorted(row["match_set"])) or "—"
        missing = ", ".join(sorted(row["missing_set"])) or "—"
        print("\n----")
        print(f"🍽️  Recipe: {name}")
        print(f"✅ Matched: {matched}")
        print(f"❌ Missing: {missing}")
        print(f"⭐ {row['match_count']} matched / {row['total_ingredients']} total")

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="ChefGPT Greedy Search (interactive input, filtered)")
    p.add_argument("--file", default="sampled_recipes.xlsx",
                   help="Path to dataset (default: sampled_recipes.xlsx)")
    p.add_argument("--ingredient-col", default="ingredients",
                   help="Column containing ingredients (default: ingredients)")
    p.add_argument("--name-col", default="name",
                   help="Column containing recipe names (default: name)")
    return p.parse_args()

# -----------------------
# Main logic
# -----------------------
def main():
    args = parse_args()
    df = load_and_prepare(args.file, ingredient_col=args.ingredient_col)

    print("👨‍🍳 Welcome to ChefGPT!")
    print("Enter your ingredients one by one.")
    print("Press Enter with no input when you're done.\n")

    user_ingredients = []
    while True:
        item = input("🧂 Ingredient: ").strip()
        if not item:
            break
        user_ingredients.append(item)

    if not user_ingredients:
        print("No ingredients entered. Exiting.")
        return

    print("\n🔍 Searching recipes with your ingredients...\n")
    results = greedy_search(user_ingredients, df, ingredient_col=args.ingredient_col)
    print_results(results, name_col=args.name_col)

if __name__ == "__main__":
    main()
