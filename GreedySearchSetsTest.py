import pandas as pd
import ast

def safe_eval_set(s):
    """Safely convert CSV string like "['butter','flour']" → Python set"""
    if s is None:
        return set()
    if isinstance(s, set):
        return s
    if isinstance(s, list):
        return set(s)
    try:
        return set(ast.literal_eval(s))
    except Exception:
        # fallback: split by commas
        return set(item.strip().strip("'\"") for item in str(s).split(",") if item.strip())

def normalize_ingredient(ing):
    """Normalize one ingredient string"""
    if not isinstance(ing, str):
        return str(ing).strip().lower()
    return ing.strip().lower()

def load_and_prepare(csv_path, ingredient_col='ingredients'):
    """Load CSV and convert ingredient strings to sets"""
    df = pd.read_excel("sampled_recipes.xlsx", dtype=str, keep_default_na=False)
    df[ingredient_col] = df[ingredient_col].apply(lambda s: 
        set(normalize_ingredient(x) for x in safe_eval_set(s))
    )
    return df

def greedy_search(user_ingredients, df, top_n=5, ingredient_col='ingredients'):
    """
    Greedy search using sets — rank recipes by match count.
    """
    user_set = set(normalize_ingredient(x) for x in user_ingredients)

    # Compute match info
    df = df.copy()
    df['match_set'] = df[ingredient_col].apply(lambda recipe_set: recipe_set & user_set)
    df['missing_set'] = df[ingredient_col].apply(lambda recipe_set: recipe_set - user_set)
    df['match_count'] = df['match_set'].apply(len)
    df['missing_count'] = df['missing_set'].apply(len)
    df['total_ingredients'] = df[ingredient_col].apply(len)

    # Sort: highest match_count → fewer missing → smaller total
    df = df.sort_values(
        by=['match_count', 'missing_count', 'total_ingredients'],
        ascending=[False, True, True]
    )

    return df.head(top_n)

# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    csv_path = "sampled_recipes.csv"  # or big_recipes.csv
    print("Loading and preparing dataset...")
    df = load_and_prepare(csv_path)

    user_ingredients = {"butter", "flour", "milk", "sugar",}

    print("\nRunning greedy search...")
    results = greedy_search(user_ingredients, df, top_n=5)

    for idx, row in results.iterrows():
        print("\n----")
        print("🍽️  Recipe:", row.get('name', 'Unknown'))
        print("✅ Matched:", ', '.join(sorted(row['match_set'])))
        print("❌ Missing:", ', '.join(sorted(row['missing_set'])))
        print("⭐ Match score:", row['match_count'], "/", row['total_ingredients'])