import argparse
import ast
import os
import sys
import re
import difflib
from pathlib import Path
import pandas as pd

DICT_PATH = Path("trained_dict.txt")
# -----------------------
# Optional dependency: pyspellchecker
# -----------------------
try:
    from spellchecker import SpellChecker
    _SPELL_OK = True
except Exception:
    _SPELL_OK = False

# -----------------------
# Normalization helpers
# -----------------------
def normalize_ingredient(ing):
    if ing is None:
        return ""
    return str(ing).strip().lower()

def safe_eval_set(s):
    """
    Convert strings like "['butter','flour']" or 'butter, flour' to a set of normalized strings.
    """
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
# Dataset loading
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
# Vocabulary building (only from ingredients column)
# -----------------------

_WORD_RE = re.compile(r"[a-z']+")

def build_dataset_vocab(df, ingredient_col="ingredients"):
    """
    Returns:
      dataset_phrases: set[str]  canonical ingredient phrases from dataset (normalized)
      dataset_words:   set[str]  all individual words found across those phrases
    """
    phrases = set()
    for s in df[ingredient_col]:
        phrases |= {normalize_ingredient(x) for x in s}

    words = set()
    for ph in phrases:
        words |= set(_WORD_RE.findall(ph))
    return phrases, words

# -----------------------
# Persistent dictionary I/O
# -----------------------
def save_dataset_words(dataset_words, path: Path):
    path.write_text("\n".join(sorted(dataset_words)), encoding="utf-8")
    print(f"💾 Saved {len(dataset_words)} words to {path}")

def load_dataset_words(path: Path):
    if not path.exists():
        return set()
    words = {w.strip().lower() for w in path.read_text(encoding="utf-8").splitlines() if w.strip()}
    print(f"📚 Loaded {len(words)} words from {path}")
    return words

# -----------------------
# Spellchecker + correction
# -----------------------
def init_spellchecker_from_words(dataset_words):
    """
    Initialize SpellChecker with NO built-in language.
    It will ONLY know words you load from trained_dict.txt.
    """
    if not _SPELL_OK:
        return None
    sp = SpellChecker(language=None, distance=2)  # <-- critical: language=None (no English dict)
    if dataset_words:
        sp.word_frequency.load_words(dataset_words)
    return sp

def token_correct(spell, phrase: str):
    """
    Token-level correction: 'bakin powdr' -> 'baking powder' (per word).
    Returns (corrected_phrase, list_of_changes).
    """
    if spell is None:
        return phrase.lower(), []

    changes = []
    def repl(m):
        w = m.group(0).lower()
        if len(w) <= 1 or w in spell:
            return w
        sug = spell.correction(w)
        if not sug:
            return w
        sug = sug.lower()
        if sug != w:
            changes.append((w, sug))
        return sug

    corrected = re.sub(r"[A-Za-z']+", repl, phrase.lower())
    return corrected, changes

def snap_to_dataset_phrase(phrase: str, dataset_phrases: set[str], cutoff=0.86):
    """
    Snap an already token-corrected phrase to the closest dataset phrase.
    Returns (snapped_phrase, snapped_target_or_None)
    """
    if not dataset_phrases:
        return phrase, None
    if phrase in dataset_phrases:
        return phrase, None
    match = difflib.get_close_matches(phrase, dataset_phrases, n=1, cutoff=cutoff)
    if match:
        return match[0], match[0]
    return phrase, None

# -----------------------
# Greedy search & results
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

    df = df[(df["missing_count"] <= 3) & (df["match_count"] >= 2)]

    df = df.sort_values(
        by=["missing_count", "match_count", "total_ingredients"],
        ascending=[True, False, True]
    )
    return df

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
    p = argparse.ArgumentParser(description="ChefGPT Greedy Search (with persistent, dataset-trained spellchecker)")
    p.add_argument("--file", default="sampled_recipes.xlsx",
                   help="Path to dataset (xlsx/csv)")
    p.add_argument("--ingredient-col", default="ingredients",
                   help="Column containing ingredients (default: ingredients)")
    p.add_argument("--name-col", default="name",
                   help="Column containing recipe names (default: name)")
    p.add_argument("--dict-path", default="trained_dict.txt",
                   help="Path to saved dictionary file (default: trained_dict.txt)")
    p.add_argument("--retrain", action="store_true",
                   help="Rebuild and overwrite the saved dictionary from the dataset.")
    p.add_argument("--no-spell", action="store_true",
                   help="Disable token-level spellcheck (still snaps to dataset phrases).")
    p.add_argument("--snap-cutoff", type=float, default=0.86,
                   help="Similarity threshold for snapping (0..1). Higher = stricter.")
    return p.parse_args()

# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    df = load_and_prepare(args.file, ingredient_col=args.ingredient_col)

    # Build phrases *every run* from dataset (used for phrase snapping)
    dataset_phrases, dataset_words_fresh = build_dataset_vocab(df, ingredient_col=args.ingredient_col)

    dict_path = Path(args.dict_path)

    # === Strict: only use trained dictionary (no English fallback) ===
    if args.retrain:
        # Rebuild from dataset and overwrite
        dataset_phrases, dataset_words_fresh = build_dataset_vocab(df, ingredient_col=args.ingredient_col)
        save_dataset_words(dataset_words_fresh, dict_path)
        dataset_words = dataset_words_fresh
        print("🧠 Dictionary rebuilt from dataset (--retrain).")
    else:
        if not dict_path.exists():
            sys.exit(f"❌ Trained dictionary not found: {dict_path}\n"
                     f"   Run once with --retrain to build it from the dataset.")
        dataset_words = load_dataset_words(dict_path)
        # Always rebuild dataset phrases for snapping (but NOT for spell vocab)
        dataset_phrases, _ = build_dataset_vocab(df, ingredient_col=args.ingredient_col)
        print("✅ Using saved dictionary only (no English fallback).")

    # Initialize spellchecker: ONLY with your words
    spell = None
    if not args.no_spell:
        if _SPELL_OK:
            spell = init_spellchecker_from_words(dataset_words)  # <- This uses language=None
        else:
            print("⚠️  pyspellchecker not installed. Run: pip install pyspellchecker")
            print("    Continuing with phrase snapping only.\n")

    # Initialize spellchecker with saved/built words
    spell = None
    if not args.no_spell:
        if _SPELL_OK:
            spell = init_spellchecker_from_words(dataset_words)
        else:
            print("⚠️  pyspellchecker not installed. Run: pip install pyspellchecker")
            print("    Continuing with phrase snapping only.\n")

    # Interactive input
    print("👨‍🍳 Welcome to ChefGPT!")
    print("Enter your ingredients one by one.")
    print("Press Enter with no input when you're done.\n")

    user_ingredients = []
    while True:
        try:
            raw = input("🧂 Ingredient: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye.")
            return

        if not raw:
            break

        # 1) token-level correction
        corrected, token_changes = token_correct(spell, raw) if spell else (raw.lower(), [])
        corrected = normalize_ingredient(corrected)

        # 2) phrase snap to dataset
        snapped, snap_target = snap_to_dataset_phrase(corrected, dataset_phrases, cutoff=args.snap_cutoff)

        if token_changes or snap_target:
            msgs = []
            if token_changes:
                uniq = []
                seen = set()
                for a, b in token_changes:
                    if (a, b) not in seen:
                        uniq.append((a, b))
                        seen.add((a, b))
                msgs.append("token: " + ", ".join([f"{a}→{b}" for a, b in uniq[:4]]))
            if snap_target:
                msgs.append(f"phrase→ '{snap_target}'")
            print("   ✍️  Auto-corrected:", " | ".join(msgs))

        user_ingredients.append(snapped)

    if not user_ingredients:
        print("No ingredients entered. Exiting.")
        return

    print("\n🔍 Searching recipes with your ingredients...\n")
    results = greedy_search(user_ingredients, df, ingredient_col=args.ingredient_col)
    print_results(results, name_col=args.name_col)

if __name__ == "__main__":
    main()
