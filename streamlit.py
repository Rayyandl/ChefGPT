import argparse
import ast
import os
import sys
import re
import difflib
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

# -----------------------
# Constants / Paths
# -----------------------
HARD_MAX_MISSING = 2  # hard cap: recipes with >2 missing are filtered out

DICT_PATH = Path("trained_dict.txt")
WEIGHTS_PATH = Path("ingredient_weights.json")
FEEDBACK_LOG = Path("feedback_log.csv")

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

def safe_eval_list(s):
    """
    Convert strings like "['step1','step2']" to list[str].
    If not parseable, try a simple split on common separators; otherwise return [string].
    """
    if s is None:
        return []
    if isinstance(s, (list, tuple)):
        return [str(x).strip() for x in s]
    if isinstance(s, set):
        return [str(x).strip() for x in s]
    text = str(s).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip() for x in parsed]
        if isinstance(parsed, str):
            return [parsed.strip()]
    except Exception:
        pass
    # fallback split: try pipes/semicolons/commas/newlines
    parts = re.split(r"\n+|[|;]|,(?=(?:[^\"']|\"[^\"]*\"|'[^']*')*$)", text)
    parts = [p.strip().strip("\"'") for p in parts if p and p.strip()]
    return parts if parts else ([text] if text else [])

# -----------------------
# Dataset loading
# -----------------------
import os
import sys
import pandas as pd
import tempfile

def load_and_prepare(file_path, ingredient_col="ingredients"):
    """
    Loads and prepares a dataset of recipes.
    Supports both normal file paths and uploaded file objects (e.g. from Streamlit).
    """
    import tempfile
    import pandas as pd
    import os
    import chardet

    # 🟢 Case 1: If it's a Streamlit UploadedFile (in-memory object)
    if hasattr(file_path, "read"):
        file_content = file_path.read()
        
        # Detect encoding for CSV files
        if hasattr(file_path, 'name') and file_path.name.lower().endswith('.csv'):
            detected = chardet.detect(file_content)
            encoding = detected.get('encoding', 'utf-8')
            # Reset file pointer
            file_path.seek(0)
        else:
            encoding = 'utf-8'
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_path.name)[1]) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        path_to_use = tmp_path
    else:
        path_to_use = file_path
        if not os.path.exists(path_to_use):
            raise FileNotFoundError(f"File not found: {path_to_use}")

    ext = os.path.splitext(path_to_use)[1].lower()

    try:
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(path_to_use, dtype=str, keep_default_na=False)
        elif ext in (".csv", ".txt"):
            # Try multiple encodings for CSV files
            encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'windows-1252', 'cp1252']
            
            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(path_to_use, dtype=str, keep_default_na=False, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                # If all encodings fail, try with error handling
                df = pd.read_csv(path_to_use, dtype=str, keep_default_na=False, encoding='utf-8', errors='replace')
        else:
            raise ValueError("Unsupported file type. Use .csv or .xlsx")

        if ingredient_col not in df.columns:
            raise ValueError(f"Column '{ingredient_col}' not found. Available: {list(df.columns)}")

        df = df.copy()
        df[ingredient_col] = df[ingredient_col].apply(safe_eval_set)
        
        # Clean up temporary file if we created one
        if hasattr(file_path, "read"):
            os.unlink(tmp_path)
            
        return df

    except Exception as e:
        # Clean up temporary file if we created one and there was an error
        if hasattr(file_path, "read") and 'tmp_path' in locals():
            os.unlink(tmp_path)
        raise e


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

def maybe_append_new_words(tokens, dict_path: Path, known_words: set[str], dry_run=False):
    new_words = [t for t in tokens if t not in known_words and t.isalpha()]
    if not new_words:
        return known_words, 0
    if not dry_run:
        with dict_path.open("a", encoding="utf-8") as f:
            for w in sorted(set(new_words)):
                f.write("\n" + w)
    known_words |= set(new_words)
    return known_words, len(set(new_words))

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
    sp = SpellChecker(language=None, distance=2)  # strict: no English fallback
    if dataset_words:
        sp.word_frequency.load_words(dataset_words)
    return sp

def token_correct(spell, phrase: str):
    """
    Token-level correction: 'bakin powdr' -> 'baking powder' (per word).
    Returns (corrected_phrase, list_of_changes, list_of_tokens).
    """
    if spell is None:
        toks = _WORD_RE.findall(phrase.lower())
        return phrase.lower(), [], toks

    changes = []
    toks = []

    def repl(m):
        w = m.group(0).lower()
        toks.append(w)
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
    return corrected, changes, toks

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
# Ingredient weights (learning)
# -----------------------
def load_weights(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_weights(weights: dict, path: Path):
    path.write_text(json.dumps(weights, ensure_ascii=False, indent=2), encoding="utf-8")

def weight_of(ing: str, weights: dict, default=0.0) -> float:
    return float(weights.get(ing, default))

def update_weights_for_feedback(
    weights: dict,
    matched_set: set[str],
    missing_set: set[str],
    accepted: bool,
    lr_pos=1.0,
    lr_neg=0.25,
    clip_min=-5.0,
    clip_max=10.0,
):
    """
    Simple online learning:
      - If accepted: increase weights of matched, slightly decrease missing
      - If rejected: slightly decrease matched (they weren't helpful)
    """
    if accepted:
        for m in matched_set:
            weights[m] = max(clip_min, min(clip_max, weight_of(m, weights) + lr_pos))
        for miss in missing_set:
            weights[miss] = max(clip_min, min(clip_max, weight_of(miss, weights) - lr_neg))
    else:
        for m in matched_set:
            weights[m] = max(clip_min, min(clip_max, weight_of(m, weights) - lr_neg))
    return weights

def log_feedback(row, accepted: bool, name_col: str):
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    is_new = not FEEDBACK_LOG.exists()
    with FEEDBACK_LOG.open("a", encoding="utf-8") as f:
        if is_new:
            f.write("timestamp,action,recipe,match_count,missing_count,matched,missing,total\n")
        line = (
            f"{datetime.now().isoformat(timespec='seconds')},"
            f"{'accept' if accepted else 'reject'},"
            f"\"{str(row.get(name_col, 'Unknown')).replace('\"','\"\"')}\","
            f"{row['match_count']},{row['missing_count']},"
            f"\"{', '.join(sorted(row['match_set']))}\","
            f"\"{', '.join(sorted(row['missing_set']))}\","
            f"{row['total_ingredients']}\n"
        )
        f.write(line)

# -----------------------
# Scoring, search & results
# -----------------------
def compute_score(row, weights: dict, alpha=0.6, beta=0.5, gamma=0.4):
    """
    Weighted score:
      base = matched / total  (0..1)
      boost = sum(weight[matched])
      penalty = gamma * missing_count
      final = alpha*base + beta*boost - penalty
    """
    total = max(1, row["total_ingredients"])
    base = row["match_count"] / total
    boost = sum(weight_of(x, weights) for x in row["match_set"])
    penalty = gamma * row["missing_count"]
    return alpha * base + beta * boost - penalty

def greedy_search(user_ingredients, df, weights, ingredient_col="ingredients",
                  hard_min_matches=2,
                  alpha=0.6, beta=0.5, gamma=0.4):
    """
    Rank primarily by learned score; keep old tie-breakers for stability.
    Hard filter: missing_count <= HARD_MAX_MISSING (3) and match_count >= hard_min_matches.
    """
    user_set = {normalize_ingredient(x) for x in user_ingredients if str(x).strip()}
    df = df.copy()

    df["match_set"] = df[ingredient_col].apply(lambda s: s & user_set)
    df["missing_set"] = df[ingredient_col].apply(lambda s: s - user_set)
    df["match_count"] = df["match_set"].apply(len)
    df["missing_count"] = df["missing_set"].apply(len)
    df["total_ingredients"] = df[ingredient_col].apply(len)

    # Hard constraints (HARD_MAX_MISSING is fixed at 3)
    df = df[(df["missing_count"] <= HARD_MAX_MISSING) & (df["match_count"] >= hard_min_matches)]

    # Learned score
    df["score"] = df.apply(lambda r: compute_score(r, weights, alpha=alpha, beta=beta, gamma=gamma), axis=1)

    df = df.sort_values(
        by=["score", "missing_count", "match_count", "total_ingredients"],
        ascending=[False, True, False, True]
    )
    return df

def print_results(results, name_col="name"):
    if results.empty:
        print("\n😕 No recipes within the missing-ingredients limit and minimum matches.")
        return

    print("\n=== Top Suggestions ===")
    for idx, (_, row) in enumerate(results.iterrows(), start=1):
        name = row.get(name_col, "Unknown")
        matched = ", ".join(sorted(row["match_set"])) or "—"
        missing = ", ".join(sorted(row["missing_set"])) or "—"
        print("\n----")
        print(f"[{idx}] 🍽️  {name}")
        print(f"✅ Matched: {matched}")
        print(f"❌ Missing: {missing}")
        print(f"⭐ {row['match_count']} matched / {row['total_ingredients']} total")
        print(f"🧮 Score: {row['score']:.3f}")

def print_full_recipe(row, name_col="name",
                      measure_col="ingredients_measurement",
                      steps_col="steps"):
    print("\n================= 📖 RECIPE =================")
    print(f"🍽️  {row.get(name_col, 'Unknown')}")
    print("---------------------------------------------")

    # Ingredients measurement
    if measure_col in row and pd.notna(row[measure_col]):
        measures = safe_eval_list(row[measure_col])
        if measures:
            print("\n🧂 Ingredients + Measurement:")
            for i, m in enumerate(measures, 1):
                print(f"  {i}. {m}")
        else:
            print("\n🧂 Ingredients + Measurement: (no structured list)")
            print(f"  {row[measure_col]}")
    else:
        print("\n🧂 Ingredients + Measurement: (missing column or value)")

    # Steps
    if steps_col in row and pd.notna(row[steps_col]):
        steps = safe_eval_list(row[steps_col])
        if steps:
            print("\n👣 Steps:")
            for i, s in enumerate(steps, 1):
                print(f"  {i}. {s}")
        else:
            print("\n👣 Steps: (no structured list)")
            print(f"  {row[steps_col]}")
    else:
        print("\n👣 Steps: (missing column or value)")

    print("=============================================\n")

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="ChefGPT Greedy Search (dataset-trained spellchecker + online learning)")
    p.add_argument("--file", default="sampled_recipes.xlsx",   # you uploaded this; change path if needed
                   help="Path to dataset (xlsx/csv)")
    p.add_argument("--ingredient-col", default="ingredients",
                   help="Column containing ingredients (default: ingredients)")
    p.add_argument("--name-col", default="name",
                   help="Column containing recipe names (default: name)")
    p.add_argument("--measure-col", default="ingredients_measurement",
                   help="Column containing ingredients + measurement (default: ingredients_measurement)")
    p.add_argument("--steps-col", default="steps",
                   help="Column containing recipe steps (default: steps)")

    # Dictionary training
    p.add_argument("--dict-path", default=str(DICT_PATH),
                   help="Path to saved dictionary file (default: trained_dict.txt)")
    p.add_argument("--retrain", action="store_true",
                   help="Rebuild and overwrite the saved dictionary from the dataset.")
    p.add_argument("--no-spell", action="store_true",
                   help="Disable token-level spellcheck (still snaps to dataset phrases).")
    p.add_argument("--snap-cutoff", type=float, default=0.86,
                   help="Similarity threshold for snapping (0..1). Higher = stricter.")
    p.add_argument("--auto-append-dict", action="store_true",
                   help="Append unseen tokens from user input to trained_dict.txt")

    # Learning / weights
    p.add_argument("--no-learn", action="store_true",
                   help="Disable interactive feedback & learning.")
    p.add_argument("--lr-pos", type=float, default=1.0,
                   help="Learning rate for positive feedback (default: 1.0)")
    p.add_argument("--lr-neg", type=float, default=0.25,
                   help="Learning rate for negative feedback (default: 0.25)")
    p.add_argument("--alpha", type=float, default=0.6,
                   help="Score weight for base match fraction (default: 0.6)")
    p.add_argument("--beta", type=float, default=0.5,
                   help="Score weight multiplier for sum of ingredient weights (default: 0.5)")
    p.add_argument("--gamma", type=float, default=0.4,
                   help="Score penalty multiplier per missing ingredient (default: 0.4)")
    p.add_argument("--hard-min-matches", type=int, default=2,
                   help="Minimum matches required (default: 2)")

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

    # Initialize spellchecker ONCE: ONLY with your words
    spell = None
    if not args.no_spell:
        if _SPELL_OK:
            spell = init_spellchecker_from_words(dataset_words)  # language=None
        else:
            print("⚠️  pyspellchecker not installed. Run: pip install pyspellchecker")
            print("    Continuing with phrase snapping only.\n")

    # Load weights (learning)
    weights = load_weights(WEIGHTS_PATH)

    # Interactive input
    print("👨‍🍳 Welcome to ChefGPT!")
    print("Enter your ingredients one by one.")
    print("Press Enter with no input when you're done.\n")

    user_ingredients = []
    user_tokens_all = []

    while True:
        try:
            raw = input("🧂 Ingredient: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye.")
            return

        if not raw:
            break

        # 1) token-level correction
        corrected, token_changes, toks = token_correct(spell, raw) if spell else (raw.lower(), [], _WORD_RE.findall(raw.lower()))
        corrected = normalize_ingredient(corrected)
        user_tokens_all.extend(toks)

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
                msgs.append("token: " + ", ".join([f"{a}→{b}" for a, b in uniq[:6]]))
            if snap_target:
                msgs.append(f"phrase→ '{snap_target}'")
            print("   ✍️  Auto-corrected:", " | ".join(msgs))

        user_ingredients.append(snapped)

    if not user_ingredients:
        print("No ingredients entered. Exiting.")
        return

    # Optionally append new tokens to dictionary
    if args.auto_append_dict:
        _ds_words, added = maybe_append_new_words(user_tokens_all, dict_path, set(dataset_words), dry_run=False)
        if added:
            print(f"📥 Appended {added} new token(s) to {dict_path}. (Next run will load them.)")

    print(f"\n🔍 Searching recipes with your ingredients (missing ≤ {HARD_MAX_MISSING})...\n")
    results = greedy_search(
        user_ingredients, df, weights,
        ingredient_col=args.ingredient_col,
        hard_min_matches=args.hard_min_matches,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma
    )
    print_results(results, name_col=args.name_col)

    if results.empty or args.no_learn:
        return

    # --------- Feedback loop (multi-rejects; accept exits) ---------
    print("\n👍👎 Help me learn:")
    print(" - Type the number of a suggestion you liked to ACCEPT it (prints full recipe).")
    print(" - Type 'r <num>' to REJECT a suggestion (you can reject multiple).")
    print(" - Press Enter to finish without feedback.\n")

    while True:
        try:
            fb = input("Feedback: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye.")
            return

        if not fb:
            # finished without accepting; no more updates
            break

        accept_idx = None
        reject_idx = None

        if fb.lower().startswith("r "):
            # REJECT path; stay in the loop
            try:
                reject_idx = int(fb.split()[1])
            except Exception:
                print("⚠️ Couldn't parse rejection index. Use 'r <num>' (e.g., r 2).")
                continue

            if reject_idx < 1 or reject_idx > len(results):
                print("⚠️ Out of range.")
                continue

            row = results.iloc[reject_idx - 1]
            weights = update_weights_for_feedback(
                weights,
                matched_set=row["match_set"],
                missing_set=row["missing_set"],
                accepted=False,
                lr_pos=args.lr_pos,
                lr_neg=args.lr_neg
            )
            save_weights(weights, WEIGHTS_PATH)
            log_feedback(row, accepted=False, name_col=args.name_col)
            print(f"🗑️ Rejected [{reject_idx}] {row.get(args.name_col, 'Unknown')}. Model updated.")
            # continue loop for more feedback
            continue
        else:
            # ACCEPT path; exit after showing full recipe
            try:
                accept_idx = int(fb)
            except Exception:
                print("⚠️ Couldn't parse acceptance index. Type a number like '1', or 'r 2' to reject.")
                continue

            if accept_idx < 1 or accept_idx > len(results):
                print("⚠️ Out of range.")
                continue

            row = results.iloc[accept_idx - 1]
            weights = update_weights_for_feedback(
                weights,
                matched_set=row["match_set"],
                missing_set=row["missing_set"],
                accepted=True,
                lr_pos=args.lr_pos,
                lr_neg=args.lr_neg
            )
            save_weights(weights, WEIGHTS_PATH)
            log_feedback(row, accepted=True, name_col=args.name_col)
            print("✅ Learned from your acceptance. Weights updated.")

            # Print full recipe (ingredients_measurement + steps) for the accepted recipe only
            print_full_recipe(row, name_col=args.name_col,
                              measure_col=args.measure_col,
                              steps_col=args.steps_col)
            # Exit feedback loop after acceptance
            break

if __name__ == "__main__":
    main()
