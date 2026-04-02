"""
ChefGPT Agent — conversational, reasoning-first recipe assistant.

Key upgrades over the original greedy CLI:
  • Natural language input parsed with intent extraction
  • Multi-turn conversation with persistent state
  • Proactive reasoning: explains *why* each suggestion is good
  • Dynamic replanning: relaxes/tightens constraints when results are poor
  • Dietary / cuisine preference filters extracted from free text
  • Typed agent loop with clear Thought → Action → Observation structure
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import difflib
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

total_suggestions = 0
accepted_suggestions = 0
accepted_ranks = []

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
HARD_MAX_MISSING      = 2
DICT_PATH             = Path("trained_dict.txt")
WEIGHTS_PATH          = Path("ingredient_weights.json")
FEEDBACK_LOG          = Path("feedback_log.csv")

# Simple keyword maps for intent extraction (extend freely)
DIETARY_KEYWORDS: dict[str, list[str]] = {
    "vegetarian":  ["vegetarian", "veggie", "no meat", "meatless"],
    "vegan":       ["vegan", "plant-based", "plant based"],
    "gluten-free": ["gluten free", "gluten-free", "no gluten", "celiac"],
    "spicy":       ["spicy", "hot", "spice", "chili"],
    "quick":       ["quick", "fast", "easy", "15 min", "30 min", "simple"],
}

CUISINE_KEYWORDS: list[str] = [
    "italian", "mexican", "indian", "chinese", "japanese", "thai",
    "french", "american", "mediterranean", "greek", "korean", "vietnamese",
]

# ──────────────────────────────────────────────────────────────
# Optional dependency: pyspellchecker
# ──────────────────────────────────────────────────────────────
try:
    from spellchecker import SpellChecker
    _SPELL_OK = True
except Exception:
    _SPELL_OK = False

# ──────────────────────────────────────────────────────────────
# Helpers — normalisation, parsing
# ──────────────────────────────────────────────────────────────
_WORD_RE = re.compile(r"[a-z']+")

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

def safe_eval_list(s) -> list[str]:
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
    parts = re.split(r"\n+|[|;]|,(?=(?:[^\"']|\"[^\"]*\"|'[^']*')*$)", text)
    parts = [p.strip().strip("\"'") for p in parts if p.strip()]
    return parts or ([text] if text else [])

# ──────────────────────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────────────────────
def load_and_prepare(file_path: str, ingredient_col="ingredients") -> pd.DataFrame:
    if not os.path.exists(file_path):
        sys.exit(f"❌  File not found: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(file_path, dtype=str, keep_default_na=False)
    elif ext in (".csv", ".txt"):
        df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
    else:
        sys.exit("❌  Unsupported file type. Use .csv or .xlsx")
    if ingredient_col not in df.columns:
        sys.exit(f"❌  Column '{ingredient_col}' not found. Available: {list(df.columns)}")
    df = df.copy()
    df[ingredient_col] = df[ingredient_col].apply(safe_eval_set)
    return df

# ──────────────────────────────────────────────────────────────
# Vocabulary / spellchecker
# ──────────────────────────────────────────────────────────────
def build_dataset_vocab(df, ingredient_col="ingredients"):
    phrases: set[str] = set()
    for s in df[ingredient_col]:
        phrases |= {normalize(x) for x in s}
    words: set[str] = set()
    for ph in phrases:
        words |= set(_WORD_RE.findall(ph))
    return phrases, words

def load_dataset_words(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {w.strip().lower() for w in path.read_text("utf-8").splitlines() if w.strip()}

def save_dataset_words(words: set[str], path: Path):
    path.write_text("\n".join(sorted(words)), encoding="utf-8")

def init_spell(words: set[str]) -> Optional["SpellChecker"]:
    if not _SPELL_OK or not words:
        return None
    sp = SpellChecker(language=None, distance=2)
    sp.word_frequency.load_words(words)
    return sp

def token_correct(spell, phrase: str):
    changes, toks = [], []
    def repl(m):
        w = m.group(0).lower()
        toks.append(w)
        if len(w) <= 1 or w in spell:
            return w
        sug = spell.correction(w)
        if sug and sug.lower() != w:
            changes.append((w, sug.lower()))
            return sug.lower()
        return w
    corrected = re.sub(r"[A-Za-z']+", repl, phrase.lower()) if spell else phrase.lower()
    return corrected, changes, toks

def snap_to_phrase(phrase: str, dataset_phrases: set[str], cutoff=0.86):
    if phrase in dataset_phrases:
        return phrase, None
    match = difflib.get_close_matches(phrase, dataset_phrases, n=1, cutoff=cutoff)
    return (match[0], match[0]) if match else (phrase, None)

# ──────────────────────────────────────────────────────────────
# Intent extraction  ← NEW AGENT FEATURE
# ──────────────────────────────────────────────────────────────

def extract_intent(text: str) -> dict:
    """
    Parse a free-text user utterance and return structured intent:
      {
        "raw_ingredients": [...],   # ingredient-like tokens
        "dietary": [...],           # e.g. ["vegetarian", "spicy"]
        "cuisine": [...],           # e.g. ["italian"]
        "add_mode": bool,           # "also add X" or "and X"
        "clear_mode": bool,         # "start over" / "clear"
        "show_recipe": int|None,    # "show me recipe 2"
        "accept": int|None,         # "I'll make #3" / "accept 3"
        "reject": int|None,         # "not 2" / "reject 2"
        "relax": bool,              # "show more" / "relax"
        "help": bool,
        "quit": bool,
      }
    """
    t = text.strip().lower()
    intent: dict = {
        "raw_ingredients": [],
        "dietary": [],
        "cuisine": [],
        "add_mode": False,
        "clear_mode": False,
        "show_recipe": None,
        "accept": None,
        "reject": None,
        "relax": False,
        "help": False,
        "quit": False,
    }

    # Quit
    if re.match(r"^(quit|exit|bye|q)$", t):
        intent["quit"] = True
        return intent

    # Help
    if re.match(r"^(help|\?|commands)$", t):
        intent["help"] = True
        return intent

    # Clear
    if re.search(r"\b(start over|clear|reset|new search)\b", t):
        intent["clear_mode"] = True
        return intent

    # Relax constraints
    if re.search(r"\b(show more|relax|less strict|more results|broaden)\b", t):
        intent["relax"] = True
        return intent

    # Accept: "1", "accept 3", "i'll make 2", "make #1", "yes 2"
    m = re.search(r"\b(?:accept|make|yes|choose|pick|i.ll make|show recipe|show me recipe)\s*#?\s*(\d+)", t)
    if m:
        intent["accept"] = int(m.group(1))
        return intent
    # bare number = accept
    if re.match(r"^#?\s*(\d+)$", t.strip()):
        intent["accept"] = int(re.match(r"^#?\s*(\d+)$", t.strip()).group(1))
        return intent

    # Reject: "reject 2", "not 1", "r 2", "skip 3"
    m = re.search(r"\b(?:reject|not|skip|r)\s+#?(\d+)", t)
    if m:
        intent["reject"] = int(m.group(1))
        return intent

    # Dietary / preference flags
    for tag, keywords in DIETARY_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            intent["dietary"].append(tag)

    # Cuisine
    for c in CUISINE_KEYWORDS:
        if c in t:
            intent["cuisine"].append(c)

    # Add mode
    if re.search(r"\b(also|add|plus|and i (?:also )?have|i also have)\b", t):
        intent["add_mode"] = True

    # Extract ingredient phrases — strip filler words then split on commas/and
    filler = re.compile(
        r"\b(i have|i've got|i got|i only have|using|with|use|"
        r"ingredients?|and i have|also have|add|please|can you|"
        r"make|suggest|find|search|something|recipe[s]?|for me|"
        r"that is|that are|which is|which are)\b"
    )
    stripped = filler.sub(" ", t)
    # Also strip dietary / cuisine words we already captured
    for tag, kws in DIETARY_KEYWORDS.items():
        for kw in kws:
            stripped = stripped.replace(kw, " ")
    for c in CUISINE_KEYWORDS:
        stripped = stripped.replace(c, " ")

    # Split on "and", ",", "+"
    parts = re.split(r"[,+]|\band\b", stripped)
    for part in parts:
        ing = part.strip().strip(".")
        if ing and len(ing) > 1:
            # keep only if it looks like an ingredient (has letters)
            if re.search(r"[a-z]", ing):
                intent["raw_ingredients"].append(ing)

    return intent

# ──────────────────────────────────────────────────────────────
# Weights / learning
# ──────────────────────────────────────────────────────────────
def load_weights(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception:
        return {}

def save_weights(weights: dict, path: Path):
    path.write_text(json.dumps(weights, ensure_ascii=False, indent=2), "utf-8")

def w(ing: str, weights: dict) -> float:
    return float(weights.get(ing, 0.0))

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

def log_feedback(row, accepted: bool, name_col: str):
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    is_new = not FEEDBACK_LOG.exists()
    with FEEDBACK_LOG.open("a", encoding="utf-8") as f:
        if is_new:
            f.write("timestamp,action,recipe,match_count,missing_count,matched,missing,total\n")
        f.write(
            f"{datetime.now().isoformat(timespec='seconds')},"
            f"{'accept' if accepted else 'reject'},"
            f"\"{str(row.get(name_col,'Unknown')).replace('\"','\"\"')}\","
            f"{row['match_count']},{row['missing_count']},"
            f"\"{','.join(sorted(row['match_set']))}\","
            f"\"{','.join(sorted(row['missing_set']))}\","
            f"{row['total_ingredients']}\n"
        )

# ──────────────────────────────────────────────────────────────
# Scoring & search
# ──────────────────────────────────────────────────────────────
def compute_score(row, weights, alpha=0.6, beta=0.5, gamma=0.4) -> float:
    total  = max(1, row["total_ingredients"])
    base   = row["match_count"] / total
    boost  = sum(w(x, weights) for x in row["match_set"])
    penalty = gamma * row["missing_count"]
    return alpha * base + beta * boost - penalty

def search(user_ings, df, weights, ingredient_col="ingredients",
           hard_min_matches=2, max_missing=HARD_MAX_MISSING,
           dietary=None, cuisine=None,
           alpha=0.6, beta=0.5, gamma=0.4) -> pd.DataFrame:
    user_set = {normalize(x) for x in user_ings if str(x).strip()}
    r = df.copy()
    r["match_set"]          = r[ingredient_col].apply(lambda s: s & user_set)
    r["missing_set"]        = r[ingredient_col].apply(lambda s: s - user_set)
    r["match_count"]        = r["match_set"].apply(len)
    r["missing_count"]      = r["missing_set"].apply(len)
    r["total_ingredients"]  = r[ingredient_col].apply(len)

    # Hard filters
    r = r[(r["missing_count"] <= max_missing) & (r["match_count"] >= hard_min_matches)]

    # Early exit — prevents the "DataFrame without columns" crash
    if r.empty:
        return r

    # Soft preference filters (name-based heuristic)
    name_col_guess = "name"
    if dietary:
        for tag in dietary:
            if tag in ("vegetarian", "vegan"):
                meat_words = {"chicken","beef","pork","lamb","fish","shrimp","bacon",
                              "turkey","tuna","salmon","prawn","crab","lobster","anchovy"}
                if tag == "vegan":
                    meat_words |= {"egg","eggs","milk","cheese","butter","cream","honey"}
                r = r[~r["match_set"].apply(lambda s: bool(s & meat_words))]
                r = r[~r["missing_set"].apply(lambda s: bool(s & meat_words))]
            elif tag == "spicy":
                spice_words = {"chili","chilli","cayenne","jalapeño","pepper","sriracha",
                               "habanero","hot sauce","red pepper"}
                r = r[r[ingredient_col].apply(lambda s: bool(s & spice_words))]
    if cuisine:
        for c in cuisine:
            r = r[r.get(name_col_guess, pd.Series(dtype=str)).str.lower().str.contains(c, na=False)
                  | r[ingredient_col].apply(lambda s: any(c in x for x in s))]

    # Second early exit — soft filters may have emptied it too
    if r.empty:
        return r

    r["score"] = r.apply(lambda row: compute_score(row, weights, alpha, beta, gamma), axis=1)
    r = r.sort_values(
        ["score", "missing_count", "match_count", "total_ingredients"],
        ascending=[False, True, False, True]
    )
    return r

# ──────────────────────────────────────────────────────────────
# Reasoning / explanation  ← NEW AGENT FEATURE
# ──────────────────────────────────────────────────────────────

def explain_suggestion(row, rank: int, name_col: str, weights: dict) -> str:
    """
    Generate a short natural-language explanation of *why* this recipe was suggested.
    This is the agent's 'reasoning' step made visible to the user.
    """
    name    = row.get(name_col, "this recipe")
    matched = sorted(row["match_set"])
    missing = sorted(row["missing_set"])
    total   = row["total_ingredients"]
    score   = row["score"]

    # Identify high-weight ingredients
    top_ing = sorted(matched, key=lambda x: w(x, weights), reverse=True)[:2]

    lines = [f"[{rank}] 🍽️  {name}  (score: {score:.2f})"]

    # Coverage
    pct = int(100 * row["match_count"] / max(1, total))
    if pct == 100:
        lines.append(f"   ✨ You have ALL {total} ingredients — perfect match!")
    else:
        lines.append(f"   ✅ {row['match_count']}/{total} ingredients covered ({pct}%)")

    # Highlight high-weight matches
    if top_ing:
        lines.append(f"   💡 Key match{'es' if len(top_ing)>1 else ''}: {', '.join(top_ing)}"
                     + (" (frequently chosen by you)" if any(w(x, weights) > 1 for x in top_ing) else ""))

    # Missing
    if missing:
        lines.append(f"   🛒 You'd need: {', '.join(missing)}")
    else:
        lines.append("   🛒 Nothing missing — cook it now!")

    return "\n".join(lines)

# ──────────────────────────────────────────────────────────────
# Full recipe display
# ──────────────────────────────────────────────────────────────
def print_recipe(row, name_col="name", measure_col="ingredients_measurement", steps_col="steps"):
    print("\n" + "═" * 50)
    print(f"  📖  {row.get(name_col, 'Unknown')}")
    print("═" * 50)
    if measure_col in row and pd.notna(row[measure_col]):
        measures = safe_eval_list(row[measure_col])
        print("\n🧂  Ingredients:")
        for i, m in enumerate(measures, 1):
            print(f"   {i}. {m}")
    if steps_col in row and pd.notna(row[steps_col]):
        steps = safe_eval_list(row[steps_col])
        print("\n👣  Steps:")
        for i, s in enumerate(steps, 1):
            print(f"   {i}. {s}")
    print("═" * 50 + "\n")

# ──────────────────────────────────────────────────────────────
# Help text
# ──────────────────────────────────────────────────────────────
HELP_TEXT = """
┌─────────────────────────────────────────────────────────┐
│  ChefGPT Agent — what you can say                       │
├─────────────────────────────────────────────────────────┤
│  Describe ingredients naturally:                        │
│    "I have chicken, garlic, and lemon"                  │
│    "Using tomatoes, pasta, and basil"                   │
│  Add preferences inline:                                │
│    "I want something spicy with beef"                   │
│    "vegetarian, I have eggs and spinach"                │
│  Add more ingredients:                                  │
│    "also add butter"  /  "I also have onions"           │
│  Accept / reject suggestions:                           │
│    "1"  or  "accept 2"  or  "I'll make 3"               │
│    "reject 2"  or  "r 2"  or  "skip 1"                  │
│  Broaden search (if too few results):                   │
│    "show more"  /  "relax"                              │
│  Start fresh:                                           │
│    "start over"  /  "clear"                             │
│  Show help:  "help"   |   Quit:  "quit"                 │
└─────────────────────────────────────────────────────────┘
"""

# ──────────────────────────────────────────────────────────────
# Agent state  ← NEW: encapsulates all mutable state
# ──────────────────────────────────────────────────────────────

class AgentState:
    def __init__(self):
        self.ingredients: list[str] = []     # current resolved ingredient list
        self.dietary: list[str]     = []
        self.cuisine: list[str]     = []
        self.results: pd.DataFrame  = pd.DataFrame()
        self.max_missing: int       = HARD_MAX_MISSING
        self.turn: int              = 0
        # 🔥 NEW METRICS
        self.total_suggestions: int = 0
        self.accepted_suggestions: int = 0
        self.accepted_ranks: list[int] = []
        
    def reset(self):
        self.ingredients = []
        self.dietary     = []
        self.cuisine     = []
        self.results     = pd.DataFrame()
        self.max_missing = HARD_MAX_MISSING
        self.turn        = 0

# ──────────────────────────────────────────────────────────────
# Ingredient resolution pipeline (spell-correct → snap)
# ──────────────────────────────────────────────────────────────

def resolve_ingredients(raw_list: list[str], spell, dataset_phrases: set[str],
                         snap_cutoff: float) -> tuple[list[str], list[str]]:
    """
    Returns (resolved_ingredients, correction_messages).
    """
    resolved, messages = [], []
    for raw in raw_list:
        corrected, changes, _ = token_correct(spell, raw)
        corrected = normalize(corrected)
        snapped, snap_target = snap_to_phrase(corrected, dataset_phrases, cutoff=snap_cutoff)

        notes = []
        if changes:
            notes.append("spell: " + ", ".join(f"{a}→{b}" for a, b in changes[:4]))
        if snap_target:
            notes.append(f"matched dataset phrase '{snap_target}'")
        if notes:
            messages.append(f"   ✍️  '{raw}' → '{snapped}'  ({'; '.join(notes)})")

        resolved.append(snapped)
    return resolved, messages

# ──────────────────────────────────────────────────────────────
# Main agent loop
# ──────────────────────────────────────────────────────────────

def run_agent(args):
    # ── Bootstrap ──────────────────────────────────────────────
    df = load_and_prepare(args.file, ingredient_col=args.ingredient_col)
    dataset_phrases, dataset_words_fresh = build_dataset_vocab(df, args.ingredient_col)

    dict_path = Path(args.dict_path)
    if args.retrain:
        save_dataset_words(dataset_words_fresh, dict_path)
        dataset_words = dataset_words_fresh
        print("🧠 Dictionary rebuilt from dataset.")
    else:
        if not dict_path.exists():
            sys.exit(f"❌  Trained dictionary not found: {dict_path}\n"
                     f"    Run once with --retrain to build it.")
        dataset_words = load_dataset_words(dict_path)
        dataset_phrases, _ = build_dataset_vocab(df, args.ingredient_col)

    spell  = None if args.no_spell else init_spell(dataset_words)
    weights = load_weights(WEIGHTS_PATH)
    state   = AgentState()

    print()
    print("╔══════════════════════════════════════════════╗")
    print("║        👨‍🍳  ChefGPT Agent  v2.0             ║")
    print("╠══════════════════════════════════════════════╣")
    print("║  Tell me what ingredients you have and I'll  ║")
    print("║  find the best recipe for you.               ║")
    print("║  Type 'help' for all commands.               ║")
    print("╚══════════════════════════════════════════════╝\n")

    # ── Agent loop ─────────────────────────────────────────────
    while True:
        state.turn += 1
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋  Goodbye! Happy cooking.")
            return

        if not user_input:
            continue

        # ── THOUGHT: parse intent ──────────────────────────────
        intent = extract_intent(user_input)

        # ── ACTION: dispatch ───────────────────────────────────

        if intent["quit"]:
            print("\n👋  Goodbye! Happy cooking.\n")

            # 🔥 Compute metrics
            if state.total_suggestions > 0:
                success_rate = state.accepted_suggestions / state.total_suggestions
            else:
                success_rate = 0

            if state.accepted_ranks:
                avg_rank = sum(state.accepted_ranks) / len(state.accepted_ranks)
            else:
                avg_rank = 0

            # 🔥 Display results
            print("📊 Evaluation Metrics:")
            print(f"   • Total suggestions shown: {state.total_suggestions}")
            print(f"   • Accepted suggestions: {state.accepted_suggestions}")
            print(f"   • Success Rate: {success_rate:.2f}")
            print(f"   • Average Accepted Rank: {avg_rank:.2f}\n")

            return

        if intent["help"]:
            print(HELP_TEXT)
            continue

        if intent["clear_mode"]:
            state.reset()
            print("🔄  Cleared! Tell me what ingredients you have.\n")
            continue

        if intent["relax"]:
            state.max_missing += 1
            print(f"🔓  Relaxed: now allowing up to {state.max_missing} missing ingredients.\n")
            if not state.ingredients:
                print("   (No ingredients set yet — tell me what you have first.)\n")
                continue
            # Fall through to re-search below

        if intent["reject"] is not None:
            idx = intent["reject"]
            if state.results.empty or idx < 1 or idx > len(state.results):
                print("⚠️   No suggestion at that number.")
                continue
            row = state.results.iloc[idx - 1]
            weights = update_weights(weights, row["match_set"], row["missing_set"],
                                     accepted=False, lr_pos=args.lr_pos, lr_neg=args.lr_neg)
            save_weights(weights, WEIGHTS_PATH)
            log_feedback(row, False, args.name_col)
            recipe_name = row.get(args.name_col, "that recipe")
            print(f"👎  Got it — noted that [{idx}] {recipe_name} wasn't what you wanted.")
            print(f"   I'll deprioritise those ingredients next time.\n")
            continue

        if intent["accept"] is not None:     
            idx = intent["accept"]
            if state.results.empty or idx < 1 or idx > len(state.results):
                print("⚠️   No suggestion at that number.")
                continue
            row = state.results.iloc[idx - 1]
            # 🔥 Update evaluation metrics
            state.accepted_suggestions += 1
            state.accepted_ranks.append(idx)
            weights = update_weights(weights, row["match_set"], row["missing_set"],
                                     accepted=True, lr_pos=args.lr_pos, lr_neg=args.lr_neg)
            save_weights(weights, WEIGHTS_PATH)
            log_feedback(row, True, args.name_col)
            print(f"✅  Great choice! Here's the full recipe:\n")
            print_recipe(row, name_col=args.name_col,
                         measure_col=args.measure_col, steps_col=args.steps_col)
            # Ask if they want to search again
            print("Want to search again? Tell me your ingredients, or type 'quit'.\n")
            state.results = pd.DataFrame()
            continue

        # ── Ingredient update ──────────────────────────────────
        if intent["raw_ingredients"]:
            resolved, corrections = resolve_ingredients(
                intent["raw_ingredients"], spell, dataset_phrases, args.snap_cutoff
            )
            if corrections:
                print("\n".join(corrections))

            if intent["add_mode"] and state.ingredients:
                added = [r for r in resolved if r not in state.ingredients]
                state.ingredients.extend(added)
                if added:
                    print(f"➕  Added: {', '.join(added)}")
            else:
                state.ingredients = resolved

        # Update preferences
        if intent["dietary"]:
            new = [d for d in intent["dietary"] if d not in state.dietary]
            state.dietary.extend(new)
            if new:
                print(f"🥗  Noted preference{'s' if len(new)>1 else ''}: {', '.join(new)}")
        if intent["cuisine"]:
            new = [c for c in intent["cuisine"] if c not in state.cuisine]
            state.cuisine.extend(new)
            if new:
                print(f"🌍  Noted cuisine: {', '.join(new)}")

        # ── Validate we have something to search ───────────────
        if not state.ingredients:
            print("\n🤔  I don't see any ingredients yet. What do you have in your kitchen?\n")
            continue

        # ── OBSERVATION: run search ────────────────────────────
        print(f"\n🔍  Thinking... (ingredients: {', '.join(state.ingredients)})")
        if state.dietary or state.cuisine:
            filters = state.dietary + state.cuisine
            print(f"   Filters: {', '.join(filters)}")

        results = search(
            state.ingredients, df, weights,
            ingredient_col=args.ingredient_col,
            hard_min_matches=args.hard_min_matches,
            max_missing=state.max_missing,
            dietary=state.dietary,
            cuisine=state.cuisine,
            alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        )

        # ── AGENT REASONING: interpret results ─────────────────
        if results.empty:
            print("\n😕  I couldn't find recipes that match those constraints.")
            _suggest_recovery(state, args)
            continue

        top_n = min(5, len(results))
        state.results = results.head(top_n).reset_index(drop=True)
        # 🔥 Count suggestions shown to user
        state.total_suggestions += len(state.results)
        print(f"\n✨  Found {len(results)} matching recipes. Here are my top {top_n}:\n")
        for rank, (_, row) in enumerate(state.results.iterrows(), start=1):
            print(explain_suggestion(row, rank, args.name_col, weights))
            print()

        print("👉  Pick one (e.g. '1'), reject one (e.g. 'r 2'), or add more ingredients.\n")


def _suggest_recovery(state: AgentState, args):
    """
    Agent proactively suggests how to get better results — replanning step.
    """
    print("\n💭  Let me think about why…")
    if state.max_missing < 4:
        print(f"   • Try 'show more' to allow up to {state.max_missing + 1} missing ingredients.")
    if state.dietary or state.cuisine:
        print(f"   • Your filters ({', '.join(state.dietary + state.cuisine)}) may be limiting results.")
        print("     Say 'start over' to reset filters.")
    if len(state.ingredients) < 3:
        print("   • More ingredients = better matches. What else do you have?")
    print()

# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ChefGPT Agent — conversational recipe assistant")
    p.add_argument("--file",            default="sampled_recipes.xlsx")
    p.add_argument("--ingredient-col",  default="ingredients")
    p.add_argument("--name-col",        default="name")
    p.add_argument("--measure-col",     default="ingredients_measurement")
    p.add_argument("--steps-col",       default="steps")
    p.add_argument("--dict-path",       default=str(DICT_PATH))
    p.add_argument("--retrain",         action="store_true")
    p.add_argument("--no-spell",        action="store_true")
    p.add_argument("--snap-cutoff",     type=float, default=0.86)
    p.add_argument("--no-learn",        action="store_true")
    p.add_argument("--lr-pos",          type=float, default=1.0)
    p.add_argument("--lr-neg",          type=float, default=0.25)
    p.add_argument("--alpha",           type=float, default=0.6)
    p.add_argument("--beta",            type=float, default=0.5)
    p.add_argument("--gamma",           type=float, default=0.4)
    p.add_argument("--hard-min-matches",type=int,   default=2)
    return p.parse_args()


if __name__ == "__main__":
    run_agent(parse_args())