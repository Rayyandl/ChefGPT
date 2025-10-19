import os
import json
import ast
import re
from pathlib import Path
from datetime import datetime
import pandas as pd
import tkinter as tk
from tkinter import messagebox

# ============== Config ==============
DATA_FILE = "sampled_recipes.xlsx"           # your sample excel
ING_COL = "ingredients"
NAME_COL = "name"
MEASURE_COL = "ingredients_measurement"
STEPS_COL = "steps"

WEIGHTS_PATH = Path("ingredient_weights.json")
FEEDBACK_LOG = Path("feedback_log.csv")

HARD_MAX_MISSING = 3        # "at most 3 missing"
HARD_MIN_MATCHES = 1        # require at least one match

# Scoring knobs
ALPHA = 0.6   # base match fraction weight
BETA  = 0.5   # sum of learned weights weight
GAMMA = 0.4   # penalty per missing

# Learning knobs
LR_POS   = 1.0    # positive boost
LR_NEG   = 0.25   # negative penalty
CLIP_MIN = -5.0
CLIP_MAX = 10.0

WORD_RE = re.compile(r"[a-z']+")

# ============== Utils ==============
def normalize_ingredient(s):
    if s is None:
        return ""
    return str(s).strip().lower()

def safe_eval_set(s):
    """Parse strings like "['butter','flour']" or 'butter, flour' to a set[str] (normalized)."""
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
    """Parse steps/measurements to list[str]."""
    if s is None:
        return []
    if isinstance(s, (list, tuple, set)):
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
    parts = [p.strip().strip("\"'") for p in parts if p and p.strip()]
    return parts if parts else ([text] if text else [])

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
    lr_pos=LR_POS,
    lr_neg=LR_NEG,
    clip_min=CLIP_MIN,
    clip_max=CLIP_MAX,
):
    """Good = +matched, -missing; Bad = -matched."""
    if accepted:
        for m in matched_set:
            weights[m] = max(clip_min, min(clip_max, weight_of(m, weights) + lr_pos))
        for miss in missing_set:
            weights[miss] = max(clip_min, min(clip_max, weight_of(miss, weights) - lr_neg))
    else:
        for m in matched_set:
            weights[m] = max(clip_min, min(clip_max, weight_of(m, weights) - lr_neg))
    return weights

def log_feedback(row, accepted: bool):
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    is_new = not FEEDBACK_LOG.exists()
    with FEEDBACK_LOG.open("a", encoding="utf-8") as f:
        if is_new:
            f.write("timestamp,action,recipe,match_count,missing_count,matched,missing,total\n")
        line = (
            f"{datetime.now().isoformat(timespec='seconds')},"
            f"{'accept' if accepted else 'reject'},"
            f"\"{str(row.get(NAME_COL, 'Unknown')).replace('\"','\"\"')}\","
            f"{row['match_count']},{row['missing_count']},"
            f"\"{', '.join(sorted(row['match_set']))}\","
            f"\"{', '.join(sorted(row['missing_set']))}\","
            f"{row['total_ingredients']}\n"
        )
        f.write(line)

# ============== Data load + index ==============
def load_dataset(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(file_path, dtype=str, keep_default_na=False)
    elif ext in (".csv", ".txt"):
        df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
    else:
        raise ValueError("Unsupported file type. Use .csv or .xlsx")

    if ING_COL not in df.columns:
        raise ValueError(f"Missing column '{ING_COL}'. Available: {list(df.columns)}")

    # Normalize and cache parsed fields
    df = df.copy()
    df[ING_COL] = df[ING_COL].apply(safe_eval_set)
    # Precompute length for faster scoring
    df["total_ingredients"] = df[ING_COL].apply(len)
    return df

def build_inverted_index(df: pd.DataFrame) -> dict[str, set]:
    """token -> set(row_idx). Speeds up candidate retrieval."""
    index: dict[str, set] = {}
    for i, s in df[ING_COL].items():
        for token in s:
            index.setdefault(token, set()).add(i)
    return index

# ============== Search + Score ==============
def compute_score(match_count, total_ingredients, match_set, missing_count, weights):
    base = match_count / max(1, total_ingredients)
    boost = sum(weight_of(x, weights) for x in match_set)
    penalty = GAMMA * missing_count
    return ALPHA * base + BETA * boost - penalty

def greedy_search(user_ingredients: list[str], df: pd.DataFrame, idx: dict[str, set], weights: dict) -> pd.DataFrame:
    user_set = {normalize_ingredient(x) for x in user_ingredients if x.strip()}
    if not user_set:
        return pd.DataFrame()

    # Candidate union from inverted index
    candidate_rows: set[int] = set()
    for tok in user_set:
        candidate_rows |= idx.get(tok, set())

    if not candidate_rows:
        # nothing matches; return empty to avoid noise
        return pd.DataFrame()

    rows = []
    for i in candidate_rows:
        ing_set = df.at[i, ING_COL]
        match_set = ing_set & user_set
        missing_set = ing_set - user_set
        match_count = len(match_set)
        missing_count = len(missing_set)

        # Hard filters
        if missing_count > HARD_MAX_MISSING or match_count < HARD_MIN_MATCHES:
            continue

        total = df.at[i, "total_ingredients"]
        score = compute_score(match_count, total, match_set, missing_count, weights)

        rows.append({
            "row_id": i,
            NAME_COL: df.at[i, NAME_COL] if NAME_COL in df.columns else "Unknown",
            "match_set": match_set,
            "missing_set": missing_set,
            "match_count": match_count,
            "missing_count": missing_count,
            "total_ingredients": total,
            "score": score
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values(
        by=["score", "missing_count", "match_count", "total_ingredients"],
        ascending=[False, True, False, True]
    ).reset_index(drop=True)
    return out

# ============== GUI ==============
class ChefGPTApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🍳 ChefGPT Recipe Finder")
        self.root.geometry("900x640")

        # State
        self.df = load_dataset(DATA_FILE)
        self.index = build_inverted_index(self.df)
        self.weights = load_weights(WEIGHTS_PATH)
        self.results = pd.DataFrame()

        # Top input
        tk.Label(root, text="Enter ingredients (comma-separated):", font=("Arial", 12)).pack(pady=(10, 4))
        self.entry = tk.Entry(root, width=90, font=("Arial", 12))
        self.entry.pack(pady=(0, 6))
        tk.Button(root, text="Search Recipes", font=("Arial", 12, "bold"),
                  command=self.on_search).pack(pady=(0, 10))

        # Scrollable results
        container = tk.Frame(root)
        container.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(container)
        self.scrollbar = tk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.scrollframe = tk.Frame(self.canvas)

        self.scrollframe.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollframe, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Footer note
        tk.Label(root, text=f"Rules: ≥{HARD_MIN_MATCHES} match & ≤{HARD_MAX_MISSING} missing | Weights auto-saved to {WEIGHTS_PATH}",
                 font=("Arial", 10), fg="#666").pack(pady=6)

    # ---------- GUI helpers ----------
    def clear_results(self):
        for w in self.scrollframe.winfo_children():
            w.destroy()

    def on_search(self):
        raw = self.entry.get().strip()
        if not raw:
            messagebox.showwarning("Input needed", "Please enter at least one ingredient.")
            return
        user_ings = [x.strip() for x in raw.split(",") if x.strip()]
        self.results = greedy_search(user_ings, self.df, self.index, self.weights)
        self.render_results()

    def render_results(self):
        self.clear_results()
        if self.results.empty:
            tk.Label(self.scrollframe, text="😕 No recipes found for your criteria.",
                     font=("Arial", 12)).pack(pady=10)
            return

        for idx, row in self.results.iterrows():
            card = tk.Frame(self.scrollframe, bd=2, relief="groove", padx=12, pady=8, bg="#fffbe6")
            card.pack(fill="x", padx=8, pady=6)

            # Title
            tk.Label(card, text=f"[{idx+1}] 🍽️ {row.get(NAME_COL, 'Unknown')}",
                     font=("Arial", 14, "bold"), bg="#fffbe6").pack(anchor="w")

            # Matched / Missing
            matched = ", ".join(sorted(row["match_set"])) or "—"
            missing = ", ".join(sorted(row["missing_set"])) or "—"
            tk.Label(card, text=f"✅ Matched: {matched}", font=("Arial", 12), fg="green", bg="#fffbe6").pack(anchor="w")
            tk.Label(card, text=f"❌ Missing: {missing}", font=("Arial", 12), fg="red", bg="#fffbe6").pack(anchor="w")

            # Score + counts
            tk.Label(card, text=f"⭐ {row['match_count']} matched / {row['total_ingredients']} total    "
                                f"🧮 Score: {row['score']:.3f}",
                     font=("Arial", 12), bg="#fffbe6").pack(anchor="w", pady=(2, 0))

            # Buttons
            btnbar = tk.Frame(card, bg="#fffbe6")
            btnbar.pack(anchor="e", pady=(6, 0))
            tk.Button(btnbar, text="👍 Good", width=10,
                      command=lambda i=idx: self.on_feedback(i, True), bg="#c6f7d0").pack(side="left", padx=5)
            tk.Button(btnbar, text="👎 Bad", width=10,
                      command=lambda i=idx: self.on_feedback(i, False), bg="#f7c6c6").pack(side="left", padx=5)

    def on_feedback(self, idx: int, accepted: bool):
        if idx < 0 or idx >= len(self.results):
            return
        row = self.results.iloc[idx]
        # Update weights
        self.weights = update_weights_for_feedback(
            self.weights,
            matched_set=row["match_set"],
            missing_set=row["missing_set"],
            accepted=accepted
        )
        save_weights(self.weights, WEIGHTS_PATH)
        log_feedback(row, accepted=accepted)

        if accepted:
            # Show full recipe steps + measurements
            self.show_full_recipe(int(row["row_id"]))
        else:
            messagebox.showinfo("Feedback", f"Recorded 👎 for “{row.get(NAME_COL, 'Unknown')}”. Model updated.")

    def show_full_recipe(self, row_id: int):
        name = self.df.at[row_id, NAME_COL] if NAME_COL in self.df.columns else "Unknown"
        measures = safe_eval_list(self.df.at[row_id, MEASURE_COL] if MEASURE_COL in self.df.columns else [])
        steps = safe_eval_list(self.df.at[row_id, STEPS_COL] if STEPS_COL in self.df.columns else [])

        text = f"🍽️ {name}\n\n"
        text += "🧂 Ingredients + Measurement:\n"
        if measures:
            for m in measures:
                text += f"  - {m}\n"
        else:
            text += "  (none)\n"
        text += "\n👣 Steps:\n"
        if steps:
            for i, s in enumerate(steps, 1):
                text += f"  {i}. {s}\n"
        else:
            text += "  (none)\n"

        messagebox.showinfo("Full Recipe", text)

# ============== Run GUI ==============
if __name__ == "__main__":
    root = tk.Tk()
    app = ChefGPTApp(root)
    root.mainloop()
