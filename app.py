# app.py

import streamlit as st
import pandas as pd
import tempfile
from pathlib import Path
from ChefGPT_final import (
    load_and_prepare, greedy_search, load_dataset_words, build_dataset_vocab,
    init_spellchecker_from_words, token_correct, snap_to_dataset_phrase,
    load_weights, save_weights, update_weights_for_feedback,
    normalize_ingredient, WEIGHTS_PATH, DICT_PATH, safe_eval_list
)

st.set_page_config(page_title="ChefGPT Streamlit", layout="wide")
st.title("👨‍🍳 ChefGPT — AI Recipe Recommender")

# Upload dataset
uploaded_file = st.file_uploader("Upload your recipe dataset (.xlsx or .csv)", type=["xlsx", "csv"])

if uploaded_file is not None:
    try:
        # Reset the file pointer to the beginning
        uploaded_file.seek(0)

        # Save the uploaded file temporarily so load_and_prepare can read it by path
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        # Load and prepare dataset from the saved file path
        df = load_and_prepare(temp_path)
        st.success(f"Dataset loaded successfully: {len(df)} recipes found.")

        dataset_phrases, dataset_words = build_dataset_vocab(df)
        spell = init_spellchecker_from_words(dataset_words)
        weights = load_weights(WEIGHTS_PATH)

        st.subheader("Enter your available ingredients:")
        raw_input = st.text_area("One ingredient per line", height=200)

        if st.button("Find Recipes"):
            user_inputs = [i.strip() for i in raw_input.split("\n") if i.strip()]
            corrected_list = []

            for ing in user_inputs:
                corrected, changes, toks = token_correct(spell, ing)
                snapped, _ = snap_to_dataset_phrase(corrected, dataset_phrases)
                corrected_list.append(snapped)

            results = greedy_search(corrected_list, df, weights)

            if results.empty:
                st.warning("No recipes found.")
            else:
                st.write(f"Found {len(results)} matching recipes.")
                
                # Store results in session state
                st.session_state.results_df = results
                st.session_state.show_full_recipe = {}

                for i, (_, row) in enumerate(results.head(10).iterrows(), start=1):
                    with st.expander(f"{i}. {row.get('name', 'Unknown')}"):
                        st.write(f"✅ Matched: {', '.join(row['match_set']) or '—'}")
                        st.write(f"❌ Missing: {', '.join(row['missing_set']) or '—'}")
                        st.write(f"⭐ Score: {row['score']:.3f}")

                        col1, col2 = st.columns(2)

                        # Initialize session state for this recipe
                        if f'show_recipe_{i}' not in st.session_state:
                            st.session_state[f'show_recipe_{i}'] = False

                        if col1.button("👍 Good", key=f"good{i}"):
                            weights = update_weights_for_feedback(
                                weights, row["match_set"], row["missing_set"], accepted=True
                            )
                            save_weights(weights, WEIGHTS_PATH)
                            st.session_state[f'show_recipe_{i}'] = True
                            st.success("Feedback saved — weights updated!")
                            st.balloons()

                        if col2.button("👎 Bad", key=f"bad{i}"):
                            weights = update_weights_for_feedback(
                                weights, row["match_set"], row["missing_set"], accepted=False
                            )
                            save_weights(weights, WEIGHTS_PATH)
                            st.warning("Negative feedback recorded.")

                        # Show full recipe if user clicked "Good"
                        if st.session_state[f'show_recipe_{i}']:
                            st.markdown("---")
                            st.subheader("📖 Full Recipe")
                            st.write(f"**🍽️ {row.get('name', 'Unknown')}**")

                            # Show ingredients with measurements
                            if 'ingredients_measurement' in row and pd.notna(row['ingredients_measurement']):
                                st.subheader("📝 Ingredients:")
                                measures = safe_eval_list(row['ingredients_measurement'])
                                for j, measure in enumerate(measures, 1):
                                    st.write(f"{j}. {measure}")
                            else:
                                # Fallback to basic ingredients
                                st.subheader("📝 Ingredients:")
                                for j, ingredient in enumerate(row['ingredients'], 1):
                                    st.write(f"{j}. {ingredient}")

                            # Show preparation steps
                            if 'steps' in row and pd.notna(row['steps']):
                                st.subheader("👣 Preparation Steps:")
                                steps = safe_eval_list(row['steps'])
                                for j, step in enumerate(steps, 1):
                                    st.write(f"{j}. {step}")
                            else:
                                st.info("No preparation steps available for this recipe.")

    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.info("Please make sure you've uploaded a valid CSV or Excel file with the correct columns.")
else:
    st.info("Upload a dataset to start.")
