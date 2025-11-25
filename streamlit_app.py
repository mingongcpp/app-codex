"""Codex dictionary classifier Streamlit app."""
import json
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Default configuration derived from the prototype script
DEFAULT_TEXT_COL = "Statement"
DEFAULT_TACTIC_DICTIONARY = {
    "scarcity": [
        "last chance",
        "last week",
        "limited time",
        "only a few",
        "before they're gone",
        "while stocks last",
    ],
    "urgency": [
        "today only",
        "now",
        "hurry",
        "right away",
        "don't wait",
        "immediately",
    ],
    "social_proof": [
        "popular",
        "bestseller",
        "customers love",
        "everyone",
        "most people",
        "thousands of",
    ],
    "discount": [
        "discount",
        "sale",
        "off",
        "% off",
        "save",
        "special offer",
        "deal",
    ],
}
SAMPLE_DATA_PATH = Path(__file__).parent / "data" / "sample_data.csv"


# ---------------------------------------------------------------------------
# Helpers

def classify_statement(text: str, dictionary: dict[str, list[str]], return_multiple: bool = False):
    """Return the tactic label(s) for a statement using simple keyword search."""
    if not isinstance(text, str):
        return np.nan

    text_lower = text.lower()
    matched_labels: list[str] = []

    for label, keywords in dictionary.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                matched_labels.append(label)
                break

    if not matched_labels:
        return np.nan

    if return_multiple:
        return ";".join(matched_labels)

    return matched_labels[0]


def parse_dictionary(raw_text: str):
    """Parse the dictionary JSON input and validate its structure."""
    try:
        parsed = json.loads(raw_text)
        if not isinstance(parsed, dict):
            raise ValueError("Dictionary must be a JSON object where keys map to keyword lists.")

        for label, keywords in parsed.items():
            if not isinstance(label, str):
                raise ValueError("Each label must be a string.")
            if not isinstance(keywords, list):
                raise ValueError(f"Keywords for '{label}' must be provided as a list of strings.")
            for kw in keywords:
                if not isinstance(kw, str):
                    raise ValueError(f"Keyword '{kw}' in label '{label}' must be a string.")

        return parsed, None
    except Exception as exc:  # noqa: BLE001
        return DEFAULT_TACTIC_DICTIONARY, str(exc)


def run_classification(df: pd.DataFrame, dictionary: dict[str, list[str]], text_col: str,
                       single_col: str, multi_col: str, include_multi: bool):
    result = df.copy()
    result[single_col] = result[text_col].apply(
        lambda x: classify_statement(x, dictionary, return_multiple=False)
    )

    if include_multi:
        result[multi_col] = result[text_col].apply(
            lambda x: classify_statement(x, dictionary, return_multiple=True)
        )

    return result


@st.cache_data
def load_sample_dataframe():
    if SAMPLE_DATA_PATH.exists():
        return pd.read_csv(SAMPLE_DATA_PATH)
    return None


# ---------------------------------------------------------------------------
# Streamlit UI

def main():
    st.set_page_config(page_title="Codex Dictionary Classifier", page_icon="🧠", layout="wide")

    st.title("🧠 Codex Dictionary Classifier")
    st.write(
        "Convert the original keyword-based classification prototype into a Streamlit experience. "
        "Upload a CSV, edit the dictionary, and download the results with single and multi-label outputs."
    )

    sample_df = load_sample_dataframe()
    if "use_sample_data" not in st.session_state:
        st.session_state.use_sample_data = False

    # Sidebar controls ------------------------------------------------------
    with st.sidebar:
        st.header("Keyword dictionary")
        default_dict_str = json.dumps(DEFAULT_TACTIC_DICTIONARY, indent=2)
        dict_text = st.text_area("JSON dictionary", value=default_dict_str, height=260)
        user_dictionary, dict_error = parse_dictionary(dict_text)
        if dict_error:
            st.error(f"Dictionary error: {dict_error}. Using defaults instead.")
        else:
            st.success("Dictionary parsed successfully ✅")

        if sample_df is not None:
            csv_buffer = StringIO()
            sample_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download sample CSV",
                data=csv_buffer.getvalue().encode("utf-8"),
                file_name="sample_data.csv",
                mime="text/csv",
                help="Grab the included dataset if you need an example.",
            )

            if st.button("Load bundled sample data"):
                st.session_state.use_sample_data = True

        st.caption("Need custom logic? Edit the JSON above to add/remove tactics and keywords.")

    # Upload / sample selection -------------------------------------------
    st.header("1. Provide your dataset")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    df = None
    data_source = ""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            data_source = f"Uploaded file: {uploaded_file.name}"
            st.session_state.use_sample_data = False
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read CSV: {exc}")
            st.stop()
    elif st.session_state.use_sample_data and sample_df is not None:
        df = sample_df.copy()
        data_source = "Bundled sample_data.csv"

    if df is None:
        st.info("Upload a CSV (or click 'Load bundled sample data' in the sidebar) to continue.")
        return

    st.success(f"Data source ready: {data_source}")
    st.dataframe(df.head(), use_container_width=True)

    # Column & option selection -------------------------------------------
    st.header("2. Configure classification")
    columns = list(df.columns)
    default_index = columns.index(DEFAULT_TEXT_COL) if DEFAULT_TEXT_COL in columns else 0
    text_col = st.selectbox("Column containing statements", columns, index=default_index)

    col1, col2 = st.columns(2)
    with col1:
        single_col = st.text_input("Single-label column name", value="Tactic_dict_single")
    with col2:
        multi_col = st.text_input("Multi-label column name", value="Tactic_dict_multi")

    include_multi = st.checkbox(
        "Return multi-label results (semicolon-separated)",
        value=True,
        help="If unchecked only the single-label column will be added.",
    )

    # Run classification ---------------------------------------------------
    st.header("3. Run classifier")
    if st.button("Classify statements", type="primary"):
        with st.spinner("Applying dictionary..."):
            result_df = run_classification(
                df=df,
                dictionary=user_dictionary,
                text_col=text_col,
                single_col=single_col,
                multi_col=multi_col,
                include_multi=include_multi,
            )

        st.success("Classification complete! 🎉")
        preview_cols = [text_col, single_col]
        if include_multi:
            preview_cols.append(multi_col)
        st.dataframe(result_df[preview_cols].head(20), use_container_width=True)

        buffer = StringIO()
        result_df.to_csv(buffer, index=False)
        st.download_button(
            label="📥 Download CSV with labels",
            data=buffer.getvalue().encode("utf-8"),
            file_name="classified_statements.csv",
            mime="text/csv",
        )

        st.subheader("Dictionary used for this run")
        st.json(user_dictionary if dict_error is None else DEFAULT_TACTIC_DICTIONARY)


if __name__ == "__main__":
    main()
