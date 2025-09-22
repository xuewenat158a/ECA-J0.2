# eca_rf_j02_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import inspect
from io import BytesIO
from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.inspection import permutation_importance
import joblib

# ---------------- Page ----------------
st.set_page_config(page_title="ECA RF — J0.2 (5 inputs)", layout="wide")
st.title("ECA Random Forest — Predict J0.2 (5 Inputs)")

with st.expander("About this app", expanded=False):
    st.markdown(
        "- Trains a Random Forest to predict **J0.2** from 5 inputs.\\n"
        "- Handles categorical inputs via version-proof OneHotEncoder.\\n"
        "- Shows test metrics, **OOB R²**, aggregated permutation importance, diagnostics with legend, and a Model Card.\\n"
        "- Try single prediction or upload a batch of feature rows."
    )

# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
DEFAULT_FILE = "ECA ML.xlsx"
input_choice = st.sidebar.selectbox(
    "Choose data source", ("Use bundled path", "Upload Excel file (.xlsx)")
)
if input_choice == "Use bundled path":
    input_path = st.sidebar.text_input("Excel path", value=DEFAULT_FILE)
    sheet_name = st.sidebar.text_input("Sheet name", value="ML database")
    uploaded_file = None
else:
    uploaded_file = st.sidebar.file_uploader("Upload .xlsx", type=["xlsx"])
    sheet_name = st.sidebar.text_input("Sheet name", value="ML database")
    input_path = None

test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
n_estimators = st.sidebar.slider("n_estimators", 200, 1000, 500, 50)
random_state = st.sidebar.number_input("random_state", min_value=0, value=42, step=1)
run_cv = st.sidebar.checkbox("Run 5-fold CV (average R²)", value=False)

st.sidebar.markdown("---")
do_save = st.sidebar.checkbox("Save trained model", value=True)
model_filename = st.sidebar.text_input("Model filename", value="eca_rf_j02.joblib")

# ---------------- Columns (your locked schema) ----------------
# Preview order from raw sheet (exact names)
PREVIEW_ORDER = [
    "Sour region", "pH", "ppH2S(bara)", "K-rate", "Notch Location", "J0.2"
]

# Training canonical names after renaming
RAW_FEATURES = ["Sour region", "pH", "ppH2S(bara)", "K-rate", "Notch Location"]
CANON_FEATURES = ["Sour Region", "pH", "ppH2S", "K-rate", "Notch Location"]
CATEGORICAL_FEATS = ["Sour Region", "Notch Location"]
TARGET_COL = "J0.2"

# ---------------- Helpers ----------------
def make_ohe():
    # Version-proof OneHotEncoder
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

@st.cache_data(show_spinner=False)
def load_excel_raw(_uploaded_file, _input_path, _sheet):
    if _uploaded_file is not None:
        df_raw = pd.read_excel(_uploaded_file, sheet_name=_sheet)
    else:
        df_raw = pd.read_excel(_input_path, sheet_name=_sheet)
    # strip hidden chars/whitespace but DO NOT change header semantics
    df_raw.columns = [c.strip().replace("\\u200b", "") for c in df_raw.columns]
    return df_raw

def apply_training_renames(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.rename(
        columns={
            "Sour region": "Sour Region",
            "ppH2S(bara)": "ppH2S",
            "a/w": "a/W",
            "Test Pressure(bara)": "Test Pressure",
        },
        inplace=True,
    )
    return df

def get_all_feature_names(preprocessor, X_sample, num_cols, cat_cols):
    # If not fitted (shouldn't happen in this flow), approximate width
    if not hasattr(preprocessor, "transformers_"):
        dim = len(num_cols)
        for c in cat_cols:
            dim += len(pd.Series(X_sample[c]).dropna().unique())
        names = [f"feat__{i}" for i in range(dim)]
        return names, names

    try:
        full = preprocessor.get_feature_names_out()
    except Exception:
        full = []
        for c in num_cols:
            full.append(f"num__{c}")
        if "cat" in preprocessor.named_transformers_:
            ohe = preprocessor.named_transformers_["cat"]
            try:
                ohe_names = ohe.get_feature_names_out(cat_cols)
            except Exception:
                ohe_names = []
                for col in cat_cols:
                    n_cat = len(pd.Series(X_sample[col]).dropna().unique())
                    ohe_names.extend([f"{col}_{i}" for i in range(n_cat)])
            full.extend([f"cat__{n}" for n in ohe_names])

    clean = [n.split("__", 1)[1] if "__" in n else n for n in full]

    try:
        width = preprocessor.transform(X_sample.iloc[:1]).shape[1]
    except Exception:
        width = len(full)

    if len(full) != width:
        full = [f"feat__{i}" for i in range(width)]
        clean = full[:]

    return full, clean

@st.cache_resource(show_spinner=True)
def train_model(X, y, num_cols, cat_cols, test_size, n_estimators, random_state):
    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", make_ohe(), cat_cols),
        ]
    )
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        oob_score=True,
        bootstrap=True,
    )
    pipe = Pipeline([("prep", preprocess), ("model", rf)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    pipe.fit(X_train, y_train)
    return pipe, (X_train, X_test, y_train, y_test)

# ---------------- Load data ----------------
try:
    df_raw = load_excel_raw(uploaded_file, input_path, sheet_name)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Preview EXACTLY your requested columns and order
st.subheader("Preview data (as provided)")
available_cols = [c for c in PREVIEW_ORDER if c in df_raw.columns]
if not available_cols:
    st.warning("None of the expected preview columns were found. Showing first 20 rows instead.")
    st.dataframe(df_raw.head(20), use_container_width=True)
else:
    st.dataframe(df_raw.loc[:, available_cols].head(20), use_container_width=True)

# Canonical training names
df = apply_training_renames(df_raw)

# Check presence
missing_feats = [c for c in CANON_FEATURES if c not in df.columns]
missing_tgt = TARGET_COL not in df.columns
if missing_feats:
    st.warning(f"Missing required feature columns (after renaming): {missing_feats}")
if missing_tgt:
    st.error(f"Missing required target column: {TARGET_COL}")
    st.stop()

# Build data (drop NA rows in features/target)
feature_cols = CANON_FEATURES[:]   # locked 5 inputs
cat_cols = [c for c in CATEGORICAL_FEATS if c in feature_cols]
num_cols = [c for c in feature_cols if c not in cat_cols]

rows_before = len(df)
data = df[feature_cols + [TARGET_COL]].dropna(axis=0).reset_index(drop=True)
rows_after = len(data)
if rows_after < 10:
    st.warning("Very small dataset after dropping NAs; results may be unstable and PI can be noisy.")

X = data[feature_cols]
y = data[TARGET_COL]

# ---------------- Train ----------------
train_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
pipe, (X_train, X_test, y_train, y_test) = train_model(
    X, y, num_cols, cat_cols, test_size, n_estimators, random_state
)
st.success("Model trained.")

# ---------------- Evaluation ----------------
y_pred = pipe.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("R² (test)", f"{r2:.3f}")
with col2:
    st.metric("RMSE (test)", f"{rmse:.3f}")
with col3:
    oob = getattr(pipe.named_steps["model"], "oob_score_", None)
    st.metric("OOB R²", f"{oob:.3f}" if oob is not None else "N/A")

# Optional CV
if run_cv:
    with st.spinner("Running 5-fold CV..."):
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        scores = cross_validate(pipe, X, y, cv=kf, scoring="r2", n_jobs=-1)
    st.write(
        f"CV R² (mean ± std): **{scores['test_score'].mean():.3f} ± {scores['test_score'].std():.3f}**"
    )

# ---------------- Permutation Importance (aggregated) ----------------
with st.spinner("Computing permutation importance..."):
    res = permutation_importance(
        pipe, X_test, y_test, n_repeats=10, random_state=random_state, n_jobs=-1
    )

full_names, clean_names = get_all_feature_names(
    pipe.named_steps["prep"], X_test, num_cols, cat_cols
)

def base_feature(clean):
    for c in cat_cols:
        if clean.startswith(c + "_"):
            return c
    return clean

pi_df = pd.DataFrame({
    "full_name": full_names[:len(res.importances_mean)],
    "clean_name": clean_names[:len(res.importances_mean)],
    "importance_mean": res.importances_mean,
    "importance_std": res.importances_std,
})
pi_df["base_feature"] = pi_df["clean_name"].apply(base_feature)
agg = (pi_df.groupby("base_feature", as_index=False)["importance_mean"]
          .sum()
          .sort_values("importance_mean", ascending=False))

st.subheader("Permutation Importance (aggregated to original features)")
st.dataframe(agg, use_container_width=True)

fig, ax = plt.subplots(figsize=(8, max(3, len(agg)*0.4)))
ax.barh(agg["base_feature"], agg["importance_mean"])
ax.invert_yaxis()
ax.set_title("Permutation Importance — J0.2")
ax.set_xlabel("Mean importance (validation)")
ax.set_ylabel("Feature")
st.pyplot(fig, clear_figure=True)

# ---------------- Diagnostics: Actual vs Predicted ----------------
with st.expander("Diagnostics: Actual vs Predicted (J0.2)", expanded=False):
    fig2, ax2 = plt.subplots()
    ax2.scatter(y_test, y_pred, label="Data points")
    ax2.set_xlabel("Actual J0.2")
    ax2.set_ylabel("Predicted J0.2")
    ax2.set_title("Actual vs Predicted — J0.2")
    mn = min(ax2.get_xlim()[0], ax2.get_ylim()[0])
    mx = max(ax2.get_xlim()[1], ax2.get_ylim()[1])
    ax2.plot([mn, mx], [mn, mx], label="Ideal (y = x)")
    ax2.legend(title="Legend", loc="best")
    st.pyplot(fig2, clear_figure=True)

st.markdown("---")

# ---------------- Model Card ----------------
with st.expander("Model Card", expanded=True):
    info = {
        "Timestamp": train_time,
        "Data rows (before dropna)": rows_before,
        "Data rows (after dropna)": rows_after,
        "Train/Test Split": f"{1 - test_size:.0%} / {test_size:.0%}",
        "Features (locked)": ", ".join(feature_cols),
        "Categorical": ", ".join(cat_cols) if len(cat_cols) else "(none)",
        "Numeric": ", ".join(num_cols) if len(num_cols) else "(none)",
        "Target": TARGET_COL,
        "n_estimators": n_estimators,
        "random_state": random_state,
        "Bootstrap": True,
        "OOB R²": f"{getattr(pipe.named_steps['model'], 'oob_score_', float('nan')):.3f}",
        "R² (test)": f"{r2:.3f}",
        "RMSE (test)": f"{rmse:.3f}",
        "CV run?": "Yes" if run_cv else "No",
    }
    df_info = pd.DataFrame(list(info.items()), columns=["Field", "Value"])
    st.table(df_info)

    # Quick export
    mc_csv = df_info.to_csv(index=False).encode("utf-8")
    st.download_button("Download Model Card (CSV)", data=mc_csv, file_name="model_card.csv", mime="text/csv")

st.markdown("---")

# ---------------- Save model ----------------
if do_save:
    try:
        joblib.dump(pipe, model_filename)
        st.success(f"Saved model to: {model_filename}")
        with open(model_filename, "rb") as f:
            st.download_button("Download model file", f, file_name=model_filename)
    except Exception as e:
        st.warning(f"Could not save model: {e}")

# ---------------- Inference (single row) ----------------
st.subheader("Try a single prediction")

if "single_pred_df" not in st.session_state:
    st.session_state.single_pred_df = None

with st.form("single_pred"):
    inputs = {}
    for col in feature_cols:
        if col in cat_cols:
            cats = sorted(pd.Series(X[col]).dropna().unique().tolist())
            if len(cats) == 0:
                st.warning(f"No categories found in training data for '{col}'. Using empty string.")
                cats = [""]
            inputs[col] = st.selectbox(col, options=cats, index=0)
        else:
            series = pd.Series(X[col]).dropna()
            default_val = float(pd.to_numeric(series, errors="coerce").median()) if len(series) else 0.0
            inputs[col] = st.number_input(col, value=default_val, step=0.001, format="%.6f")

    submitted = st.form_submit_button("Predict J0.2")
    if submitted:
        try:
            row = pd.DataFrame([inputs])
            pred = pipe.predict(row)
            st.session_state.single_pred_df = pd.DataFrame({"J0.2_pred": [float(pred[0]) ]})
            st.success("Prediction ready below.")
        except Exception as e:
            st.session_state.single_pred_df = None
            st.error(f"Prediction failed: {e}")

# render results & download OUTSIDE the form
if st.session_state.single_pred_df is not None:
    st.write("Prediction:")
    st.dataframe(st.session_state.single_pred_df, use_container_width=True)
    csv = st.session_state.single_pred_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download prediction (CSV)",
        data=csv,
        file_name="single_prediction_j02.csv",
        mime="text/csv",
        key="single_pred_download"
    )

# ---------------- Batch Inference ----------------
st.subheader("Batch prediction (upload new rows)")
batch_file = st.file_uploader(
    "Upload CSV or Excel with **feature columns only** "
    "(Sour Region, pH, ppH2S, K-rate, Notch Location allowed; raw names OK).",
    type=["csv","xlsx"], key="batch"
)
if batch_file is not None:
    try:
        if batch_file.name.lower().endswith(".csv"):
            newX = pd.read_csv(batch_file)
        else:
            newX = pd.read_excel(batch_file)

        # Accept either raw or canonical names
        newX = newX.rename(columns={
            "Sour region": "Sour Region",
            "ppH2S(bara)": "ppH2S",
            "a/w": "a/W",
            "Test Pressure(bara)": "Test Pressure",
        })

        st.write("Preview of uploaded features:")
        st.dataframe(newX.head(), use_container_width=True)

        # Precompute training fill values (medians for num, modes for cat)
        num_fills = {c: float(pd.to_numeric(X[c], errors="coerce").median()) if X[c].dropna().size else 0.0
                     for c in num_cols}
        cat_fills = {c: (pd.Series(X[c]).mode().iat[0] if not pd.Series(X[c]).mode().empty else "")
                     for c in cat_cols}

        # Ensure required columns exist & coerce types
        for c in feature_cols:
            if c not in newX.columns:
                newX[c] = cat_fills[c] if c in cat_cols else num_fills.get(c, 0.0)

        # Numeric -> numeric + fill
        for c in num_cols:
            newX[c] = pd.to_numeric(newX[c], errors="coerce")
            newX[c].fillna(num_fills[c], inplace=True)

        # Categorical -> fill first, then cast to string (avoids literal "nan")
        for c in cat_cols:
            col = newX[c]
            if col.dtype.name != "string":
                col = col.astype("string")
            col = col.fillna(cat_fills[c])
            newX[c] = col.astype(str)

        preds = pipe.predict(newX[feature_cols])
        preds_df = pd.DataFrame({"J0.2_pred": preds})
        out = pd.concat([newX.reset_index(drop=True), preds_df], axis=1)

        st.write("Predictions:")
        st.dataframe(out.head(50), use_container_width=True)

        out_buf = BytesIO()
        with pd.ExcelWriter(out_buf, engine="openpyxl") as writer:
            out.to_excel(writer, index=False, sheet_name="Predictions")
        st.download_button("Download predictions (Excel)", data=out_buf.getvalue(), file_name="eca_predictions_j02.xlsx")
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")
