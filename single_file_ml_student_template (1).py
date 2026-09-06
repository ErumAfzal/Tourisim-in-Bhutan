"""
SINGLE-FILE DATA SCIENCE / ML PROJECT TEMPLATE
===============================================

Designed for:
- GitHub classroom projects
- Streamlit deployment
- Google Colab adaptation
- Introductory EDA, visualization and machine learning

STUDENT GOAL
------------
Use ONE Python file to document and demonstrate a complete project:

    Problem
      ↓
    Data
      ↓
    Data Quality
      ↓
    EDA
      ↓
    Visualization
      ↓
    Feature Preparation
      ↓
    Machine Learning
      ↓
    Evaluation
      ↓
    Interpretation
      ↓
    Reflection / Limitations

HOW TO USE THIS TEMPLATE
------------------------
OPTION A — STREAMLIT
1. Copy this file into a GitHub repository.
2. Rename it, for example:
       app.py
3. Edit the STUDENT CONFIGURATION section below.
4. Put a CSV URL into DEFAULT_DATA_URL, OR upload a CSV in the app.
5. Deploy the file as the Streamlit entry point.

Typical local command:
    streamlit run single_file_ml_student_template.py

OPTION B — GOOGLE COLAB
1. Upload this .py file to Colab.
2. In a notebook cell run:
       %run single_file_ml_student_template.py
3. The script will automatically use notebook/console mode if it is not being
   executed by Streamlit.

IMPORTANT DEPENDENCIES
----------------------
Core:
    pandas
    numpy

Streamlit mode:
    streamlit

Optional Colab plotting:
    matplotlib

Machine learning is implemented with NumPy so the template remains highly
portable and the main algorithms can be inspected by students.

SUPPORTED ML TASKS
------------------
1. Regression:
   - Linear Regression
   - Ridge Regression

2. Classification:
   - K-Nearest Neighbours (KNN), implemented from scratch
   - Supports binary or multiclass targets

This is a teaching template, not a production ML framework.
"""

# ============================================================================
# 0. STUDENT CONFIGURATION — EDIT THIS SECTION FIRST
# ============================================================================

PROJECT_TITLE = "My Data Science & Machine Learning Project"

STUDENT_NAME = "YOUR NAME"

PROBLEM_STATEMENT = """
WRITE YOUR PROBLEM HERE.

Example:
Can selected demographic, economic or behavioural variables help us understand
and predict the target outcome?
"""

RESEARCH_QUESTIONS = [
    "Write research question 1 here.",
    "Write research question 2 here.",
    "Write research question 3 here.",
]

# Put a public RAW CSV URL here if you want the app to load data automatically.
# GitHub example:
# https://raw.githubusercontent.com/USERNAME/REPOSITORY/main/data.csv
DEFAULT_DATA_URL = ""

# Set this after you inspect your dataset.
# Examples:
# TARGET_COLUMN = "price"
# TARGET_COLUMN = "species"
TARGET_COLUMN = ""

# Choose:
#   "auto"
#   "regression"
#   "classification"
TASK_TYPE = "auto"

# Student reflection fields shown in Streamlit.
EXPECTED_USERS = "Who would use the analysis or model?"
EXPECTED_VALUE = "What useful decision, understanding or action could it support?"
KNOWN_LIMITATIONS = "What limitations do you already expect?"


# ============================================================================
# 1. IMPORTS
# ============================================================================

import io
import math
import sys
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import streamlit as st
except Exception:
    st = None


# ============================================================================
# 2. ENVIRONMENT DETECTION
# ============================================================================

def running_inside_streamlit() -> bool:
    """
    Detect whether this script is currently executed by Streamlit.

    Why?
    The same .py file can then behave as:
    - an interactive Streamlit app, OR
    - a notebook/console teaching script.
    """
    if st is None:
        return False

    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


STREAMLIT_MODE = running_inside_streamlit()


# ============================================================================
# 3. OPTIONAL DISPLAY HELPERS
# ============================================================================

def notebook_display(obj):
    """
    Use rich display in Jupyter/Colab when available.
    Fall back to print otherwise.
    """
    try:
        from IPython.display import display
        display(obj)
    except Exception:
        print(obj)


def print_section(title: str):
    """Simple console/notebook section heading."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================================
# 4. DATA LOADING
# ============================================================================

def read_csv_from_url(url: str) -> pd.DataFrame:
    """
    Read a CSV from a public URL.

    Using urllib keeps the dependency footprint small.
    """
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 DataScienceTeachingTemplate"},
    )

    with urllib.request.urlopen(request, timeout=15) as response:
        raw = response.read()

    return pd.read_csv(io.BytesIO(raw))


def create_example_dataset() -> pd.DataFrame:
    """
    Create a deterministic built-in dataset so the template always runs.

    IMPORTANT:
    Students should replace this with their own data for the final project.

    Dataset story:
    A fictional tourism study with destination attributes and annual visitors.
    """
    rng = np.random.default_rng(7)
    n = 120

    regions = rng.choice(
        ["North", "South", "East", "West"],
        size=n,
    )

    promotion_budget = rng.normal(60_000, 18_000, n).clip(10_000, None)
    avg_hotel_price = rng.normal(95, 25, n).clip(35, None)
    attraction_score = rng.uniform(2.0, 5.0, n)
    accessibility_score = rng.uniform(1.0, 5.0, n)
    social_media_mentions = rng.normal(15_000, 5_000, n).clip(1_000, None)

    region_effect = pd.Series(regions).map(
        {
            "North": 8_000,
            "South": 14_000,
            "East": 10_000,
            "West": 18_000,
        }
    ).to_numpy()

    visitors = (
        0.30 * promotion_budget
        - 85 * avg_hotel_price
        + 9_000 * attraction_score
        + 5_000 * accessibility_score
        + 0.55 * social_media_mentions
        + region_effect
        + rng.normal(0, 7_500, n)
    ).clip(5_000, None)

    df = pd.DataFrame(
        {
            "region": regions,
            "promotion_budget": promotion_budget.round(0),
            "avg_hotel_price": avg_hotel_price.round(2),
            "attraction_score": attraction_score.round(2),
            "accessibility_score": accessibility_score.round(2),
            "social_media_mentions": social_media_mentions.round(0),
            "annual_visitors": visitors.round(0),
        }
    )

    # Add a small amount of missingness for teaching.
    missing_idx = rng.choice(df.index, size=6, replace=False)
    df.loc[missing_idx[:3], "avg_hotel_price"] = np.nan
    df.loc[missing_idx[3:], "social_media_mentions"] = np.nan

    return df


def load_default_data() -> Tuple[pd.DataFrame, str]:
    """
    Data source priority:
    1. DEFAULT_DATA_URL if provided
    2. Built-in example dataset
    """
    if DEFAULT_DATA_URL.strip():
        try:
            return read_csv_from_url(DEFAULT_DATA_URL.strip()), "Public CSV URL"
        except Exception as exc:
            return (
                create_example_dataset(),
                f"Built-in example dataset because URL loading failed: {exc}",
            )

    return create_example_dataset(), "Built-in example dataset"


# ============================================================================
# 5. DATA AUDIT / EDA HELPERS
# ============================================================================

def data_audit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a compact data dictionary automatically.
    """
    rows = []

    for col in df.columns:
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "non_missing": int(df[col].notna().sum()),
                "missing": int(df[col].isna().sum()),
                "missing_percent": round(df[col].isna().mean() * 100, 2),
                "unique_values": int(df[col].nunique(dropna=True)),
            }
        )

    return pd.DataFrame(rows)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics for numeric columns."""
    numeric = df.select_dtypes(include=np.number)

    if numeric.empty:
        return pd.DataFrame()

    summary = numeric.describe().T
    summary["missing"] = numeric.isna().sum()
    return summary.round(3)


def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summary of categorical columns."""
    categorical = df.select_dtypes(exclude=np.number)

    rows = []
    for col in categorical.columns:
        mode_values = categorical[col].mode(dropna=True)
        most_common = (
            mode_values.iloc[0]
            if not mode_values.empty
            else None
        )

        rows.append(
            {
                "column": col,
                "unique_values": categorical[col].nunique(dropna=True),
                "most_common": most_common,
                "missing": categorical[col].isna().sum(),
            }
        )

    return pd.DataFrame(rows)


def iqr_outlier_count(series: pd.Series) -> int:
    """Count potential numeric outliers with the 1.5*IQR rule."""
    series = pd.to_numeric(series, errors="coerce").dropna()

    if len(series) < 4:
        return 0

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return int(((series < lower) | (series > upper)).sum())


# ============================================================================
# 6. PREPROCESSING
# ============================================================================

def infer_task_type(y: pd.Series, requested: str = "auto") -> str:
    """
    Infer regression vs classification.

    Rule:
    - User choice wins if explicitly regression/classification.
    - Non-numeric target -> classification.
    - Numeric target with few unique values -> classification.
    - Otherwise -> regression.
    """
    if requested in {"regression", "classification"}:
        return requested

    if not pd.api.types.is_numeric_dtype(y):
        return "classification"

    unique_count = y.nunique(dropna=True)
    threshold = max(10, int(len(y) * 0.05))

    if unique_count <= threshold:
        return "classification"

    return "regression"


def clean_features_and_target(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Beginner-friendly preprocessing.

    NUMERIC FEATURES
    - convert to numeric
    - fill missing values with median

    CATEGORICAL FEATURES
    - convert to string
    - fill missing values with mode or 'Missing'
    - one-hot encode with pandas.get_dummies()

    TARGET
    - rows with missing target are removed
    """
    work = df[feature_columns + [target_column]].copy()

    # Remove rows where the target is missing.
    work = work[work[target_column].notna()].copy()

    X = work[feature_columns].copy()
    y = work[target_column].copy()

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [
        col for col in X.columns
        if col not in numeric_cols
    ]

    # Numeric imputation.
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        median_value = X[col].median()

        if pd.isna(median_value):
            median_value = 0.0

        X[col] = X[col].fillna(median_value)

    # Categorical imputation.
    for col in categorical_cols:
        X[col] = X[col].astype("object")

        mode_values = X[col].mode(dropna=True)
        fill_value = (
            mode_values.iloc[0]
            if not mode_values.empty
            else "Missing"
        )

        X[col] = X[col].fillna(fill_value).astype(str)

    # One-hot encoding.
    X = pd.get_dummies(
        X,
        columns=categorical_cols,
        drop_first=False,
        dtype=float,
    )

    # Final numeric conversion.
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return X, y


# ============================================================================
# 7. TRAIN / TEST SPLIT
# ============================================================================

def train_test_split_manual(
    X: pd.DataFrame,
    y: pd.Series,
    test_fraction: float = 0.25,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Reproducible random train/test split without scikit-learn."""
    rng = np.random.default_rng(random_state)

    indices = np.arange(len(X))
    rng.shuffle(indices)

    test_size = max(1, int(round(len(X) * test_fraction)))
    test_size = min(test_size, len(X) - 1)

    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    return (
        X.iloc[train_idx].copy(),
        X.iloc[test_idx].copy(),
        y.iloc[train_idx].copy(),
        y.iloc[test_idx].copy(),
    )


# ============================================================================
# 8. STANDARDIZATION
# ============================================================================

class Standardizer:
    """Minimal standard scaler fitted only on training data."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ============================================================================
# 9. REGRESSION MODELS
# ============================================================================

class LinearRidgeRegressor:
    """
    Closed-form linear or ridge regression.

    alpha = 0 -> Linear Regression
    alpha > 0 -> Ridge Regression
    """

    def __init__(self, alpha: float = 0.0):
        self.alpha = float(alpha)
        self.coef_ = None

    @staticmethod
    def _design(X):
        return np.column_stack([np.ones(len(X)), X])

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xd = self._design(X)

        identity = np.eye(Xd.shape[1])
        identity[0, 0] = 0.0  # No penalty on intercept.

        self.coef_ = (
            np.linalg.pinv(
                Xd.T @ Xd + self.alpha * identity
            )
            @ Xd.T
            @ y
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._design(X) @ self.coef_


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Common regression metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae_value = np.mean(np.abs(y_true - y_pred))
    rmse_value = np.sqrt(np.mean((y_true - y_pred) ** 2))

    denominator = np.sum((y_true - np.mean(y_true)) ** 2)

    if denominator == 0:
        r2_value = float("nan")
    else:
        r2_value = 1 - (
            np.sum((y_true - y_pred) ** 2) / denominator
        )

    return {
        "MAE": float(mae_value),
        "RMSE": float(rmse_value),
        "R2": float(r2_value),
    }


# ============================================================================
# 10. CLASSIFICATION MODEL — KNN FROM SCRATCH
# ============================================================================

class KNNClassifier:
    """
    Very small educational K-Nearest Neighbours classifier.

    Prediction:
    1. Calculate Euclidean distance from a test row to all training rows.
    2. Select the k closest training examples.
    3. Return the most common class among those neighbours.
    """

    def __init__(self, k: int = 5):
        self.k = int(k)
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        return self

    def _predict_one(self, row: np.ndarray):
        distances = np.sqrt(
            np.sum(
                (self.X_train - row) ** 2,
                axis=1,
            )
        )

        nearest_indices = np.argsort(distances)[: self.k]
        nearest_labels = self.y_train[nearest_indices]

        values, counts = np.unique(
            nearest_labels,
            return_counts=True,
        )

        return values[np.argmax(counts)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array(
            [self._predict_one(row) for row in X]
        )


def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    """Accuracy plus macro precision/recall/F1 implemented manually."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = float(np.mean(y_true == y_pred))

    labels = np.unique(
        np.concatenate([y_true, y_pred])
    )

    precisions = []
    recalls = []
    f1s = []

    for label in labels:
        tp = np.sum(
            (y_true == label) & (y_pred == label)
        )
        fp = np.sum(
            (y_true != label) & (y_pred == label)
        )
        fn = np.sum(
            (y_true == label) & (y_pred != label)
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "Accuracy": accuracy,
        "Macro Precision": float(np.mean(precisions)),
        "Macro Recall": float(np.mean(recalls)),
        "Macro F1": float(np.mean(f1s)),
    }


def confusion_table(y_true, y_pred) -> pd.DataFrame:
    """Create a confusion matrix with pandas."""
    return pd.crosstab(
        pd.Series(y_true, name="Actual"),
        pd.Series(y_pred, name="Predicted"),
        margins=False,
    )


# ============================================================================
# 11. GENERIC MODEL PIPELINE
# ============================================================================

def build_and_evaluate_model(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    requested_task_type: str = "auto",
    test_fraction: float = 0.25,
    regression_model: str = "Ridge Regression",
    ridge_alpha: float = 5.0,
    knn_k: int = 5,
) -> Dict:
    """
    End-to-end machine-learning pipeline.

    Returns a dictionary so Streamlit mode and Colab mode can use the same
    model logic.
    """
    X, y = clean_features_and_target(
        df,
        feature_columns,
        target_column,
    )

    task = infer_task_type(
        y,
        requested=requested_task_type,
    )

    X_train, X_test, y_train, y_test = train_test_split_manual(
        X,
        y,
        test_fraction=test_fraction,
        random_state=42,
    )

    scaler = Standardizer()

    X_train_scaled = scaler.fit_transform(
        X_train.to_numpy(dtype=float)
    )

    X_test_scaled = scaler.transform(
        X_test.to_numpy(dtype=float)
    )

    if task == "regression":
        alpha = (
            0.0
            if regression_model == "Linear Regression"
            else ridge_alpha
        )

        model = LinearRidgeRegressor(alpha=alpha)

        y_train_numeric = pd.to_numeric(
            y_train,
            errors="coerce",
        ).to_numpy(dtype=float)

        y_test_numeric = pd.to_numeric(
            y_test,
            errors="coerce",
        ).to_numpy(dtype=float)

        # Regression target must be numeric.
        if np.isnan(y_train_numeric).any() or np.isnan(y_test_numeric).any():
            raise ValueError(
                "Regression target contains values that cannot be converted "
                "to numeric."
            )

        model.fit(
            X_train_scaled,
            y_train_numeric,
        )

        predictions = model.predict(
            X_test_scaled
        )

        metrics = regression_metrics(
            y_test_numeric,
            predictions,
        )

        coefficients = pd.DataFrame(
            {
                "feature": ["intercept"] + X.columns.tolist(),
                "coefficient": model.coef_,
            }
        )

        return {
            "task": task,
            "model": model,
            "scaler": scaler,
            "features_after_encoding": X.columns.tolist(),
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train_numeric,
            "y_test": y_test_numeric,
            "predictions": predictions,
            "metrics": metrics,
            "coefficients": coefficients,
        }

    # CLASSIFICATION
    y_train_labels = y_train.astype(str).to_numpy()
    y_test_labels = y_test.astype(str).to_numpy()

    k = max(1, min(int(knn_k), len(X_train)))

    model = KNNClassifier(k=k)
    model.fit(
        X_train_scaled,
        y_train_labels,
    )

    predictions = model.predict(
        X_test_scaled
    )

    metrics = classification_metrics(
        y_test_labels,
        predictions,
    )

    return {
        "task": task,
        "model": model,
        "scaler": scaler,
        "features_after_encoding": X.columns.tolist(),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train_labels,
        "y_test": y_test_labels,
        "predictions": predictions,
        "metrics": metrics,
        "confusion_matrix": confusion_table(
            y_test_labels,
            predictions,
        ),
    }


# ============================================================================
# 12. STREAMLIT APPLICATION
# ============================================================================

def run_streamlit_app():
    st.set_page_config(
        page_title=PROJECT_TITLE,
        page_icon="📊",
        layout="wide",
    )

    st.title(PROJECT_TITLE)
    st.caption(f"Student: {STUDENT_NAME}")

    # ------------------------------------------------------------------------
    # SIDEBAR
    # ------------------------------------------------------------------------

    with st.sidebar:
        st.header("Project Navigation")

        page = st.radio(
            "Section",
            [
                "1. Problem",
                "2. Data",
                "3. EDA",
                "4. Visualization",
                "5. Machine Learning",
                "6. Results & Reflection",
            ],
        )

        st.divider()

        uploaded_file = st.file_uploader(
            "Optional: upload your CSV",
            type=["csv"],
        )

    # ------------------------------------------------------------------------
    # DATA SELECTION
    # ------------------------------------------------------------------------

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        source_description = "User-uploaded CSV"
    else:
        df, source_description = load_default_data()

    if df.empty:
        st.error("The dataset is empty.")
        st.stop()

    # ------------------------------------------------------------------------
    # PAGE 1 — PROBLEM
    # ------------------------------------------------------------------------

    if page == "1. Problem":
        st.header("1. Problem Definition")

        st.subheader("Problem statement")
        st.info(PROBLEM_STATEMENT.strip())

        st.subheader("Research questions")
        for number, question in enumerate(
            RESEARCH_QUESTIONS,
            start=1,
        ):
            st.write(f"{number}. {question}")

        st.subheader("Project context")

        c1, c2 = st.columns(2)

        with c1:
            st.text_area(
                "Who are the expected users/stakeholders?",
                value=EXPECTED_USERS,
                height=120,
            )

        with c2:
            st.text_area(
                "What value could the project provide?",
                value=EXPECTED_VALUE,
                height=120,
            )

        st.subheader("Before touching the model, answer these")
        st.markdown(
            """
- What exactly is the unit of analysis?
- What outcome are you trying to understand or predict?
- Is prediction actually necessary?
- What data would ideally be available?
- Who could be affected by an incorrect prediction?
"""
        )

    # ------------------------------------------------------------------------
    # PAGE 2 — DATA
    # ------------------------------------------------------------------------

    elif page == "2. Data":
        st.header("2. Data")

        st.write(f"**Current source:** {source_description}")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Rows", len(df))
        with c2:
            st.metric("Columns", len(df.columns))
        with c3:
            st.metric(
                "Missing cells",
                int(df.isna().sum().sum()),
            )

        st.subheader("Data preview")
        st.dataframe(
            df.head(25),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Automatic data dictionary")
        st.dataframe(
            data_audit(df),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download current data as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="project_data.csv",
            mime="text/csv",
        )

        with st.expander("Student task: document your source"):
            st.markdown(
                """
Write down:
1. Dataset name.
2. Publisher / owner.
3. URL.
4. Date accessed.
5. Unit of analysis.
6. Time period.
7. Important variable definitions.
8. Licensing or reuse conditions.
"""
            )

    # ------------------------------------------------------------------------
    # PAGE 3 — EDA
    # ------------------------------------------------------------------------

    elif page == "3. EDA":
        st.header("3. Exploratory Data Analysis")

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Numeric summary",
                "Categorical summary",
                "Missing values",
                "Outliers",
            ]
        )

        with tab1:
            summary = numeric_summary(df)

            if summary.empty:
                st.info("No numeric columns found.")
            else:
                st.dataframe(
                    summary,
                    use_container_width=True,
                )

                st.subheader("Correlation")
                correlation = (
                    df.select_dtypes(include=np.number)
                    .corr()
                    .round(3)
                )

                st.dataframe(
                    correlation,
                    use_container_width=True,
                )

        with tab2:
            cat_summary = categorical_summary(df)

            if cat_summary.empty:
                st.info("No categorical columns found.")
            else:
                st.dataframe(
                    cat_summary,
                    use_container_width=True,
                    hide_index=True,
                )

                cat_col = st.selectbox(
                    "Inspect value counts",
                    cat_summary["column"].tolist(),
                )

                counts = (
                    df[cat_col]
                    .astype(str)
                    .value_counts(dropna=False)
                    .head(20)
                    .rename_axis(cat_col)
                    .reset_index(name="count")
                )

                st.bar_chart(
                    counts.set_index(cat_col),
                    use_container_width=True,
                )

        with tab3:
            missing = pd.DataFrame(
                {
                    "column": df.columns,
                    "missing": [
                        int(df[c].isna().sum())
                        for c in df.columns
                    ],
                    "missing_percent": [
                        round(df[c].isna().mean() * 100, 2)
                        for c in df.columns
                    ],
                }
            )

            st.dataframe(
                missing,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown(
                """
**Questions**
- Is missingness small or substantial?
- Is it concentrated in one variable?
- Could missingness itself be meaningful?
- Would dropping rows create bias?
"""
            )

        with tab4:
            numeric_cols = (
                df.select_dtypes(include=np.number)
                .columns
                .tolist()
            )

            if not numeric_cols:
                st.info("No numeric columns found.")
            else:
                outlier_rows = []

                for col in numeric_cols:
                    outlier_rows.append(
                        {
                            "column": col,
                            "IQR_outlier_count": iqr_outlier_count(
                                df[col]
                            ),
                        }
                    )

                st.dataframe(
                    pd.DataFrame(outlier_rows),
                    use_container_width=True,
                    hide_index=True,
                )

                st.warning(
                    "A statistical outlier is not automatically a data error. "
                    "Investigate before removing it."
                )

    # ------------------------------------------------------------------------
    # PAGE 4 — VISUALIZATION
    # ------------------------------------------------------------------------

    elif page == "4. Visualization":
        st.header("4. Visualization")

        numeric_cols = (
            df.select_dtypes(include=np.number)
            .columns
            .tolist()
        )

        if not numeric_cols:
            st.info("This dataset has no numeric columns to plot.")
            st.stop()

        chart_type = st.selectbox(
            "Chart type",
            [
                "Histogram",
                "Line",
                "Bar",
                "Scatter",
            ],
        )

        if chart_type == "Histogram":
            col = st.selectbox(
                "Numeric variable",
                numeric_cols,
            )

            values = pd.to_numeric(
                df[col],
                errors="coerce",
            ).dropna()

            bins = st.slider(
                "Number of bins",
                5,
                30,
                12,
            )

            counts, edges = np.histogram(
                values,
                bins=bins,
            )

            hist_df = pd.DataFrame(
                {
                    "bin_start": edges[:-1],
                    "count": counts,
                }
            ).set_index("bin_start")

            st.bar_chart(
                hist_df,
                use_container_width=True,
            )

        elif chart_type in {"Line", "Bar"}:
            x_col = st.selectbox(
                "X variable",
                df.columns.tolist(),
            )

            y_col = st.selectbox(
                "Y variable",
                numeric_cols,
            )

            plot_df = (
                df[[x_col, y_col]]
                .dropna()
                .copy()
            )

            # Limit categories so charts stay readable.
            plot_df = plot_df.head(200)

            try:
                chart_df = plot_df.set_index(x_col)[[y_col]]

                if chart_type == "Line":
                    st.line_chart(
                        chart_df,
                        use_container_width=True,
                    )
                else:
                    st.bar_chart(
                        chart_df,
                        use_container_width=True,
                    )

            except Exception as exc:
                st.error(
                    f"Could not create chart: {exc}"
                )

        else:
            if len(numeric_cols) < 2:
                st.info(
                    "A scatter chart needs at least two numeric columns."
                )
                st.stop()

            x_col = st.selectbox(
                "X numeric variable",
                numeric_cols,
                key="scatter_x",
            )

            y_options = [
                c for c in numeric_cols
                if c != x_col
            ]

            y_col = st.selectbox(
                "Y numeric variable",
                y_options,
                key="scatter_y",
            )

            scatter_df = (
                df[[x_col, y_col]]
                .dropna()
            )

            st.scatter_chart(
                scatter_df,
                x=x_col,
                y=y_col,
                use_container_width=True,
            )

            corr = scatter_df[x_col].corr(
                scatter_df[y_col]
            )

            st.metric(
                "Pearson correlation",
                f"{corr:.3f}",
            )

        st.subheader("Interpret the visualization")
        st.text_area(
            "Write 2–4 sentences: What pattern do you see? "
            "What might explain it? What can you NOT conclude?",
            height=140,
        )

    # ------------------------------------------------------------------------
    # PAGE 5 — MACHINE LEARNING
    # ------------------------------------------------------------------------

    elif page == "5. Machine Learning":
        st.header("5. Machine Learning")

        columns = df.columns.tolist()

        default_target = (
            TARGET_COLUMN
            if TARGET_COLUMN in columns
            else columns[-1]
        )

        target_column = st.selectbox(
            "Target column",
            columns,
            index=columns.index(default_target),
        )

        feature_options = [
            c for c in columns
            if c != target_column
        ]

        default_features = feature_options[: min(5, len(feature_options))]

        feature_columns = st.multiselect(
            "Feature columns",
            feature_options,
            default=default_features,
        )

        if not feature_columns:
            st.warning("Choose at least one feature.")
            st.stop()

        task_choice = st.selectbox(
            "Task type",
            ["auto", "regression", "classification"],
            index=(
                ["auto", "regression", "classification"]
                .index(TASK_TYPE)
                if TASK_TYPE in {
                    "auto",
                    "regression",
                    "classification",
                }
                else 0
            ),
        )

        test_fraction = st.slider(
            "Test-set fraction",
            0.20,
            0.40,
            0.25,
            0.05,
        )

        inferred = infer_task_type(
            df[target_column].dropna(),
            requested=task_choice,
        )

        st.write(f"**Task used:** {inferred}")

        regression_model = "Ridge Regression"
        ridge_alpha = 5.0
        knn_k = 5

        if inferred == "regression":
            regression_model = st.selectbox(
                "Regression model",
                [
                    "Linear Regression",
                    "Ridge Regression",
                ],
            )

            if regression_model == "Ridge Regression":
                ridge_alpha = st.slider(
                    "Ridge alpha",
                    0.0,
                    50.0,
                    5.0,
                    1.0,
                )

        else:
            knn_k = st.slider(
                "KNN: number of neighbours (k)",
                1,
                15,
                5,
                2,
            )

        if st.button(
            "Train and evaluate model",
            type="primary",
        ):
            try:
                result = build_and_evaluate_model(
                    df=df,
                    target_column=target_column,
                    feature_columns=feature_columns,
                    requested_task_type=task_choice,
                    test_fraction=test_fraction,
                    regression_model=regression_model,
                    ridge_alpha=ridge_alpha,
                    knn_k=knn_k,
                )

            except Exception as exc:
                st.exception(exc)
                st.stop()

            st.session_state["ml_result"] = result
            st.session_state["ml_target"] = target_column
            st.session_state["ml_features"] = feature_columns

        if "ml_result" in st.session_state:
            result = st.session_state["ml_result"]

            st.subheader("Evaluation metrics")

            metric_items = list(
                result["metrics"].items()
            )

            metric_cols = st.columns(
                len(metric_items)
            )

            for container, (name, value) in zip(
                metric_cols,
                metric_items,
            ):
                with container:
                    if isinstance(value, float):
                        if math.isnan(value):
                            st.metric(name, "N/A")
                        else:
                            st.metric(name, f"{value:.3f}")
                    else:
                        st.metric(name, str(value))

            st.subheader("Predictions")

            prediction_df = pd.DataFrame(
                {
                    "actual": result["y_test"],
                    "predicted": result["predictions"],
                }
            )

            st.dataframe(
                prediction_df,
                use_container_width=True,
                hide_index=True,
            )

            if result["task"] == "regression":
                st.subheader(
                    "Actual vs predicted chart"
                )

                st.line_chart(
                    prediction_df,
                    use_container_width=True,
                )

                st.subheader(
                    "Standardized feature coefficients"
                )

                st.dataframe(
                    result["coefficients"],
                    use_container_width=True,
                    hide_index=True,
                )

            else:
                st.subheader(
                    "Confusion matrix"
                )

                st.dataframe(
                    result["confusion_matrix"],
                    use_container_width=True,
                )

            st.subheader(
                "Encoded features used by the model"
            )

            st.write(
                result["features_after_encoding"]
            )

            st.warning(
                "Do not judge a model from one metric alone. "
                "Check sample size, class balance, leakage, data quality, "
                "generalizability and whether the model is useful for the "
                "original problem."
            )

    # ------------------------------------------------------------------------
    # PAGE 6 — RESULTS & REFLECTION
    # ------------------------------------------------------------------------

    else:
        st.header("6. Results & Reflection")

        st.subheader("Main findings")
        st.text_area(
            "Write your main findings here.",
            height=150,
        )

        st.subheader("Model interpretation")
        st.text_area(
            "Explain what the evaluation metrics mean in the context "
            "of your problem.",
            height=150,
        )

        st.subheader("Limitations")
        st.text_area(
            "Document data, method and interpretation limitations.",
            value=KNOWN_LIMITATIONS,
            height=150,
        )

        st.subheader("Responsible AI / ML questions")
        st.markdown(
            """
1. Could the target or features contain bias?
2. Could a model error disadvantage a person or group?
3. Is the dataset representative?
4. Are there privacy concerns?
5. Is the model explainable enough for the intended users?
6. What human oversight is needed?
7. What should the model explicitly NOT be used for?
"""
        )

        st.subheader("Final student checklist")

        checklist = pd.DataFrame(
            [
                ["Problem is specific and understandable", False],
                ["Data source is documented", False],
                ["Variables are explained", False],
                ["Missing values are inspected", False],
                ["At least 2 useful visualizations are interpreted", False],
                ["Target and features are justified", False],
                ["Train/test separation is used", False],
                ["Evaluation metrics are explained", False],
                ["Limitations are discussed", False],
                ["Ethical/responsible use is discussed", False],
            ],
            columns=["Requirement", "Complete?"],
        )

        st.data_editor(
            checklist,
            use_container_width=True,
            hide_index=True,
            disabled=["Requirement"],
        )

    st.divider()
    st.caption(
        "Single-file student project template • "
        "Problem → Data → EDA → Visualization → ML → Reflection"
    )


# ============================================================================
# 13. COLAB / NOTEBOOK MODE
# ============================================================================

def run_notebook_mode():
    """
    Run a compact end-to-end demonstration when this file is executed in Colab,
    Jupyter or a standard Python environment.

    Students can edit the configuration at the top and rerun the file.
    """
    print_section(PROJECT_TITLE)

    print(f"Student: {STUDENT_NAME}")
    print("\nProblem statement:")
    print(PROBLEM_STATEMENT.strip())

    print("\nResearch questions:")
    for i, question in enumerate(
        RESEARCH_QUESTIONS,
        start=1,
    ):
        print(f"{i}. {question}")

    # ------------------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------------------

    print_section("DATA")

    df, source_description = load_default_data()

    print(f"Source: {source_description}")
    print(f"Shape: {df.shape}")

    notebook_display(
        df.head()
    )

    # ------------------------------------------------------------------------
    # EDA
    # ------------------------------------------------------------------------

    print_section("DATA AUDIT")

    notebook_display(
        data_audit(df)
    )

    print_section("NUMERIC SUMMARY")

    notebook_display(
        numeric_summary(df)
    )

    print_section("CATEGORICAL SUMMARY")

    notebook_display(
        categorical_summary(df)
    )

    # ------------------------------------------------------------------------
    # BASIC VISUALIZATION IN COLAB
    # ------------------------------------------------------------------------

    print_section("VISUALIZATION")

    try:
        import matplotlib.pyplot as plt

        numeric_cols = (
            df.select_dtypes(include=np.number)
            .columns
            .tolist()
        )

        if numeric_cols:
            first_numeric = numeric_cols[0]

            df[first_numeric].dropna().plot(
                kind="hist",
                bins=12,
                title=f"Distribution of {first_numeric}",
            )

            plt.xlabel(first_numeric)
            plt.show()

    except Exception as exc:
        print(
            "Optional matplotlib visualization skipped:",
            exc,
        )

    # ------------------------------------------------------------------------
    # MACHINE LEARNING
    # ------------------------------------------------------------------------

    print_section("MACHINE LEARNING")

    if TARGET_COLUMN and TARGET_COLUMN in df.columns:
        target = TARGET_COLUMN
    else:
        # Teaching fallback:
        # use the last column as target.
        target = df.columns[-1]

        print(
            f"TARGET_COLUMN is not configured; using '{target}' "
            "for this demonstration."
        )

    features = [
        col for col in df.columns
        if col != target
    ]

    # Keep the first 5 features to make the demo readable.
    features = features[: min(5, len(features))]

    print("Target:", target)
    print("Features:", features)

    try:
        result = build_and_evaluate_model(
            df=df,
            target_column=target,
            feature_columns=features,
            requested_task_type=TASK_TYPE,
            test_fraction=0.25,
            regression_model="Ridge Regression",
            ridge_alpha=5.0,
            knn_k=5,
        )

        print("Task:", result["task"])
        print("Metrics:")

        for key, value in result["metrics"].items():
            print(f"  {key}: {value:.4f}")

        prediction_df = pd.DataFrame(
            {
                "actual": result["y_test"],
                "predicted": result["predictions"],
            }
        )

        notebook_display(
            prediction_df.head(20)
        )

        if result["task"] == "regression":
            print("\nCoefficients:")
            notebook_display(
                result["coefficients"]
            )

        else:
            print("\nConfusion matrix:")
            notebook_display(
                result["confusion_matrix"]
            )

    except Exception as exc:
        print("Model demonstration could not run:")
        print(exc)

    # ------------------------------------------------------------------------
    # STUDENT TODO
    # ------------------------------------------------------------------------

    print_section("STUDENT TODO")

    print(
        """
1. Edit PROJECT_TITLE and STUDENT_NAME.
2. Replace the problem statement.
3. Replace the research questions.
4. Add your public CSV URL OR adapt load_default_data().
5. Set TARGET_COLUMN.
6. Inspect missing values.
7. Produce and interpret at least two visualizations.
8. Justify your selected features.
9. Evaluate the model.
10. Explain limitations and responsible-use considerations.
"""
    )


# ============================================================================
# 14. PROGRAM ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    if STREAMLIT_MODE:
        run_streamlit_app()
    else:
        run_notebook_mode()
