"""
Bhutan Tourism Explorer: EDA + Visualization + Machine Learning
================================================================

TEACHING PURPOSE
----------------
This is a SINGLE-FILE Streamlit application designed for an introductory
Machine Learning / AI application-development lesson.

Students can learn how to:
1. Define a real-world problem.
2. Retrieve data from the internet.
3. Inspect and clean data.
4. Perform exploratory data analysis (EDA).
5. Create interactive visualizations.
6. Prepare features for machine learning.
7. Train and evaluate regression models.
8. Make a new prediction.
9. Interpret results and discuss limitations.

DEPLOYMENT
----------
Put this file in a GitHub repository and select it as the Streamlit app entry file.

Typical command for local use:
    streamlit run bhutan_tourism_ml_streamlit.py

WHY THIS FILE USES ONLY COMMON PACKAGES
---------------------------------------
To keep deployment simple, the core project relies only on:
    streamlit
    pandas
    numpy

The regression algorithms and evaluation metrics are implemented in this file
with NumPy. This makes the code easier to teach and reduces dependency problems.

DATA
----
Primary live source:
World Bank API, Bhutan (country code BTN)

Indicators:
- ST.INT.ARVL      International tourism, number of arrivals
- ST.INT.RCPT.CD   International tourism, receipts (current US$)
- NY.GDP.MKTP.CD   GDP (current US$)
- SP.POP.TOTL      Population, total
- FP.CPI.TOTL.ZG   Inflation, consumer prices (annual %)

World Bank tourism indicator metadata:
https://data.worldbank.org/indicator/ST.INT.ARVL

IMPORTANT:
Tourism data availability varies by year and indicator. The app therefore
contains a clearly labelled deterministic DEMO fallback dataset so a class can
continue even when the API or classroom internet is unavailable.

This is an educational application, not an official tourism forecasting system.
"""

# ============================================================================
# 1. IMPORTS
# ============================================================================

import json
import math
import urllib.request
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Bhutan Tourism ML Explorer",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# 3. PROJECT CONFIGURATION
# ============================================================================

PROJECT_TITLE = "Bhutan Tourism Explorer"
PROJECT_SUBTITLE = "From Internet Data to EDA, Visualization and Machine Learning"

WORLD_BANK_COUNTRY = "BTN"

INDICATORS: Dict[str, str] = {
    "tourist_arrivals": "ST.INT.ARVL",
    "tourism_receipts_usd": "ST.INT.RCPT.CD",
    "gdp_usd": "NY.GDP.MKTP.CD",
    "population": "SP.POP.TOTL",
    "inflation_percent": "FP.CPI.TOTL.ZG",
}

FRIENDLY_NAMES = {
    "year": "Year",
    "tourist_arrivals": "International tourist arrivals",
    "tourism_receipts_usd": "Tourism receipts (USD)",
    "gdp_usd": "GDP (USD)",
    "population": "Population",
    "inflation_percent": "Inflation (%)",
}

SOURCE_URL = "https://data.worldbank.org/indicator/ST.INT.ARVL"


# ============================================================================
# 4. SMALL HELPER FUNCTIONS
# ============================================================================

def format_number(value) -> str:
    """Convert a numeric value into a compact human-readable string."""
    if pd.isna(value):
        return "N/A"
    value = float(value)
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.2f}"


def safe_float(value, default=0.0) -> float:
    """Safely convert a value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


# ============================================================================
# 5. INTERNET DATA RETRIEVAL
# ============================================================================

def fetch_world_bank_indicator(
    indicator_code: str,
    country_code: str = WORLD_BANK_COUNTRY,
    start_year: int = 1995,
    end_year: int = 2025,
) -> pd.DataFrame:
    """
    Fetch one World Bank indicator for Bhutan.

    World Bank API pattern:
    https://api.worldbank.org/v2/country/BTN/indicator/INDICATOR
        ?format=json&per_page=100

    The API usually returns:
        [metadata, observations]

    We convert observations into:
        year | value
    """
    url = (
        f"https://api.worldbank.org/v2/country/{country_code}/indicator/"
        f"{indicator_code}?format=json&per_page=100"
    )

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 StreamlitTeachingApp"},
    )

    with urllib.request.urlopen(request, timeout=12) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        raise ValueError(f"No observations returned for {indicator_code}")

    rows = []
    for obs in payload[1]:
        try:
            year = int(obs.get("date"))
        except (TypeError, ValueError):
            continue

        if start_year <= year <= end_year:
            rows.append(
                {
                    "year": year,
                    "value": obs.get("value"),
                }
            )

    return pd.DataFrame(rows)


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_world_bank_data() -> Tuple[pd.DataFrame, str]:
    """
    Download several indicators and merge them by year.

    Returns
    -------
    dataframe
        Merged dataset.
    source_status
        Text explaining whether data came from the live API or fallback.
    """
    merged = None

    for column_name, indicator_code in INDICATORS.items():
        indicator_df = fetch_world_bank_indicator(indicator_code)
        indicator_df = indicator_df.rename(columns={"value": column_name})

        if merged is None:
            merged = indicator_df
        else:
            merged = pd.merge(
                merged,
                indicator_df,
                on="year",
                how="outer",
            )

    if merged is None or merged.empty:
        raise ValueError("World Bank API returned no usable data.")

    merged = merged.sort_values("year").reset_index(drop=True)

    # Convert all non-year columns to numeric.
    for col in merged.columns:
        if col != "year":
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    return merged, "LIVE World Bank API data"


# ============================================================================
# 6. OFFLINE DEMO FALLBACK
# ============================================================================

def create_demo_bhutan_dataset() -> pd.DataFrame:
    """
    Create a deterministic teaching dataset.

    IMPORTANT:
    This fallback is SYNTHETIC. It is not presented as official historical data.
    It exists only so that teaching can continue if the live API is unavailable.

    A fixed random seed makes the same demo data appear every time.
    """
    rng = np.random.default_rng(42)
    years = np.arange(1995, 2021)
    n = len(years)

    population = 550_000 + (years - 1995) * 8_200 + rng.normal(0, 7_000, n)
    gdp = 350_000_000 * np.exp(0.085 * (years - 1995)) * rng.normal(1.0, 0.08, n)

    # Educational trend: gradual tourism growth followed by a 2020 shock.
    arrivals = (
        6_000
        + (years - 1995) ** 1.75 * 1_100
        + rng.normal(0, 8_000, n)
    )
    arrivals = np.maximum(arrivals, 2_000)
    arrivals[years == 2020] = 25_000

    receipts = arrivals * rng.normal(2_800, 350, n)
    inflation = rng.normal(5.2, 2.0, n)

    return pd.DataFrame(
        {
            "year": years,
            "tourist_arrivals": arrivals.round(0),
            "tourism_receipts_usd": receipts.round(0),
            "gdp_usd": gdp.round(0),
            "population": population.round(0),
            "inflation_percent": inflation.round(2),
        }
    )


def get_data() -> Tuple[pd.DataFrame, str]:
    """
    Attempt live download first.
    Use the clearly labelled synthetic fallback only if needed.
    """
    try:
        data, status = load_world_bank_data()

        # We want enough data for an educational train/test exercise.
        if data["tourist_arrivals"].notna().sum() < 8:
            raise ValueError("Too few tourism-arrival observations for the demo.")

        return data, status

    except Exception as exc:
        demo = create_demo_bhutan_dataset()
        return demo, f"SYNTHETIC OFFLINE DEMO data — live API unavailable ({exc})"


# ============================================================================
# 7. DATA CLEANING HELPERS
# ============================================================================

def missing_value_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing counts and percentages for each column."""
    result = pd.DataFrame(
        {
            "missing_values": df.isna().sum(),
            "missing_percent": (df.isna().mean() * 100).round(2),
            "data_type": df.dtypes.astype(str),
        }
    )
    result.index.name = "column"
    return result.reset_index()


def iqr_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Detect potential outliers with the 1.5 * IQR rule.

    This does NOT mean every flagged row is an error.
    In tourism, a crisis year can be a real and analytically important outlier.
    """
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if len(series) < 4:
        return pd.DataFrame()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return df[(df[column] < lower) | (df[column] > upper)].copy()


# ============================================================================
# 8. MACHINE LEARNING IMPLEMENTATION
# ============================================================================

def chronological_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_fraction: float = 0.25,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Chronological split for time-oriented data.

    Earlier observations -> training
    Later observations   -> testing

    This is often more realistic than randomly mixing past and future.
    """
    split_index = max(1, int(len(X) * (1 - test_fraction)))
    split_index = min(split_index, len(X) - 1)

    return (
        X.iloc[:split_index].copy(),
        X.iloc[split_index:].copy(),
        y.iloc[:split_index].copy(),
        y.iloc[split_index:].copy(),
    )


def random_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_fraction: float = 0.25,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Simple reproducible random split implemented with NumPy."""
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


class Standardizer:
    """Minimal feature standardizer: z = (x - mean) / std."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class NumpyRegressor:
    """
    Educational linear/ridge regression using the closed-form solution.

    Linear regression:
        beta = pinv(X'X) X'y

    Ridge regression:
        beta = pinv(X'X + lambda*I) X'y

    The intercept is not regularized.
    """

    def __init__(self, ridge_alpha: float = 0.0):
        self.ridge_alpha = float(ridge_alpha)
        self.coef_ = None

    @staticmethod
    def _add_intercept(X: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(X)), X])

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_design = self._add_intercept(X)
        identity = np.eye(X_design.shape[1])
        identity[0, 0] = 0.0  # Do not penalize the intercept.

        penalty = self.ridge_alpha * identity

        self.coef_ = (
            np.linalg.pinv(X_design.T @ X_design + penalty)
            @ X_design.T
            @ y
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_design = self._add_intercept(X)
        return X_design @ self.coef_


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(
        np.sqrt(
            np.mean(
                (np.asarray(y_true) - np.asarray(y_pred)) ** 2
            )
        )
    )


def r2_score_manual(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    if denominator == 0:
        return float("nan")

    numerator = np.sum((y_true - y_pred) ** 2)
    return float(1 - numerator / denominator)


def prepare_ml_dataset(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
) -> pd.DataFrame:
    """
    Keep only selected columns and remove rows with missing values.

    For a first ML lesson, complete-case analysis keeps the workflow transparent.
    In later lessons, students can replace this with proper imputation.
    """
    cols = feature_columns + [target_column]
    work = df[cols].copy()

    for col in cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna().reset_index(drop=True)
    return work


# ============================================================================
# 9. AUTOMATED INTERPRETATION
# ============================================================================

def generate_model_interpretation(
    metrics: Dict[str, float],
    test_rows: int,
    feature_names: List[str],
) -> str:
    """
    Produce a simple automated explanation.

    This is NOT a generative AI model. It is rule-based interpretation,
    included to discuss the difference between analytics automation and LLMs.
    """
    r2 = metrics.get("R2", float("nan"))
    rmse_value = metrics.get("RMSE", float("nan"))

    if math.isnan(r2):
        fit_text = "R² could not be interpreted for this test set."
    elif r2 >= 0.80:
        fit_text = "The model explains a large share of variation in the held-out data."
    elif r2 >= 0.50:
        fit_text = "The model explains a moderate share of variation in the held-out data."
    elif r2 >= 0.0:
        fit_text = "The model has limited explanatory performance on the held-out data."
    else:
        fit_text = (
            "The model performs worse than predicting the test-set mean, "
            "so it should not be treated as a reliable forecasting model."
        )

    feature_text = ", ".join(feature_names)

    return (
        f"{fit_text} The evaluation used {test_rows} test observations. "
        f"The selected predictors were: {feature_text}. "
        f"RMSE is {format_number(rmse_value)} arrivals. "
        "Because Bhutan tourism data contain relatively few annual observations "
        "and may include structural breaks, the result should be used as a "
        "teaching demonstration rather than an operational forecast."
    )


# ============================================================================
# 10. LOAD DATA
# ============================================================================

df, source_status = get_data()

# Sort once because this project is time-oriented.
df = df.sort_values("year").reset_index(drop=True)

numeric_columns = [
    col for col in df.columns
    if pd.api.types.is_numeric_dtype(df[col])
]


# ============================================================================
# 11. SIDEBAR NAVIGATION
# ============================================================================

with st.sidebar:
    st.title("🏔️ Bhutan Tourism ML")
    st.caption("Single-file teaching application")

    page = st.radio(
        "Choose lesson section",
        [
            "1. Project & Data",
            "2. EDA",
            "3. Visualization",
            "4. Machine Learning",
            "5. Prediction Lab",
            "6. Teaching Guide",
        ],
    )

    st.divider()
    st.write("**Data status**")
    st.caption(source_status)

    st.write("**Primary source**")
    st.markdown(f"[World Bank indicator page]({SOURCE_URL})")

    if "SYNTHETIC" in source_status:
        st.warning(
            "You are viewing the offline synthetic teaching fallback, "
            "not official historical observations."
        )


# ============================================================================
# 12. HEADER
# ============================================================================

st.title(PROJECT_TITLE)
st.subheader(PROJECT_SUBTITLE)
st.caption(
    "A complete introductory workflow: problem → data → EDA → visualization "
    "→ model → evaluation → prediction → reflection"
)


# ============================================================================
# PAGE 1: PROJECT & DATA
# ============================================================================

if page == "1. Project & Data":
    st.header("1. Define the problem")

    st.markdown(
        """
**Teaching problem statement**

> How can historical economic and tourism indicators help us understand and
> predict international tourist arrivals to Bhutan?

This project is deliberately framed as an **educational regression problem**.
The goal is not to claim that a simple model can fully forecast tourism.
Instead, students learn the complete data-science workflow and then critique
the assumptions.
"""
    )

    st.subheader("Research / analytics questions")
    st.markdown(
        """
1. How have recorded international tourist arrivals changed over time?
2. How are arrivals associated with tourism receipts, GDP and population?
3. Which years appear unusual?
4. Can a simple regression model predict held-out annual arrival values?
5. What are the limitations of using a small annual dataset for forecasting?
"""
    )

    st.subheader("Data retrieved from the internet")

    c1, c2, c3, c4 = st.columns(4)

    latest_arrival_row = (
        df.dropna(subset=["tourist_arrivals"])
        .sort_values("year")
        .tail(1)
    )

    with c1:
        st.metric("Rows", f"{len(df):,}")
    with c2:
        st.metric("Columns", f"{len(df.columns):,}")
    with c3:
        if not latest_arrival_row.empty:
            latest_year = int(latest_arrival_row["year"].iloc[0])
            st.metric("Latest arrival year in data", latest_year)
        else:
            st.metric("Latest arrival year in data", "N/A")
    with c4:
        missing_total = int(df.isna().sum().sum())
        st.metric("Missing cells", f"{missing_total:,}")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("What should students notice?"):
        st.markdown(
            """
- Real datasets often contain **missing values**.
- Different indicators may stop in different years.
- A dataset can be useful even when it is not perfectly complete.
- Before training a model, we must decide which rows and columns are suitable.
- A source should always be documented.
"""
        )

    st.subheader("Download the current working dataset")
    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="bhutan_tourism_world_bank_or_demo.csv",
        mime="text/csv",
    )


# ============================================================================
# PAGE 2: EDA
# ============================================================================

elif page == "2. EDA":
    st.header("2. Exploratory Data Analysis (EDA)")
    st.write(
        "EDA helps us understand data quality, distributions, relationships, "
        "missingness and unusual observations before we model anything."
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Preview", "Missing values", "Statistics", "Outliers"]
    )

    with tab1:
        rows_to_show = st.slider(
            "Rows to preview",
            min_value=5,
            max_value=min(30, len(df)),
            value=min(10, len(df)),
        )
        st.dataframe(
            df.head(rows_to_show),
            use_container_width=True,
            hide_index=True,
        )

        st.write("**Column data types**")
        dtype_df = pd.DataFrame(
            {
                "column": df.columns,
                "dtype": [str(x) for x in df.dtypes],
            }
        )
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    with tab2:
        missing_df = missing_value_table(df)
        st.dataframe(
            missing_df,
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "Teaching point: missing values are not automatically 'bad'. "
            "We first investigate why they are missing and how much data would "
            "be lost by dropping incomplete rows."
        )

    with tab3:
        summary = df[numeric_columns].describe().T
        summary["missing"] = df[numeric_columns].isna().sum()
        st.dataframe(
            summary.round(2),
            use_container_width=True,
        )

        st.subheader("Correlation table")
        corr = df[numeric_columns].corr(numeric_only=True).round(3)
        st.dataframe(
            corr,
            use_container_width=True,
        )

        st.caption(
            "Correlation describes association, not causation. "
            "For time series, common trends can create strong correlations."
        )

    with tab4:
        outlier_col = st.selectbox(
            "Choose a numeric column",
            numeric_columns,
            index=(
                numeric_columns.index("tourist_arrivals")
                if "tourist_arrivals" in numeric_columns
                else 0
            ),
        )

        outliers = iqr_outliers(df, outlier_col)

        if outliers.empty:
            st.success("No potential IQR outliers were detected.")
        else:
            st.write(
                f"Potential outliers detected in **{FRIENDLY_NAMES.get(outlier_col, outlier_col)}**:"
            )
            st.dataframe(
                outliers,
                use_container_width=True,
                hide_index=True,
            )

        st.warning(
            "Do not remove an outlier only because a formula flags it. "
            "A tourism crisis year may be an important real event."
        )


# ============================================================================
# PAGE 3: VISUALIZATION
# ============================================================================

elif page == "3. Visualization":
    st.header("3. Interactive Visualization")

    viz_type = st.selectbox(
        "Visualization type",
        ["Line chart", "Bar chart", "Area chart", "Scatter chart"],
    )

    y_column = st.selectbox(
        "Variable to visualize",
        numeric_columns,
        index=(
            numeric_columns.index("tourist_arrivals")
            if "tourist_arrivals" in numeric_columns
            else 0
        ),
        format_func=lambda c: FRIENDLY_NAMES.get(c, c),
    )

    plot_df = df[["year", y_column]].dropna().copy()

    normalize = st.checkbox(
        "Normalize to an index where first available year = 100",
        value=False,
    )

    display_column = y_column

    if normalize and not plot_df.empty:
        first_value = plot_df[y_column].iloc[0]
        if first_value != 0:
            display_column = f"{y_column}_index"
            plot_df[display_column] = (
                plot_df[y_column] / first_value * 100
            )

    st.subheader(
        f"{FRIENDLY_NAMES.get(y_column, y_column)} over time"
    )

    chart_data = plot_df.set_index("year")[[display_column]]

    if viz_type == "Line chart":
        st.line_chart(chart_data, use_container_width=True)

    elif viz_type == "Bar chart":
        st.bar_chart(chart_data, use_container_width=True)

    elif viz_type == "Area chart":
        st.area_chart(chart_data, use_container_width=True)

    else:
        scatter_df = plot_df.rename(
            columns={
                "year": "Year",
                display_column: FRIENDLY_NAMES.get(y_column, y_column),
            }
        )
        st.scatter_chart(
            scatter_df,
            x="Year",
            y=FRIENDLY_NAMES.get(y_column, y_column),
            use_container_width=True,
        )

    st.divider()
    st.subheader("Relationship explorer")

    candidate_xy = [
        col for col in numeric_columns
        if col != "year"
    ]

    x_var = st.selectbox(
        "X variable",
        candidate_xy,
        index=0,
        format_func=lambda c: FRIENDLY_NAMES.get(c, c),
        key="x_var",
    )

    y_var_options = [c for c in candidate_xy if c != x_var]

    if y_var_options:
        default_y = (
            "tourist_arrivals"
            if "tourist_arrivals" in y_var_options
            else y_var_options[0]
        )

        y_var = st.selectbox(
            "Y variable",
            y_var_options,
            index=y_var_options.index(default_y),
            format_func=lambda c: FRIENDLY_NAMES.get(c, c),
            key="y_var",
        )

        relation_df = df[[x_var, y_var]].dropna()

        st.scatter_chart(
            relation_df,
            x=x_var,
            y=y_var,
            use_container_width=True,
        )

        corr_value = relation_df[x_var].corr(relation_df[y_var])
        st.metric("Pearson correlation", f"{corr_value:.3f}")

        st.caption(
            "Ask students: Does this pattern imply causation? "
            "Could both variables be increasing because of time?"
        )


# ============================================================================
# PAGE 4: MACHINE LEARNING
# ============================================================================

elif page == "4. Machine Learning":
    st.header("4. Machine Learning: Predict Tourist Arrivals")

    st.markdown(
        """
We will train a **regression model** because the target
`tourist_arrivals` is numeric.

The application compares:
- **Linear Regression** — no regularization.
- **Ridge Regression** — shrinks coefficients to reduce instability.

For a time-oriented project, a chronological split is often more realistic:
train on earlier years and test on later years.
"""
    )

    target_column = "tourist_arrivals"

    available_features = [
        c for c in numeric_columns
        if c != target_column
    ]

    preferred_features = [
        c for c in [
            "year",
            "tourism_receipts_usd",
            "gdp_usd",
            "population",
        ]
        if c in available_features
    ]

    feature_columns = st.multiselect(
        "Choose predictor features",
        options=available_features,
        default=preferred_features,
        format_func=lambda c: FRIENDLY_NAMES.get(c, c),
    )

    if not feature_columns:
        st.error("Select at least one feature to continue.")
        st.stop()

    split_method = st.radio(
        "Train/test strategy",
        ["Chronological split", "Random split"],
        horizontal=True,
    )

    test_fraction = st.slider(
        "Test-set fraction",
        min_value=0.20,
        max_value=0.40,
        value=0.25,
        step=0.05,
    )

    model_choice = st.selectbox(
        "Regression model",
        ["Linear Regression", "Ridge Regression"],
    )

    ridge_alpha = 0.0
    if model_choice == "Ridge Regression":
        ridge_alpha = st.slider(
            "Ridge penalty (alpha)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=1.0,
        )

    ml_df = prepare_ml_dataset(
        df,
        feature_columns=feature_columns,
        target_column=target_column,
    )

    st.write(
        f"**Complete rows available for this model:** {len(ml_df)}"
    )

    if len(ml_df) < 8:
        st.error(
            "Too few complete observations are available for a meaningful "
            "teaching split. Try fewer features."
        )
        st.stop()

    X = ml_df[feature_columns]
    y = ml_df[target_column]

    if split_method == "Chronological split":
        # Ensure chronological order whenever year exists.
        if "year" in ml_df.columns:
            ml_df = ml_df.sort_values("year").reset_index(drop=True)
            X = ml_df[feature_columns]
            y = ml_df[target_column]

        X_train, X_test, y_train, y_test = chronological_train_test_split(
            X, y, test_fraction
        )
    else:
        X_train, X_test, y_train, y_test = random_train_test_split(
            X, y, test_fraction, random_state=42
        )

    # Standardize based ONLY on the training set.
    scaler = Standardizer()
    X_train_scaled = scaler.fit_transform(X_train.to_numpy(dtype=float))
    X_test_scaled = scaler.transform(X_test.to_numpy(dtype=float))

    model = NumpyRegressor(ridge_alpha=ridge_alpha)
    model.fit(
        X_train_scaled,
        y_train.to_numpy(dtype=float),
    )

    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    metrics = {
        "MAE": mae(y_test, test_pred),
        "RMSE": rmse(y_test, test_pred),
        "R2": r2_score_manual(y_test, test_pred),
    }

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Training rows", len(X_train))
    with c2:
        st.metric("Test rows", len(X_test))
    with c3:
        st.metric("Test MAE", format_number(metrics["MAE"]))
    with c4:
        r2_text = (
            "N/A"
            if math.isnan(metrics["R2"])
            else f"{metrics['R2']:.3f}"
        )
        st.metric("Test R²", r2_text)

    st.subheader("Actual vs. predicted")

    prediction_table = X_test.copy()
    prediction_table["actual_arrivals"] = y_test.values
    prediction_table["predicted_arrivals"] = np.maximum(test_pred, 0)
    prediction_table["absolute_error"] = np.abs(
        prediction_table["actual_arrivals"]
        - prediction_table["predicted_arrivals"]
    )

    st.dataframe(
        prediction_table.round(2),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Prediction comparison chart")
    chart_df = pd.DataFrame(
        {
            "Actual": y_test.to_numpy(dtype=float),
            "Predicted": np.maximum(test_pred, 0),
        }
    )

    if "year" in X_test.columns:
        chart_df.index = X_test["year"].astype(int).values
        chart_df.index.name = "Year"

    st.line_chart(chart_df, use_container_width=True)

    st.subheader("Model coefficients")

    coef_rows = [{"term": "intercept", "coefficient": model.coef_[0]}]
    for feature, coef in zip(feature_columns, model.coef_[1:]):
        coef_rows.append(
            {
                "term": feature,
                "coefficient": coef,
            }
        )

    st.dataframe(
        pd.DataFrame(coef_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Because predictors were standardized, feature coefficients are more "
        "comparable than coefficients fitted on their original scales."
    )

    st.subheader("Automated interpretation")
    st.info(
        generate_model_interpretation(
            metrics=metrics,
            test_rows=len(X_test),
            feature_names=feature_columns,
        )
    )

    with st.expander("Important ML discussion questions"):
        st.markdown(
            """
1. Is the dataset large enough for a reliable forecasting model?
2. What happens if we include tourism receipts as a predictor of arrivals?
3. Could that create information leakage or circularity?
4. Why might chronological splitting be preferable to random splitting?
5. What happened to tourism during extraordinary crisis periods?
6. Can a strong correlation still be misleading?
7. What additional variables would improve a real tourism model?
8. Should policy decisions be made from this model alone? Why not?
"""
        )


# ============================================================================
# PAGE 5: PREDICTION LAB
# ============================================================================

elif page == "5. Prediction Lab":
    st.header("5. Prediction Lab")

    st.write(
        "Train a model using complete historical rows, then enter a new scenario."
    )

    target_column = "tourist_arrivals"

    candidate_features = [
        c for c in ["year", "tourism_receipts_usd", "gdp_usd", "population"]
        if c in df.columns
    ]

    prediction_features = st.multiselect(
        "Features for the prediction model",
        options=[
            c for c in numeric_columns
            if c != target_column
        ],
        default=candidate_features,
        format_func=lambda c: FRIENDLY_NAMES.get(c, c),
        key="prediction_features",
    )

    if not prediction_features:
        st.warning("Choose at least one feature.")
        st.stop()

    model_df = prepare_ml_dataset(
        df,
        prediction_features,
        target_column,
    )

    if len(model_df) < 8:
        st.error("Not enough complete observations. Select fewer features.")
        st.stop()

    X_all = model_df[prediction_features].to_numpy(dtype=float)
    y_all = model_df[target_column].to_numpy(dtype=float)

    scaler = Standardizer()
    X_scaled = scaler.fit_transform(X_all)

    final_model = NumpyRegressor(ridge_alpha=5.0)
    final_model.fit(X_scaled, y_all)

    st.subheader("Enter a scenario")

    input_values = []

    for feature in prediction_features:
        series = pd.to_numeric(model_df[feature], errors="coerce").dropna()

        min_value = float(series.min())
        max_value = float(series.max())
        median_value = float(series.median())

        if feature == "year":
            default_value = int(max_value + 1)
            value = st.number_input(
                FRIENDLY_NAMES.get(feature, feature),
                min_value=int(min_value),
                max_value=int(max_value + 10),
                value=default_value,
                step=1,
            )
        else:
            value = st.number_input(
                FRIENDLY_NAMES.get(feature, feature),
                value=median_value,
                format="%.2f",
            )

        input_values.append(float(value))

    scenario = np.array([input_values], dtype=float)
    scenario_scaled = scaler.transform(scenario)
    predicted_arrivals = float(
        np.maximum(final_model.predict(scenario_scaled)[0], 0)
    )

    st.success(
        f"Predicted tourist arrivals for this scenario: "
        f"**{predicted_arrivals:,.0f}**"
    )

    st.warning(
        "This prediction is an educational model output, not an official "
        "tourism forecast. Predictions beyond the historical feature ranges "
        "are extrapolations and can be highly unreliable."
    )


# ============================================================================
# PAGE 6: TEACHING GUIDE
# ============================================================================

else:
    st.header("6. Teaching Guide")

    st.markdown(
        """
### Suggested 75–90 minute lesson

**Part A — Problem framing (10 min)**  
Ask students to identify the target, predictors, unit of analysis and users of
the model. Discuss the difference between describing tourism and forecasting it.

**Part B — Data acquisition (10 min)**  
Show the World Bank API function. Explain API, JSON, indicator codes and why
source documentation matters.

**Part C — EDA (15 min)**  
Inspect dimensions, missing values, summary statistics and outliers. Ask:
"Would you immediately delete the 2020 observation?"

**Part D — Visualization (10 min)**  
Use line charts and scatter plots. Discuss trend, association and causal claims.

**Part E — Machine learning (20 min)**  
Select features, create train/test data, standardize predictors, fit the model
and evaluate MAE, RMSE and R².

**Part F — Critical reflection (10–20 min)**  
Discuss data size, leakage, structural breaks, ethical use, uncertainty and what
additional information a production system would need.
"""
    )

    st.subheader("Concept map")

    concept_df = pd.DataFrame(
        [
            ["Problem", "What decision or question are we addressing?"],
            ["Data", "What observations and variables do we have?"],
            ["EDA", "What patterns, missing values and anomalies exist?"],
            ["Visualization", "How can we communicate those patterns?"],
            ["Features", "What information is available to the model?"],
            ["Target", "What are we trying to predict?"],
            ["Training", "What data does the model learn from?"],
            ["Testing", "What unseen data do we evaluate on?"],
            ["Metric", "How do we measure prediction error?"],
            ["Interpretation", "What can and cannot be concluded?"],
            ["Deployment", "How does a user interact with the model?"],
        ],
        columns=["Stage", "Key question"],
    )

    st.dataframe(
        concept_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Mini student tasks")

    st.markdown(
        """
1. Change the feature set and record how R² changes.
2. Compare chronological and random splitting.
3. Remove `tourism_receipts_usd` and discuss whether this reduces leakage risk.
4. Identify the year with the largest prediction error.
5. Add one new World Bank indicator to the `INDICATORS` dictionary.
6. Rewrite the problem statement for a policy-maker rather than a data scientist.
7. Explain why this simple model should not be used as an official forecast.
"""
    )

    st.subheader("Extension ideas")

    st.markdown(
        """
- Add monthly arrivals from a suitable official source.
- Compare multiple countries.
- Add seasonality when monthly data are available.
- Add classification: "high-growth year" vs "low-growth year".
- Add clustering to identify similar tourism periods.
- Add a real LLM only after discussing privacy, cost, hallucination and
  reproducibility.
- Create a model card documenting intended use and limitations.
"""
    )

st.divider()
st.caption(
    "Educational single-file Streamlit project • Bhutan Tourism • "
    "EDA + Visualization + Machine Learning"
)
