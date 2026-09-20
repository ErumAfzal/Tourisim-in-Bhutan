"""
Bhutan Student Wellbeing — Basic ML Streamlit App

GitHub setup:
1. Save this file as app.py.
2. Create requirements.txt containing:
   streamlit
   pandas
   numpy
   scikit-learn
   matplotlib
   seaborn
3. Run locally with: streamlit run app.py

Important: This classroom project uses synthetic data only. It is not a
medical, psychological, screening, or diagnostic tool.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(
    page_title="Bhutan Student Wellbeing",
    page_icon="🌿",
    layout="wide",
)

TARGET = "support_need"
POSITIVE = "Support recommended"


@st.cache_data
def generate_synthetic_data(n_rows=800, seed=42):
    """Create fictional records for teaching; no real people are represented."""
    rng = np.random.default_rng(seed)
    districts = [
        "Thimphu",
        "Paro",
        "Punakha",
        "Wangdue Phodrang",
        "Sarpang",
        "Trashigang",
        "Mongar",
        "Samtse",
        "Chukha",
        "Bumthang",
    ]

    gender = rng.choice(
        ["Woman", "Man", "Non-binary/Prefer not to say"],
        n_rows,
        p=[0.49, 0.48, 0.03],
    )
    age = rng.integers(18, 27, n_rows)
    district = rng.choice(districts, n_rows)
    residence = rng.choice(["Urban", "Rural"], n_rows, p=[0.58, 0.42])
    sleep_hours = np.clip(rng.normal(6.8, 1.25, n_rows), 3.5, 10).round(1)
    study_hours = np.clip(rng.normal(5.7, 2.0, n_rows), 1, 12).round(1)
    physical_days = np.clip(rng.poisson(2.7, n_rows), 0, 7)
    stress = np.clip(np.rint(rng.normal(5.6, 2.0, n_rows)), 1, 10).astype(int)
    social_support = np.clip(
        np.rint(rng.normal(6.4, 2.0, n_rows)), 1, 10
    ).astype(int)
    financial_pressure = np.clip(
        np.rint(rng.normal(5.0, 2.3, n_rows)), 1, 10
    ).astype(int)
    academic_pressure = np.clip(
        np.rint(rng.normal(6.1, 2.0, n_rows)), 1, 10
    ).astype(int)
    screen_hours = np.clip(rng.normal(5.3, 2.0, n_rows), 0.5, 12).round(1)
    awareness = rng.choice(
        ["Low", "Medium", "High"], n_rows, p=[0.25, 0.49, 0.26]
    )

    awareness_score = (
        pd.Series(awareness)
        .map({"Low": 0.7, "Medium": 0.1, "High": -0.4})
        .to_numpy()
    )

    # Transparent fictional rule used only to create the teaching label.
    latent_score = (
        0.55 * stress
        + 0.38 * academic_pressure
        + 0.30 * financial_pressure
        + 0.18 * screen_hours
        - 0.55 * social_support
        - 0.42 * sleep_hours
        - 0.22 * physical_days
        + awareness_score
        + rng.normal(0, 1.5, n_rows)
    )
    threshold = np.quantile(latent_score, 0.55)
    support_need = np.where(
        latent_score >= threshold,
        POSITIVE,
        "No immediate flag",
    )

    return pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "district": district,
            "residence": residence,
            "sleep_hours": sleep_hours,
            "study_hours_per_day": study_hours,
            "physical_activity_days": physical_days,
            "stress_level": stress,
            "social_support": social_support,
            "financial_pressure": financial_pressure,
            "academic_pressure": academic_pressure,
            "screen_time_hours": screen_hours,
            "mental_health_awareness": awareness,
            TARGET: support_need,
        }
    )


@st.cache_resource
def train_model(data):
    X = data.drop(columns=[TARGET])
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    categorical_columns = X.select_dtypes(include="object").columns.tolist()
    numeric_columns = X.select_dtypes(exclude="object").columns.tolist()

    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )

    model = Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    positive_index = list(model.classes_).index(POSITIVE)
    probabilities = model.predict_proba(X_test)[:, positive_index]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "roc_auc": roc_auc_score(
            (y_test == POSITIVE).astype(int), probabilities
        ),
        "matrix": confusion_matrix(
            y_test, predictions, labels=model.classes_
        ),
        "report": classification_report(
            y_test, predictions, output_dict=True
        ),
        "classes": model.classes_,
    }
    return model, metrics


data = generate_synthetic_data()
model, metrics = train_model(data)

st.title("Student Wellbeing in Bhutan")
st.caption(
    "A beginner machine-learning project using 800 fictional records"
)
st.warning(
    "This app is not a medical or diagnostic tool. The result must not be "
    "used for healthcare, academic, employment, or disciplinary decisions. "
    "If someone may be in immediate danger, contact local emergency services "
    "or a qualified professional."
)

overview_tab, data_tab, prediction_tab, results_tab = st.tabs(
    ["Overview", "Explore data", "Prediction demo", "Model results"]
)

with overview_tab:
    st.subheader("Project objective")
    st.write(
        "This app shows how logistic regression can identify patterns "
        "associated with a fictional support-recommendation label. The data "
        "were generated inside this Python file and are not survey responses "
        "from Bhutanese students."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Fictional records", f"{len(data):,}")
    col2.metric("Input features", len(data.columns) - 1)
    col3.metric("Algorithm", "Logistic regression")

    st.markdown("### Machine-learning workflow")
    st.write(
        "1. Generate synthetic data → 2. Split data → 3. Preprocess variables "
        "→ 4. Train model → 5. Evaluate model → 6. Display results"
    )
    st.info(
        "A real project would require informed consent, ethics approval, "
        "secure data handling, local validation, fairness testing, and "
        "qualified mental-health oversight."
    )

with data_tab:
    st.subheader("Exploratory data analysis")
    selected_feature = st.selectbox(
        "Choose a numeric feature",
        [
            "stress_level",
            "sleep_hours",
            "social_support",
            "academic_pressure",
            "financial_pressure",
            "screen_time_hours",
            "physical_activity_days",
        ],
    )

    figure, axis = plt.subplots(figsize=(9, 4.5))
    sns.histplot(
        data=data,
        x=selected_feature,
        hue=TARGET,
        multiple="stack",
        ax=axis,
    )
    axis.set_title(
        selected_feature.replace("_", " ").title()
        + " by Synthetic Outcome"
    )
    st.pyplot(figure)

    st.markdown("### First 20 fictional records")
    st.dataframe(data.head(20), use_container_width=True)
    st.download_button(
        "Download synthetic CSV",
        data.to_csv(index=False),
        "bhutan_student_wellbeing_synthetic.csv",
        "text/csv",
    )

with prediction_tab:
    st.subheader("Try one fictional student profile")
    left, right = st.columns(2)

    with left:
        age = st.slider("Age", 18, 26, 21)
        gender = st.selectbox(
            "Gender",
            ["Woman", "Man", "Non-binary/Prefer not to say"],
        )
        district = st.selectbox(
            "District", sorted(data["district"].unique())
        )
        residence = st.selectbox("Residence", ["Urban", "Rural"])
        sleep = st.slider(
            "Sleep hours per night", 3.5, 10.0, 7.0, 0.1
        )
        study = st.slider(
            "Study hours per day", 1.0, 12.0, 6.0, 0.5
        )
        activity = st.slider(
            "Physically active days per week", 0, 7, 3
        )

    with right:
        stress = st.slider("Stress level (1 low–10 high)", 1, 10, 5)
        support = st.slider(
            "Social support (1 low–10 high)", 1, 10, 6
        )
        finance = st.slider(
            "Financial pressure (1 low–10 high)", 1, 10, 5
        )
        academic = st.slider(
            "Academic pressure (1 low–10 high)", 1, 10, 6
        )
        screen = st.slider(
            "Screen time hours per day", 0.5, 12.0, 5.0, 0.5
        )
        awareness = st.selectbox(
            "Mental-health awareness", ["Low", "Medium", "High"]
        )

    profile = pd.DataFrame(
        [
            {
                "age": age,
                "gender": gender,
                "district": district,
                "residence": residence,
                "sleep_hours": sleep,
                "study_hours_per_day": study,
                "physical_activity_days": activity,
                "stress_level": stress,
                "social_support": support,
                "financial_pressure": finance,
                "academic_pressure": academic,
                "screen_time_hours": screen,
                "mental_health_awareness": awareness,
            }
        ]
    )

    if st.button("Run educational prediction", type="primary"):
        positive_index = list(model.classes_).index(POSITIVE)
        probability = model.predict_proba(profile)[0][positive_index]
        st.metric("Model-estimated probability", f"{probability:.1%}")

        if probability >= 0.50:
            st.info(
                "Demo result: support recommended. This is a model exercise, "
                "not a clinical conclusion."
            )
        else:
            st.success(
                "Demo result: no immediate flag. This does not prove that "
                "support is unnecessary."
            )

with results_tab:
    st.subheader("Hold-out test results")
    metric1, metric2 = st.columns(2)
    metric1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    metric2.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    figure, axis = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        metrics["matrix"],
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=metrics["classes"],
        yticklabels=metrics["classes"],
        ax=axis,
    )
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.set_title("Confusion Matrix")
    st.pyplot(figure)

    st.markdown("### Classification report")
    report_table = pd.DataFrame(metrics["report"]).transpose()
    st.dataframe(report_table.round(3), use_container_width=True)

    st.caption(
        "The random seed and train-test split are fixed, making the results "
        "reproducible. Strong results are expected because the target was "
        "created using a known synthetic rule."
    )

st.divider()
st.caption(
    "Student project | Synthetic data | Educational demonstration only"
)
