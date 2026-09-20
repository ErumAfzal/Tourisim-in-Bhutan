# Mental Health and Student Wellbeing in Bhutan

## A Basic Machine-Learning Application Using Streamlit

This student project demonstrates a complete beginner-friendly machine-learning workflow using Python, scikit-learn, and Streamlit. The application generates a **synthetic dataset of 800 fictional student profiles**, trains a logistic-regression model, presents exploratory charts, and provides an interactive prediction demonstration.

> **Important:** This project uses synthetic data only. It does not contain information collected from real Bhutanese students and must not be used as a medical, psychological, screening, or diagnostic tool.

## Project Objective

The project investigates the following educational research question:

> Can basic lifestyle, academic-pressure, and social-support variables predict a synthetically generated “support recommended” label?

The purpose is to teach students how to:

- Generate and explore a dataset
- Prepare numeric and categorical variables
- Divide data into training and testing sets
- Train a classification model
- Evaluate model performance
- Build an interactive Streamlit interface
- Discuss limitations and responsible AI use

## Project Structure

```text
bhutan-mental-health-streamlit/
├── app.py
├── requirements.txt
└── README.md
```

The entire application, dataset-generation process, and machine-learning pipeline are contained in `app.py`. No separate dataset or model file is required.

## Dataset

The application automatically generates 800 fictional records for adult students aged 18–26. The variables include:

| Variable | Description |
|---|---|
| `age` | Fictional student age |
| `gender` | Gender category |
| `district` | Selected Bhutanese district |
| `residence` | Urban or rural residence |
| `sleep_hours` | Average hours of sleep |
| `study_hours_per_day` | Daily study time |
| `physical_activity_days` | Active days per week |
| `stress_level` | Stress score from 1 to 10 |
| `social_support` | Social-support score from 1 to 10 |
| `financial_pressure` | Financial-pressure score from 1 to 10 |
| `academic_pressure` | Academic-pressure score from 1 to 10 |
| `screen_time_hours` | Daily screen time |
| `mental_health_awareness` | Low, medium, or high awareness |
| `support_need` | Synthetic target variable |

The target has two classes:

- `Support recommended`
- `No immediate flag`

The target is created through a transparent fictional rule plus random noise. Therefore, the results cannot be generalized to students in Bhutan.

## Machine-Learning Method

The application uses **logistic regression** because it is suitable for basic binary classification and is easier to explain than many complex models.

The workflow is:

1. Generate reproducible synthetic data using random seed 42.
2. Separate the input variables and target variable.
3. Divide the dataset into 80% training and 20% testing data.
4. Replace missing numeric values with the median.
5. Replace missing categorical values with the most frequent category.
6. Standardize numeric variables.
7. One-hot encode categorical variables.
8. Train a logistic-regression classifier.
9. Evaluate the model using unseen testing data.

## Evaluation Measures

The application presents:

- Accuracy
- ROC-AUC
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report

Using the fixed synthetic dataset and test split, the expected results are approximately:

- **Accuracy: 78.7%**
- **ROC-AUC: 0.892**

These values show how the model performs on its generated test data. They do not demonstrate clinical validity.

## Streamlit Pages

The application contains four tabs:

1. **Overview** – explains the project, model, and responsible-use conditions.
2. **Explore data** – displays a selectable distribution chart and fictional records.
3. **Prediction demo** – allows users to enter one fictional profile.
4. **Model results** – presents evaluation metrics and the confusion matrix.

## Installation and Local Use

### 1. Download or clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/bhutan-mental-health-streamlit.git
cd bhutan-mental-health-streamlit
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the packages

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application should open at `http://localhost:8501`.

## requirements.txt

Create a file named `requirements.txt` and add:

```txt
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
```

## Deploy on Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload `app.py`, `requirements.txt`, and `README.md`.
3. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
4. Select **Create app**.
5. Select the GitHub repository and branch.
6. Enter `app.py` as the main file path.
7. Select **Deploy**.
8. Open the deployed application and test every tab.

## Ethical Considerations

Mental-health information is sensitive. A real project would require:

- Research ethics approval
- Informed and voluntary consent
- Secure and restricted data storage
- Collection of only necessary information
- Anonymization or pseudonymization
- Validation with Bhutanese students and professionals
- Fairness testing across relevant groups
- A safe referral and support procedure
- Qualified mental-health oversight

The model must not determine healthcare, university admission, grades, employment, discipline, or access to services.

## Limitations

- The dataset is artificial rather than collected from participants.
- The target is generated and not based on a validated clinical assessment.
- The included districts are used only to provide a recognizable interface context.
- Logistic regression may not capture complex relationships.
- The application has not been externally validated.
- The results cannot be interpreted as findings about mental health in Bhutan.
- Strong performance is partly expected because the target follows a known synthetic rule.

## Future Improvements

With ethics approval and local collaboration, future work could:

- Conduct an anonymous and culturally appropriate survey
- Compare multiple machine-learning algorithms
- Use cross-validation and probability calibration
- Assess fairness and performance across relevant groups
- Add model-explanation techniques
- Include qualitative interviews
- Co-design the application with Bhutanese students and professionals

## Suggested Screenshots for Submission

Include screenshots of:

1. Application overview
2. Data-exploration chart
3. Fictional-profile prediction
4. Accuracy and ROC-AUC
5. Confusion matrix
6. Classification report

## Suggested Viva Questions

1. Why was logistic regression selected?
2. Why is the dataset divided into training and testing sets?
3. What does one-hot encoding do?
4. Why are numeric variables standardized?
5. What is the difference between accuracy and recall?
6. What does ROC-AUC measure?
7. Why can this application not diagnose mental illness?
8. What are the advantages and limitations of synthetic data?
9. What ethical approval would real data collection require?
10. How could the model be improved in future research?

## Disclaimer

This repository is a student learning project. The application and its predictions are for educational demonstration only and are not professional advice. Anyone concerned about their wellbeing should contact an appropriately qualified professional or local support service.

## Author

**Student name:** Add your name  
**Institution:** Add your institution  
**Course:** Add your course  
**Year:** 2026

