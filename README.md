# Student Mental Health & Performance Data Analytics

A comprehensive data analytics and machine learning project exploring the relationships between student mental health indicators and academic performance, using a dataset of 1 million student records.

## Overview

This project performs end-to-end data analysis and predictive modeling on student mental health and burnout data. It covers data cleaning, exploratory data analysis (EDA), correlation analysis, and the training of multiple machine learning models to predict key student well-being outcomes. The notebook concludes with an interactive prediction interface built using `ipywidgets`.

## Dataset

- **File:** `student_mental_health_burnout_1M.csv`
- **Size:** ~1 million student records
- **Key Features:**
  - Academic: `academic_year`, `study_hours_per_day`, `exam_pressure`, `academic_performance`
  - Mental Health: `stress_level`, `anxiety_score`, `depression_score`, `burnout_score`, `mental_health_index`
  - Lifestyle: `sleep_hours`, `physical_activity`, `social_support`, `screen_time`, `internet_usage`
  - Socioeconomic: `financial_stress`, `family_expectation`
  - Demographic: `age`, `gender`
  - Targets: `risk_level` (Low / Medium / High), `dropout_risk`

## Project Structure

```
├── student_mental_health_burnout_1M.csv   # Dataset (not included in repo)
└── Student_Mental_Health_and_Performance_Data_Analytics_Project.ipynb
```

## Notebook Sections

### 1. Data Loading & Initial Inspection
- Load the CSV into a pandas DataFrame
- Inspect data types, shape, and descriptive statistics
- Check for missing values

### 2. Data Cleaning & Preprocessing
- Drop rows with missing values
- One-hot encode `gender`
- Label encode `risk_level` (Low=0, Medium=1, High=2)
- Apply rounding rules to convert float features to integers (ceil, floor, round)

### 3. Exploratory Data Analysis (EDA)
- **3.1** Distribution plots for all numerical features (histograms + KDE)
- **3.2** Correlation heatmap; strong-correlation heatmap (|r| > 0.5); scatter plots for strongly correlated feature pairs
- **3.3** Target variable (`risk_level_encoded`) distribution
- **3.4** Box plots showing feature distributions across risk levels
- **3.5** Gender vs. risk level analysis

### 4. Machine Learning Models

| Task | Model | Metric | Score |
|------|-------|--------|-------|
| Risk Level Classification | Logistic Regression | Accuracy | 93.34% |
| Risk Level Classification | Random Forest Classifier | Accuracy | 93.03% |
| Dropout Risk Regression | Random Forest Regressor | R² | 0.5816 |
| Mental Health Index Regression | Random Forest Regressor | — | — |
| Burnout Score Regression | Random Forest Regressor | — | — |

- 70/30 train-test split with stratification (classification)
- 5-fold stratified cross-validation
- `StandardScaler` applied for Logistic Regression
- Feature importance analysis for Random Forest Classifier

### 5. Interactive Prediction Interface
Three `ipywidgets`-powered panels for real-time predictions:
- **Panel 1:** Predict Risk Level (Logistic Regression + Random Forest) & Dropout Risk
- **Panel 2:** Predict Mental Health Index
- **Panel 3:** Predict Burnout Score

## Key Findings

1. **`burnout_score`** and **`mental_health_index`** are the strongest predictors of student risk level.
2. `stress_level`, `anxiety_score`, and `depression_score` are highly positively correlated with each other and with `burnout_score`.
3. `mental_health_index` is strongly negatively correlated with `stress_level`, `anxiety_score`, and `depression_score`.
4. The dataset is heavily skewed toward the **Low Risk** class, making correct identification of **High Risk** students the primary challenge.
5. Lower `sleep_hours` and `social_support` are clearly associated with higher risk levels.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
ipywidgets
scipy
```

Install all dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn ipywidgets scipy
```

## Usage

1. Upload the notebook to [Google Colab](https://colab.research.google.com/) or run locally in Jupyter.
2. Place `student_mental_health_burnout_1M.csv` at `/content/` (Colab) or update `file_path` in the first cell.
3. Run all cells in order (`Runtime → Run all` in Colab).
4. Scroll to **Section 5** to use the interactive prediction panels.

## License

This project is for educational and research purposes.
