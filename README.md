Student Mental Health and Burnout Prediction
Project Overview
This project aims to analyze and predict student mental health risk levels, dropout risk, mental health index, and burnout scores using a comprehensive dataset. By leveraging machine learning models, we identify key factors influencing student well-being and academic outcomes, providing insights that can help in developing targeted interventions.

Dataset
The dataset used in this project contains 1 million entries with various features related to student demographics, academic life, personal well-being, and mental health indicators. Key features include:

Demographics: age, gender, academic_year
Academic Factors: study_hours_per_day, exam_pressure, academic_performance
Mental Health Indicators: stress_level, anxiety_score, depression_score, sleep_hours, physical_activity, social_support, screen_time, internet_usage, financial_stress, family_expectation
Target Variables: burnout_score, mental_health_index, risk_level (Low, Medium, High), dropout_risk
Methodology
The project follows a structured machine learning pipeline:

Data Loading and Initial Inspection: The dataset was loaded into a pandas DataFrame, and an initial inspection was performed to understand its structure, data types, and identify missing values.
Data Cleaning and Preprocessing:
Missing values were handled (though none were found in this specific dataset).
Categorical features (gender and risk_level) were encoded: gender using one-hot encoding and risk_level using label encoding (Low: 0, Medium: 1, High: 2) due to its ordinal nature.
Numerical features were transformed to integers based on specific rounding rules to ensure data consistency.
Exploratory Data Analysis (EDA) and Descriptive Analysis:
Distribution Analysis: Histograms and count plots were generated to visualize the distributions of all features, including numerical and encoded categorical variables.
Correlation Analysis: A heatmap was used to visualize the correlation matrix of numerical features. Strong correlations (absolute value > 0.5) were further visualized using scatter plots to understand the relationships between feature pairs.
Target Variable Analysis: The distribution of risk_level_encoded was analyzed, and box plots were used to illustrate how numerical features vary across different risk levels.
Gender Relationship: The relationship between encoded gender and risk level was explored using count plots.
Machine Learning Model Building: Multiple machine learning models were trained for different prediction tasks:
Classification (Predicting Risk Level): Logistic Regression and Random Forest Classifier were used to predict risk_level_encoded.
Regression (Predicting Dropout Risk): A Random Forest Regressor was trained to predict dropout_risk.
Regression (Predicting Mental Health Index): A Random Forest Regressor was trained to predict mental_health_index.
Regression (Predicting Burnout Score): A Random Forest Regressor was trained to predict burnout_score.
Hyperparameter tuning for the Random Forest Classifier was skipped due to sufficient untuned performance and computational cost for the large dataset.
Interactive Prediction Interface: An interactive interface using ipywidgets was developed to allow users to input feature values and obtain real-time predictions from the trained models for risk level, dropout risk, mental health index, and burnout score.
Key Findings and Model Performance
Model Performance:
Risk Level Classification:
Logistic Regression: Achieved an accuracy of 0.9334 (test set) and 0.9329 (cross-validation). While overall accuracy is high, recall for the 'High Risk' category (Class 2) was lower (0.60), indicating challenges in correctly identifying this minority class.
Random Forest Classifier: Achieved an accuracy of 0.9303 (test set) and 0.9298 (cross-validation). Similar to Logistic Regression, it performed very well on 'Low Risk' and 'Medium Risk', but less effectively on 'High Risk'.
Dropout Risk Regression:
Random Forest Regressor: Achieved a Mean Absolute Error (MAE) of 0.6938, Mean Squared Error (MSE) of 0.7921, and an R-squared (R²) of 0.5816. This R² value indicates that about 58% of the variance in dropout_risk can be explained by the model's features.
Mental Health Index Regression:
Random Forest Regressor: Achieved a Mean Absolute Error (MAE) of 0.1916, Mean Squared Error (MSE) of 0.0964, and an R-squared (R²) of 0.9466. This model shows excellent predictive capability for the mental_health_index.
Burnout Score Regression:
Random Forest Regressor: Achieved a Mean Absolute Error (MAE) of 0.7141, Mean Squared Error (MSE) of 0.8750, and an R-squared (R²) of 0.6931. This model demonstrates good predictive power for burnout_score.
Key Data Insights from EDA and Feature Importance:
Dominant Predictors for Risk Level: burnout_score and mental_health_index are the most influential features in predicting risk_level_encoded. stress_level, anxiety_score, and depression_score also contribute significantly.
Strong Correlations: A strong negative correlation exists between burnout_score and mental_health_index. Burnout_score, stress_level, anxiety_score, and depression_score are highly positively correlated. Mental_health_index is negatively correlated with these stress-related scores. Exam_pressure and study_hours_per_day show a notable positive correlation.
Target Variable Imbalance: The risk_level_encoded distribution is heavily skewed towards 'Low Risk', explaining the high overall accuracy but also the challenge in predicting 'High Risk' cases.
Feature Relationships with Risk Level: Higher burnout_score, stress_level, anxiety_score, and depression_score are associated with higher risk_level_encoded. Conversely, lower mental_health_index, sleep_hours, and social_support are generally linked to higher risk_level_encoded.
Gender Impact: While included, gender appears to have less influence on risk_level_encoded compared to psychological and academic stress-related variables.
These insights highlight the critical role of mental well-being indicators in determining a student's risk level and dropout likelihood. Interventions targeting these areas could be highly effective.

Conclusion
This project successfully developed and evaluated machine learning models for predicting various aspects of student mental health and academic risk. The interactive prediction interface provides a practical tool for exploring these predictions based on input features, offering valuable insights for educators, counselors, and policymakers aiming to support student well-being.
