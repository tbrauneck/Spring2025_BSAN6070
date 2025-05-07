
# Gender Pay Gap Analysis Using Ensemble Learning on ACS Data

## Overview

This project investigates gender-based income disparities in the United States by leveraging ensemble machine learning models on the American Community Survey (ACS) dataset. The primary objective is to predict individual incomes and analyze the influence of gender, occupation, education, and other demographic factors on wage disparities.

## Project Structure

* **Data Source**: American Community Survey (ACS) Public Use Microdata Sample (PUMS)
* **Main Notebook**: `BSAN6070_final_project_v2025.05.06.0.ipynb`
* **Models Implemented**:

  * Gradient Boosting Regressor
  * XGBoost Regressor
  * AdaBoost Regressor

## Methodology

### 1. Data Preprocessing

* **Feature Engineering**: Categorical variables were encoded, and relevant features such as occupation codes, education levels, and weekly work hours were selected.
* **Handling Missing Values**: Rows with missing critical information were removed to ensure data quality.


### 2. Model Training
We trained multiple ensemble models, including Gradient Boosting Regressor, XGBoost Regressor, and AdaBoost Regressor, to predict individual income based on demographic and socioeconomic features.

Training Workflow for Each Model Included:

* Baseline Initialization: Models were initially trained using basic hyperparameters to establish a reference point.

* Hyperparameter Tuning: Extensive hyperparameter searches were conducted using RandomizedSearchCV with 5-fold cross-validation. For example, the XGBoost model explored parameters such as n_estimators, learning_rate, max_depth, subsample, and colsample_bytree to optimize performance.

* Cross-Validation: A KFold strategy with shuffling was employed to ensure stable and generalizable results across different subsets of the training data.

* Manual Hyperparameter Sensitivity Testing: To deepen understanding of model behavior, individual hyperparameters (e.g., learning_rate, max_depth, n_estimators) were manually varied, and performance metrics such as MAPE were plotted to assess their impact.

* Performance Assessment: Final models were evaluated using R² and MAPE on the test set. Feature importance was derived from the best-performing models using XGBoost’s built-in importance scores.
### 3. Model Evaluation

* **Metrics Used**:

  * Mean Absolute Percentage Error (MAPE)
  * R-squared (R²)
* **Feature Importance Analysis**: Applied permutation importance to identify key predictors influencing income.

### 4. Counterfactual Analysis

Simulated income predictions by altering the gender variable while keeping other features constant to assess the direct impact of gender on income predictions.

## Results

* **Top Predictors**:

  * Occupation code
  * Education level
  * Weekly work hours
* **Model Performance**:

| Model             | Test MAPE | R²   |
| ----------------- | --------- | ---- |
| Gradient Boosting | 5.92%     | 0.41 |
| XGBoost           | 5.85%     | 0.43 |
| AdaBoost          | 6.04%     | 0.39 |

* **Key Findings**:

  * Occupation had a significantly higher impact on income predictions compared to gender.
  * The interaction between education and occupation explained a notable portion of the variance in income.

## Model Deployment
To make the analysis accessible and interactive, we deployed the final model using Streamlit, a Python-based web framework for data science apps.

Deployment Features:

* User Interface: Users can input demographic and work-related variables (e.g., gender, occupation, education, weekly work hours) through dropdowns and sliders.

* Prediction Output: The app returns an estimated income based on the trained model and provides insight into how changes in input variables affect the prediction.

* Visualization: Embedded charts display feature importance and counterfactual scenarios to interpret model behavior.



## Conclusion

The analysis confirms the presence of a gender pay gap in the U.S., with occupation and education playing pivotal roles. Ensemble learning models, particularly XGBoost, effectively captured the nuances in the data, providing a robust framework for such socio-economic analyses.

## Dependencies

* Python 3.8+
* pandas
* numpy
* scikit-learn
* xgboost
* matplotlib
* seaborn

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/tbrauneck/Spring2025_BSAN6070.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Spring2025_BSAN6070/Final_Project
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Jupyter Notebook:

   ```bash
   jupyter notebook BSAN6070_final_project_v2025.05.06.0.ipynb
   ```

