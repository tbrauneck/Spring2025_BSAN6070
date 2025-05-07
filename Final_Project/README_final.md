
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

Each model underwent:

* **Baseline Training**: Initial models were trained with default or basic hyperparameters.
* **Hyperparameter Optimization**: Utilized `RandomizedSearchCV` with 100 iterations to fine-tune model parameters.
* **Cross-Validation**: Employed 5-fold cross-validation to assess model performance.

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

