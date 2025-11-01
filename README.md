## Loan Risk Predictor

A machine learning project that predicts loan approval risk (High Risk or Low Risk) based on financial and demographic data.
The notebook performs exploratory data analysis, preprocessing, feature engineering, and model training using a Random Forest classifier.

### Dataset

The dataset used in this project is:
financial_risk_analysis_large.csv
It was accessed from the Hugging Face dataset hub:
```
hf://datasets/Naat97/financial-risk-data/financial_risk_analysis_large.csv
```
The target column is renamed to Risk, with values:

High Risk → Loan likely to default.

Low Risk → Loan likely to be repaid.

### Workflow Overview
1. Data Preprocessing
```
Encoded target values (1 → Low Risk, 0 → High Risk)

Balanced dataset using downsampling of the majority class

Split into training (70%) and testing (30%) sets
```
2. Feature Engineering
```
Identified numerical and categorical features

Applied preprocessing pipeline using:

StandardScaler for numerical features

OneHotEncoder for categorical features
```
3. Exploratory Data Analysis (EDA)
```
Checked for class imbalance

Computed feature correlations

Visualized correlation matrix using a Seaborn heatmap

Observation: No strong correlations → PCA not suitable here
```
4. Model Training

Model used: RandomForestClassifier

Parameters:
```
n_estimators = 200

class_weight = 'balanced'

random_state = 42
```
### Evaluated using:
```
accuracy_score

classification_report

confusion_matrix
```
### Results

The model achieved strong performance on the test set, with well-balanced precision and recall across both classes.
(Random forest was found to outperform logistic regression during experimentation.)

### Tech Stack

Python
```
Pandas, NumPy — Data handling

Matplotlib, Seaborn — Visualization

Scikit-learn — ML modeling and preprocessing

Imbalanced-learn (imblearn) — Data resampling

Hugging Face Datasets — Dataset source
```

## How to Run

Clone the repository:
```
git clone <https://github.com/here-2007/Loan-Risk>
cd loan-risk-predictor

```
Install dependencies:
```
pip install -r requirements.txt
```

Run the notebook:
```
jupyter notebook loan-risk-predictor.ipynb
```
