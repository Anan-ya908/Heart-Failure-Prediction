# Heart Disease Prediction

Predicts the presence of heart disease from patient clinical data, comparing a baseline
Logistic Regression model against a Voting Classifier ensemble (Logistic Regression + Random Forest).

## Dataset

- **Source:** UCI Heart Disease dataset (303 patient records, 13 clinical features)
- **Features:** age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar,
  resting ECG, max heart rate, exercise-induced angina, ST depression, slope, number of major
  vessels, thalassemia
- **Target:** `1` = heart disease present, `0` = no heart disease

## Approach

1. Exploratory data analysis - class balance, correlation between features
2. Train/test split (80/20, stratified, `random_state=2`)
3. Baseline model: Logistic Regression (scaled features)
4. Ensemble model: soft-voting classifier combining Logistic Regression + Random Forest
5. Evaluation with accuracy, precision, recall, and F1-score (not accuracy alone, since
   false negatives matter more than false positives in a medical screening context)

## Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression (baseline) | 78.7% | 81.3% | 78.8% | 80.0% |
| Voting Ensemble (LR + Random Forest) | **85.3%** | 85.3% | 87.9% | **86.6%** |

**+8.3% relative accuracy improvement** from the ensemble over the baseline.

## Limitations

- Small dataset (303 rows) - a single train/test split has meaningful variance; results
  should ideally be confirmed with k-fold cross-validation.
- This is the standard UCI benchmark dataset, used here to demonstrate methodology
  (ensembling, proper evaluation metrics) rather than for clinical use.

## How to run

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
jupyter notebook Heart_Disease_Prediction_Enhanced.ipynb
```

Requires `heart_disease_data.csv` in the same directory.
