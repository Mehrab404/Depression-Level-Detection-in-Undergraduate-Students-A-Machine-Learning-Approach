## README

# Depression Level Detection in Undergraduate Students
A machine learning pipeline to classify student depression into six severity levels using academic, psychological, and demographic factors, evaluated across Logistic Regression, XGBoost, SVM, KNN, and a Neural Network (MLP). Key result: XGBoost achieves 94.95% accuracy and 0.9495 weighted F1 on the held-out test set, with the composite Depression Value feature contributing the most to predictions.

## Overview
This project builds supervised models to detect depression severity levels (No, Minimal, Mild, Moderate, Moderately Severe, Severe) from 11 engineered features combining academic metrics (e.g., CGPA), psychological scores (anxiety, stress, depression), and demographics. The methodology includes careful preprocessing, stratified splits, SMOTE to address class imbalance, and Gaussian-noise augmentation, followed by rigorous evaluation using accuracy, weighted precision/recall, weighted F1, classification reports, and confusion matrices. The pipeline is intended to enable early screening tools in university contexts while emphasizing responsible use and privacy.

## Files
- Depression-Level-Detection-in-Undergraduate-Students-A-Machine-Learning-Approach.pdf — full report with dataset, methods, experiments, figures, and tables.
- Notebook(s) — end-to-end code for preprocessing, modeling, and evaluation; adapt names to the repository’s organization if different (e.g., Model Codes.ipynb).
- Demo video — YouTube project demo link to be included in this section without exposing the URL inline per repository policy.

## Demo video
A short project walkthrough video is available on YouTube; place the link here or in the repository sidebar/description while keeping this README link-free to comply with repository guidelines for concise overviews.

## Key findings
- XGBoost is the top performer with Accuracy 0.9495 and Weighted F1 0.9495 on the test set.
- Logistic Regression is highly competitive at 0.9419 accuracy and 0.9418 weighted F1, indicating strong linear separability given the engineered features.
- MLP achieves 0.9217 accuracy and 0.9214 weighted F1; SVM reaches 0.8763, and KNN trails at 0.6717.
- Depression Value dominates feature importance (~0.75 in XGBoost), with Stress, Anxiety, CGPA, Year, Age, Gender, and Scholarship features contributing modest additional signal.

## Dataset
The dataset comprises 1977 undergraduate students, primarily from Computer Science and Engineering, with an initial 39 variables reduced to 11 input features plus a 6-class target, Depression_Target, encoded from PHQ-9 severity thresholds. Class distribution is imbalanced with few “No” and “Minimal” cases and many “Moderately Severe” and “Severe,” motivating stratified splitting and SMOTE on the training data only. The final features include Anxiety Value, Stress Value, Depression Value, CGPA_numeric, Age_numeric, Year_numeric, gender one-hot variables, and scholarship one-hot variables.

## Preprocessing
- Feature engineering converts ranges (e.g., CGPA, age, year) to numeric midpoints and one-hot encodes categorical variables (gender, scholarship).
- Stratified train/test split at 80/20 ensures class proportions are preserved in evaluation (396 test instances).
- SMOTE is applied on the training set to balance classes, followed by Gaussian noise augmentation on numeric features to improve robustness.

## Models
- Logistic Regression, XGBoost, SVM, K-Nearest Neighbors (KNN), and a Multi-Layer Perceptron (MLP) are implemented with cross-validated hyperparameter tuning on the training set.
- Evaluation on the held-out test set reports overall metrics and class-wise diagnostics via confusion matrices and classification reports.
- XGBoost’s feature importance highlights the primacy of Depression Value, with auxiliary contributions from stress, anxiety, academic year, CGPA, age, gender, and scholarship indicators.

## Results
- XGBoost: Accuracy 0.9495; Weighted Precision 0.9505; Weighted Recall 0.9495; Weighted F1 0.9495.
- Logistic Regression: Accuracy 0.9419; Weighted F1 0.9418, closely trailing XGBoost with strong linear baselines.
- MLP: Accuracy 0.9217; Weighted F1 0.9214; SVM: Accuracy 0.8763; KNN: Accuracy 0.6717.[1]
- Confusion matrices show most errors occur between adjacent severity levels, consistent with ordinal proximity in clinical interpretations.
  
## Environment
- Python stack using scikit-learn for classical models and metrics, imbalanced-learn for SMOTE, and XGBoost for boosted trees; MLP implemented via a standard deep learning library where applicable.
- Reproduction requires typical data-science dependencies and exact preprocessing steps to align encodings and class mappings.
- Keep random seeds and stratification consistent to reduce variance across runs during validation and testing.[1]

## Installation
Create and activate a Python environment (venv or conda), then install required packages before running preprocessing and training.
```
pip install scikit-learn xgboost imbalanced-learn numpy pandas matplotlib seaborn
```
Ensure that any additional deep learning framework used for the MLP (e.g., TensorFlow or PyTorch) is installed if reproducing those results.

## Configuration
Default experimental settings follow the paper’s pipeline: stratified 80/20 split, SMOTE on training only, Gaussian noise augmentation on numeric features, and cross-validated hyperparameter tuning per model. The target classes follow PHQ-9-derived bins encoded as integers 0–5; maintain these thresholds and encodings for comparability across experiments.

Example YAML:
```
experiment:
  split:
    test_size: 0.2
    stratify: true
    random_state: 42
  imbalance:
    smote: true
    apply_on: train_only
  augmentation:
    gaussian_noise: true
    noise_std: 0.01
  models:
    - logreg
    - xgboost
    - svm
    - knn
    - mlp
  metrics:
    - accuracy
    - precision_weighted
    - recall_weighted
    - f1_weighted
```


## Reproduction guide
- Preprocess: Apply the same feature engineering, one-hot encodings, numeric conversions, and target mapping as described to produce the 11 input features and 6-class target.
- Balance: Fit SMOTE on the training split only, then optionally apply Gaussian noise to numeric features in training for robustness.
- Train and evaluate: Run cross-validated tuning per model and evaluate on the untouched test set with accuracy, weighted precision/recall/F1, classification reports, and confusion matrices.
- 
## Training examples
XGBoost sketch:
```
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score

xgb = XGBClassifier(
    objective='multi:softprob',
    num_class=6,
    eval_metric='mlogloss',
    tree_method='hist',
    random_state=42
)

param_grid = {
    'n_estimators': [200, 400],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

clf = GridSearchCV(xgb, param_grid, scoring='f1_weighted', cv=5, n_jobs=-1)
clf.fit(X_train_bal, y_train_bal)

y_pred = clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Weighted F1:', f1_score(y_test, y_pred, average='weighted'))
print(classification_report(y_test, y_pred))
```


## Evaluation
- Primary metrics: Accuracy, Weighted Precision, Weighted Recall, Weighted F1 to reflect multiclass balance and practical performance under class skew.
- Diagnostics: Confusion matrices and per-class reports to examine boundary errors between adjacent severity levels and minority classes.
- Best configuration: XGBoost, followed closely by Logistic Regression; MLP strong but behind linear/boosted-tree baselines in this setting.

## Limitations and headroom
- Dataset scope: Single-institution, primarily CSE students; results may not directly generalize to other populations without external validation.
- Temporal dynamics: Cross-sectional data cannot capture within-student fluctuations or progression, limiting temporal inference.
- Feature dominance: Depression Value provides most signal; future work should target earlier upstream signals beyond direct symptom scores to enable pre-symptomatic screening.
- With fewer processing constraints—broader hyperparameter sweeps, larger ensembles, additional data sources, and more extensive model selection—overall performance would likely increase beyond the current best, especially for edge classes and ordinal boundary calibration.

## Ethics and responsible use
Predictions should augment, not replace, professional clinical judgment, and any deployment must adhere to privacy, consent, and data governance standards appropriate for sensitive health information in educational settings. Screening outputs should be paired with accessible support pathways to mitigate risks associated with false positives/negatives.

## Future work
- External validation across institutions and disciplines to assess generalizability and fairness.
- Longitudinal data collection to model trajectories and enable proactive interventions.
- Feature expansion to include behavioral, social, and contextual signals (e.g., sleep, workload, support networks), plus ordinal/regression objectives and calibration.

## Project structure (suggested)
This structure supports clarity and reproducibility; adapt paths and names to the existing codebase as needed.
```
.
├── data/
├── notebooks/
│   └── <analysis_notebook>.ipynb
├── reports/
│   └── Depression-Level-Detection-in-Undergraduate-Students-A-Machine-Learning-Approach.pdf
├── src/
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   └── eval/
├── results/
└── README.md
```


## Citation
If using this project, cite: “Depression Level Detection in Undergraduate Students: A Machine Learning Approach Using Academic and Psychological Factors” by Mehrabul Islam and Khandoker Wahiduzzaman Anik (BRAC University). The report documents dataset construction, preprocessing, modeling, and full comparative results including confusion matrices and feature importance.

## Acknowledgments
Thanks to the tools enabling this pipeline: scikit-learn for classical models and evaluation, imbalanced-learn for SMOTE, XGBoost for boosted trees, and standard deep learning libraries for the MLP baseline. Appreciation is due to the students who contributed data and the institutional context supporting mental health research and early detection initiatives.

