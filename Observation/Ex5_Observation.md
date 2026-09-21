# Experiment 5
**Breast Cancer Classification using Decision Tree and Random Forest**

**Aim:**
To implement and compare Decision Tree and Random Forest classifiers on the Wisconsin Diagnostic Breast Cancer dataset. Evaluate the effect of hyperparameter tuning to mitigate tree overfitting and compare standalone trees versus bagging ensembles.

**Hyperparameter Tuning (Grid Search)**

| Model | Best Parameters |
| :--- | :--- |
| Decision Tree | criterion: entropy, max_depth: 5, min_samples_split: 5 |
| Random Forest | n_estimators: 100, max_depth: 10, bootstrap: True |

**Evaluation Metrics Comparison**

| Metric | Decision Tree | Random Forest |
| :--- | :--- | :--- |
| Accuracy | 0.9386 | 0.9649 |
| Precision | 0.9302 | 0.9762 |
| Recall | 0.9302 | 0.9535 |
| F1-Score | 0.9302 | 0.9647 |
| ROC-AUC | 0.9351 | 0.9930 |

**5-Fold Cross-Validation Accuracy**

| Fold | Decision Tree | Random Forest |
| :--- | :--- | :--- |
| Fold 1 | 0.9211 | 0.9649 |
| Fold 2 | 0.9298 | 0.9737 |
| Fold 3 | 0.9386 | 0.9561 |
| Fold 4 | 0.9474 | 0.9649 |
| Fold 5 | 0.9381 | 0.9646 |
| **Average** | **0.9350** | **0.9648** |

**Learning Outcomes:**
* Understood the structure of Decision Trees and how splitting criteria (Gini vs Entropy) determine node splits to maximize information gain.
* Recognized the tendency of singular, deep Decision Trees to overfit to the training data.
* Learned how Random Forest ensembles overcome overfitting and reduce variance by aggregating predictions from multiple bootstrapped decision trees.
