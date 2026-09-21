# Experiment 6
**Bagging, Boosting, and Stacked Ensemble Models**

**Aim:**
To understand and implement ensemble learning strategies including Bagging, Boosting, and Stacking. To compare these models in terms of accuracy, stability, and generalization on the Wisconsin Diagnostic Breast Cancer dataset.

**Algorithm/Theory:**
1. **Bagging:** Uses a `BaggingClassifier` with `DecisionTreeClassifier` as the base model, training on bootstrap samples to reduce variance.
2. **Boosting (AdaBoost):** Sequentially fits copies of a classifier, adjusting weights of misclassified instances to reduce bias.
3. **Boosting (Gradient Boosting):** Builds an additive model in a forward stage-wise fashion, optimizing arbitrary differentiable loss functions.
4. **Stacked Ensemble:** Combines diverse base models (`SVC`, `GaussianNB`, `DecisionTreeClassifier`) and trains a meta-model (`LogisticRegression`) on their predictions.

**Dataset Characteristics:**
| Dataset | Number of Samples | Number of Features | Number of Classes | Train-Test Split |
| :--- | :--- | :--- | :--- | :--- |
| Wisconsin Diagnostic Breast Cancer | 569 | 30 | 2 (Malignant, Benign) | 80-20 |

**Hyperparameter Tuning:**

**Table 1: Bagging Hyperparameter Evaluation**
| n_estimators | max_samples | Avg CV Accuracy (%) | Avg CV F1 Score |
| :--- | :--- | :--- | :--- |
| 50 | 0.8 | 96.13 | 0.9696 |

**Table 2: Boosting Hyperparameter Evaluation**
| n_estimators | learning_rate | Avg CV Accuracy (%) | Avg CV F1 Score |
| :--- | :--- | :--- | :--- |
| 50 | 1.0 | 97.01 | 0.9765 |

**Table 3: Stacked Ensemble Evaluation**
| Base Models | Meta Learner | Avg CV Accuracy (%) | Avg CV F1 Score |
| :--- | :--- | :--- | :--- |
| DT, SVM, NB | Logistic Reg | 97.01 | 0.9763 |

**Performance Comparison of Ensemble Models:**
| Model | Accuracy (%) | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- |
| Bagging | 96.49 | 0.9589 | 0.9859 | 0.9722 |
| AdaBoost | 96.49 | 0.9589 | 0.9859 | 0.9722 |
| Gradient Boosting | 95.61 | 0.9583 | 0.9718 | 0.9650 |
| Stacked Ensemble | 96.49 | 0.9589 | 0.9859 | 0.9722 |

**Result:**
The ensemble learning strategies were successfully implemented. The models achieved over 96% accuracy with Stacked Ensemble and AdaBoost demonstrating robust cross-validated F1 scores.
