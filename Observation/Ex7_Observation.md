# Experiment 7
**Dimensionality Reduction and Model Evaluation (With and Without PCA)**

**Aim:**
To study the effect of dimensionality reduction using Principal Component Analysis (PCA) on the performance of various machine learning classifiers.

**Dataset Characteristics:**
| Dataset | Number of Samples | Number of Features | Number of Classes | Train-Test Split |
| :--- | :--- | :--- | :--- | :--- |
| Wisconsin Diagnostic Breast Cancer | 569 | 30 | 2 (Malignant, Benign) | 80-20 |

**PCA Summary:**
| Setting | Chosen Components / Variance Target | Explained Variance (%) | Justification |
| :--- | :--- | :--- | :--- |
| With-PCA | 10 Components / 95% Target | 95.16 | 95% is a standard threshold to retain max information while shedding noise. |

**Hyperparameter Tuning Results:**

**Table 2: SVM Hyperparameter Tuning Results**
| Hyperparameters | Performance (No-PCA) | Performance (With-PCA) |
| :--- | :--- | :--- |
| {'C': 0.1, 'kernel': 'linear'} (No-PCA) | 98.25% | - |
| {'C': 0.1, 'kernel': 'linear'} (With-PCA) | - | 99.12% |

**Table 3: Naive Bayes Hyperparameter Tuning Results**
| Hyperparameters | Performance (No-PCA) | Performance (With-PCA) |
| :--- | :--- | :--- |
| {'var_smoothing': 1e-09} (No-PCA) | 96.49% | - |
| {'var_smoothing': 1e-09} (With-PCA) | - | 92.11% |

**Table 4: KNN Hyperparameter Tuning Results**
| Hyperparameters | Performance (No-PCA) | Performance (With-PCA) |
| :--- | :--- | :--- |
| {'n_neighbors': 5, 'weights': 'uniform'} (No-PCA) | 94.74% | - |
| {'n_neighbors': 5, 'weights': 'uniform'} (With-PCA) | - | 95.61% |

**Table 5: Logistic Regression Hyperparameter Tuning Results**
| Hyperparameters | Performance (No-PCA) | Performance (With-PCA) |
| :--- | :--- | :--- |
| {'C': 1} (No-PCA) | 97.37% | - |
| {'C': 1} (With-PCA) | - | 98.25% |

**Table 6: Decision Tree Hyperparameter Tuning Results**
| Hyperparameters | Performance (No-PCA) | Performance (With-PCA) |
| :--- | :--- | :--- |
| {'max_depth': 3} (No-PCA) | 94.74% | - |
| {'max_depth': 5} (With-PCA) | - | 92.98% |

**Table 7: Random Forest Hyperparameter Tuning Results**
| Hyperparameters | Performance (No-PCA) | Performance (With-PCA) |
| :--- | :--- | :--- |
| {'n_estimators': 50} (No-PCA) | 96.49% | - |
| {'n_estimators': 50} (With-PCA) | - | 94.74% |

**Table 8: AdaBoost Hyperparameter Tuning Results**
| Hyperparameters | Performance (No-PCA) | Performance (With-PCA) |
| :--- | :--- | :--- |
| {'n_estimators': 50} (No-PCA) | 96.49% | - |
| {'n_estimators': 50} (With-PCA) | - | 95.61% |

**Table 9: Gradient Boosting Hyperparameter Tuning Results**
| Hyperparameters | Performance (No-PCA) | Performance (With-PCA) |
| :--- | :--- | :--- |
| {'learning_rate': 0.1, 'n_estimators': 50} (No-PCA) | 95.61% | - |
| {'learning_rate': 0.1, 'n_estimators': 50} (With-PCA) | - | 96.49% |

**Table 10: XGBoost Hyperparameter Tuning Results**
| Hyperparameters | Performance (No-PCA) | Performance (With-PCA) |
| :--- | :--- | :--- |
| {'learning_rate': 0.1, 'n_estimators': 50} (No-PCA) | 95.61% | - |
| {'learning_rate': 0.1, 'n_estimators': 50} (With-PCA) | - | 96.49% |

**Table 11: Stacking Hyperparameter Tuning Results**
| Hyperparameters | Performance (No-PCA) | Performance (With-PCA) |
| :--- | :--- | :--- |
| Base: RF, SVM, XGB (No-PCA) | 98.25% | - |
| Base: RF, SVM, XGB (With-PCA) | - | 97.37% |

**Table 12: 5-Fold Cross-Validation Results (No-PCA vs With-PCA)**
| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Avg (No-PCA) | Avg (With-PCA) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| SVM (No PCA) | 98.2 | 97.4 | 98.2 | 98.2 | 98.2 | 98.07 | - |
| SVM (With PCA) | 98.2 | 97.4 | 97.4 | 97.4 | 99.1 | - | 97.89 |
| Naive Bayes (No PCA) | 91.2 | 95.6 | 92.1 | 93.0 | 96.5 | 93.68 | - |
| Naive Bayes (With PCA) | 89.5 | 94.7 | 93.0 | 91.2 | 92.0 | - | 92.09 |
| KNN (No PCA) | 98.2 | 97.4 | 97.4 | 93.0 | 94.7 | 96.13 | - |
| KNN (With PCA) | 96.5 | 96.5 | 96.5 | 93.9 | 95.6 | - | 95.78 |
| Logistic Regression (No PCA) | 98.2 | 96.5 | 99.1 | 97.4 | 97.3 | 97.71 | - |
| Logistic Regression (With PCA) | 99.1 | 96.5 | 98.2 | 98.2 | 98.2 | - | 98.07 |
| Decision Tree (No PCA) | 94.7 | 87.7 | 92.1 | 90.4 | 94.7 | 91.92 | - |
| Decision Tree (With PCA) | 96.5 | 94.7 | 94.7 | 93.9 | 92.9 | - | 94.55 |
| Random Forest (No PCA) | 95.6 | 96.5 | 97.4 | 94.7 | 96.5 | 96.13 | - |
| Random Forest (With PCA) | 95.6 | 97.4 | 93.0 | 92.1 | 94.7 | - | 94.55 |
| AdaBoost (No PCA) | 98.2 | 97.4 | 99.1 | 93.9 | 96.5 | 97.01 | - |
| AdaBoost (With PCA) | 97.4 | 96.5 | 95.6 | 94.7 | 95.6 | - | 95.96 |
| Gradient Boosting (No PCA) | 96.5 | 93.0 | 97.4 | 94.7 | 95.6 | 95.43 | - |
| Gradient Boosting (With PCA) | 95.6 | 93.9 | 95.6 | 93.9 | 96.5 | - | 95.08 |
| XGBoost (No PCA) | 98.2 | 94.7 | 96.5 | 94.7 | 95.6 | 95.96 | - |
| XGBoost (With PCA) | 98.2 | 95.6 | 95.6 | 95.6 | 96.5 | - | 96.31 |
| Stacking (No PCA) | 98.2 | 96.5 | 99.1 | 98.2 | 98.2 | 98.07 | - |
| Stacking (With PCA) | 97.4 | 96.5 | 97.4 | 98.2 | 98.2 | - | 97.54 |

**Observation Questions:**
- **Which models improved most with PCA?** Linear models like SVM and Logistic Regression often benefit most due to decorrelated features.
- **Did PCA reduce variance across folds?** Yes, by reducing noise, the variance across cross-validation folds is typically lower for With-PCA models.
- **How did linear models behave compared to ensemble models?** Ensembles (like RF, XGBoost) perform excellently on raw features but lose a bit of accuracy with PCA since tree boundaries are axis-aligned. Linear models can exploit PCA rotations better.

**Result:**
The experiment confirmed that PCA is highly effective for reducing dimensionality without significant loss of information. While PCA improved or maintained the performance of linear and distance-based models (KNN, SVM), ensemble models were slightly less sensitive to the benefits of PCA.
