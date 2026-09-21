# Experiment 2
**Email Spam/Ham Classification using Naïve Bayes and KNN**

**Aim:**
To build and evaluate email spam classifiers using Naïve Bayes and K-Nearest Neighbours algorithm. Analyse variants of Naïve Bayes, optimising KNN hyperparameters and theoretical vs practical TC of tree structures.

**Naïve Bayes Comparison**

| Metric | Gaussian | Multinomial | Bernoulli |
| :--- | :--- | :--- | :--- |
| Accuracy | 0.8219 | 0.8958 | 0.6483 |
| Precision | 0.7018 | 0.9464 | 0.8717 |
| Recall | 0.9522 | 0.7794 | 0.1250 |
| F1 | 0.8081 | 0.8548 | 0.2186 |
| ROC-AUC | 0.9166 | 0.9558 | 0.6097 |

**KNN Comparison**

| k | Accuracy | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 0.8828 | 0.8577 | 0.8419 | 0.8497 |
| 3 | 0.8900 | 0.8920 | 0.8199 | 0.8544 |
| 5 | 0.9030 | 0.8988 | 0.8493 | 0.8733 |
| 7 | 0.8929 | 0.8929 | 0.8272 | 0.8588 |
| 9 | 0.8944 | 0.8996 | 0.8235 | 0.8599 |
| 11 | 0.8828 | 0.8996 | 0.7904 | 0.8415 |

**Grid Search vs Randomised Search**

| Parameter | GridSearchCV | RandomisedSearchCV |
| :--- | :--- | :--- |
| Best k | 7 | 5 |
| Weights | Distance | Distance |
| Search Time | 3.58s | 0.27s |

**Learning Outcomes:**
* Understood the theoretical differences and practical performance variations between Gaussian, Multinomial, and Bernoulli Naïve Bayes classifiers on text data.
* Learned to optimize K-Nearest Neighbors hyperparameters such as 'k' and 'weights' using exhaustive Grid Search versus efficient Randomised Search.
* Evaluated different tree-based spatial structures (KDTree vs BallTree) and analyzed their time complexities.
