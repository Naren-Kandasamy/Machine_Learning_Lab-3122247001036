# Experiment 3
**Regression and Regularization on Loan Approval Dataset**

**Aim:**
To implement Linear Regression to predict loan amounts and evaluate the impact of L1, L2, and ElasticNet regularization techniques. Analyze how regularization parameters affect coefficient shrinkage and model generalization.

**Linear Regression Baseline**

| Metric | Train | Test |
| :--- | :--- | :--- |
| MAE | 0.2843 | 0.2789 |
| RMSE | 0.3729 | 0.3696 |
| R2 Score | 0.8593 | 0.8736 |

**Regularized Models Comparison**

| Model | Best Parameters | CV R2 Mean | Test R2 Score |
| :--- | :--- | :--- | :--- |
| Linear Regression | None | 0.8578 | 0.8736 |
| Ridge (L2) | alpha = 0.1 | 0.8578 | 0.8736 |
| Lasso (L1) | alpha = 0.01 | 0.8580 | 0.8724 |
| ElasticNet | alpha = 0.01, l1_ratio = 0.8 | 0.8578 | 0.8727 |

**Learning Outcomes:**
* Understood how to model relationships between continuous dependent variables and independent features using Multiple Linear Regression.
* Learned how to mitigate multicollinearity and prevent model overfitting using Ridge and Lasso regularization.
* Observed the theoretical and practical effect of varying alpha parameters on shrinking feature coefficients toward zero.
