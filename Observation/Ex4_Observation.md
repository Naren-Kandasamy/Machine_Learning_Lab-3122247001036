# Experiment 4
**Spam Classification using Logistic Regression and SVM**

**Aim:**
To classify spam emails using Logistic Regression and Support Vector Machines (SVM). Evaluate the impact of different SVM kernels on classification boundaries and optimize hyperparameters using randomized search.

**Logistic Regression Performance**

| Metric | Value |
| :--- | :--- |
| Accuracy | 0.9276 |
| Precision | 0.9405 |
| Recall | 0.8713 |
| F1 Score | 0.9046 |

**SVM Kernel Comparison**

| Kernel | Accuracy | F1 Score | Training Time (s) |
| :--- | :--- | :--- | :--- |
| Linear | 0.9276 | 0.9053 | 0.2618 |
| Poly | 0.8437 | 0.7632 | 0.2041 |
| RBF | 0.9204 | 0.8952 | 0.1541 |
| Sigmoid | 0.8726 | 0.8352 | 0.1695 |

**Hyperparameter Tuning & Cross Validation**

| Model | Best Parameters | CV Accuracy | CV F1 Score |
| :--- | :--- | :--- | :--- |
| Logistic Regression | solver: liblinear, penalty: l1, C: 100 | 0.9268 | 0.9055 |
| Support Vector Machine | kernel: linear, gamma: scale, C: 10 | 0.9283 | 0.9079 |

**Learning Outcomes:**
* Understood how Logistic Regression utilizes a sigmoid function to map linear predictions into class probabilities for binary classification tasks.
* Explored how Support Vector Machines construct hyperplanes to maximize the margin of separation between classes.
* Learned how different kernel functions (Linear vs Non-linear like RBF/Poly) transform feature spaces to handle complex, non-linearly separable data distributions.
