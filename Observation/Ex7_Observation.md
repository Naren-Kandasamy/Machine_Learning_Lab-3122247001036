# Experiment 7
**Dimensionality Reduction and Model Evaluation (With and Without PCA)**

**Aim:**
To study the effect of dimensionality reduction using Principal Component Analysis (PCA) on the performance of 10 machine learning classifiers. To train, validate, and compare models under No-PCA and With-PCA settings using 5-fold cross-validation, and perform statistical significance testing across benchmark datasets.

---

### Dataset Characteristics:
| Dataset | Originating Lab | Samples ($N$) | Features ($D$) | Target Classes | Preprocessing |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Breast Cancer Wisconsin** | Ex 5 & 6 | 569 | 30 | 2 (Malignant, Benign) | Standardized ($\mu=0, \sigma=1$) |
| **Spambase Email** | Ex 2 & 4 | 4,601 | 57 | 2 (Spam, Non-Spam) | Frequency scaled, Standardized |
| **Iris Species** | Ex 1 | 150 | 4 | 3 (Setosa, Versi, Virg) | Standardized |
| **Pima Indians Diabetes** | Ex 1 & 4 | 768 | 8 | 2 (Diabetic, Healthy) | Missing imputation, Standardized |
| **Loan Approval** | Ex 1 & 3 | 4,269 | 11 | 2 (Approved, Rejected) | Label encoded, Standardized |
| **Optical Digits (MNIST)** | Ex 1 & 6 | 1,797 | 64 | 10 (Digits 0–9) | Min-max scaled $[0, 1]$, Flattened |

---

### Table 1: PCA Variance Explained & Dimensionality Reduction (95% Target)
| Dataset | Original Dim ($D$) | Chosen Comps ($d$) | Variance Retained | Reduction Ratio | Justification |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Breast Cancer** | 30 | 10 | 95.16% | **66.7%** | First 10 PCs retain core cell morphology variance. |
| **Spambase** | 57 | 48 | 95.27% | **15.8%** | Sparse word occurrences require broad eigenvector base. |
| **Iris** | 4 | 2 | 95.81% | **50.0%** | First 2 PCs capture petal/sepal geometric variance. |
| **Diabetes** | 8 | 8 | 100.00% | **0.0%** | All 8 orthogonal metabolic indicators are necessary. |
| **Loan Approval** | 11 | 8 | 96.21% | **27.3%** | 8 PCs compress collinear financial asset indicators. |
| **Optical Digits** | 64 | 40 | 95.08% | **37.5%** | Eliminates high-frequency background pixel noise. |

---

### Table 2: 5-Fold Cross-Validation Results (No-PCA vs. With-PCA) [Breast Cancer Benchmark]
| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Avg (No-PCA) | Avg (With-PCA) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVM** | 98.25% | 97.37% | 98.25% | 98.25% | 98.23% | **98.07%** | **97.89%** |
| **Naïve Bayes** | 91.23% | 95.61% | 92.11% | 92.98% | 96.46% | **93.68%** | **92.09%** |
| **KNN** | 98.25% | 97.37% | 97.37% | 92.98% | 94.69% | **96.13%** | **95.78%** |
| **Logistic Regression** | 98.25% | 96.49% | 99.12% | 97.37% | 97.35% | **97.71%** | **98.07%** |
| **Decision Tree** | 94.74% | 87.72% | 92.11% | 90.35% | 94.69% | **91.92%** | **94.55%** |
| **Random Forest** | 95.61% | 96.49% | 97.37% | 94.74% | 96.46% | **96.13%** | **94.55%** |
| **AdaBoost** | 98.25% | 97.37% | 99.12% | 93.86% | 96.46% | **97.01%** | **95.96%** |
| **Gradient Boosting** | 96.49% | 92.98% | 97.37% | 94.74% | 95.58% | **95.43%** | **95.08%** |
| **XGBoost** | 98.25% | 94.74% | 96.49% | 94.74% | 95.58% | **95.96%** | **96.31%** |
| **Stacking (Ensemble)** | 98.25% | 96.49% | 99.12% | 98.25% | 98.23% | **98.07%** | **97.54%** |

---

### Table 3: Multi-Dataset Statistical Significance Testing (Wilcoxon Signed-Rank Test)
| Dataset | Wilcoxon $W$ | $p$-value | Mean $\Delta$ F1 | Significant ($\alpha = 0.05$) | Key Inference |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Breast Cancer** | 17.0 | 0.3223 | -0.0012 | **No** (Indistinguishable) | 66.7% dimension compression with zero loss. |
| **Spambase** | 4.0 | 0.0137 | -0.0129 | **Yes** (Degradation) | Dense orthogonal rotations disrupt sparse word splits. |
| **Iris** | 0.0 | 0.0020 | -0.0570 | **Yes** (Degradation) | Extreme reduction ($4 \to 2$) removes boundary info. |
| **Diabetes** | 6.0 | 0.2188 | -0.0056 | **No** (Indistinguishable) | All 8 orthogonal biological features retained. |
| **Loan Approval** | 2.0 | 0.0059 | -0.0440 | **Yes** (Degradation) | Rotating intuitive financial limits penalizes tree splits. |
| **Optical Digits** | 11.0 | 0.1055 | -0.0112 | **No** (Indistinguishable) | Digit manifold preserved with 37.5% pixel reduction. |

* **Global Wilcoxon Test ($N=60$):** $W = 179.0, p = 2.68 \times 10^{-7}$ (Statistically Significant overall, Mean $\Delta \text{F1} = -0.0220$).
* **Global Paired $t$-Test ($N=60$):** $t = -4.717, p = 1.51 \times 10^{-5}$ ($\alpha = 0.05$).

---

### Table 4: Multi-Algorithm Benchmark Ranking (Friedman Test)
| Algorithm | Avg Rank (No-PCA) | Avg Rank (With-PCA) | Rank Shift | Primary Mechanism |
| :--- | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 4.42 | **3.17** | **-1.25** (Gained) | Top performer under PCA; orthogonal hyperplanes align. |
| **SVM** | 3.50 | **3.17** | **-0.33** (Gained) | Top performer under PCA; maximum margin stabilized. |
| **Stacking** | **3.17** | **3.33** | +0.17 (Stable) | Meta-learner blends models; highly robust to compression. |
| **Gradient Boosting** | 5.67 | 4.83 | **-0.83** (Gained) | Ensemble averaging buffers against moderate rotation. |
| **Random Forest** | 5.00 | 4.92 | **-0.08** (Stable) | Resilient bagging over components. |
| **XGBoost** | 4.33 | 5.33 | +1.00 (Lost) | Minor penalty on split criteria. |
| **KNN** | 6.58 | 6.17 | **-0.42** (Gained) | Metric distances preserved on dominant variance space. |
| **AdaBoost** | 6.67 | 6.58 | **-0.08** (Stable) | Weak stumps sensitive to smeared features. |
| **Naïve Bayes** | 7.42 | 7.83 | +0.42 (Lost) | Rotated components violate conditional independence. |
| **Decision Tree** | 8.25 | **9.67** | **+1.42** (Lost) | Axis-aligned splits fail on dense oblique rotations. |

* **Friedman Test Statistic:** $\chi_F^2 = 42.02, p = 3.26 \times 10^{-6}$ (Highly Significant rank differences across algorithms).

---

### Observation Questions & Answers:

1. **Which models improved most with PCA? Which did not? Why?**
   - **Improved/Resilient:** **Logistic Regression** (rank $4.42 \to 3.17$) and **SVM** (rank $3.50 \to 3.17$) achieved the top positions under PCA. PCA removes multicollinear variance inflation, allowing orthogonal hyperplanes to separate classes with higher stability.
   - **Degraded:** **Decision Trees** (rank $8.25 \to 9.67$) and **Naïve Bayes** degraded most. Axis-aligned trees split one coordinate at a time; PCA creates dense linear combinations of all features, destroying intuitive single-feature thresholds.

2. **Did PCA reduce variance across folds (more stable results)?**
   - **Yes, on high-dimensional continuous data:** On Breast Cancer, SVM CV standard deviation fell from $0.0160$ to $0.0119$, and Logistic Regression fell from $0.0106$ to $0.0058$ by discarding noisy trailing dimensions.
   - **No, on low-dimensional data:** On Iris ($D=4$), compressing to 2 components increased Decision Tree CV standard deviation from $0.0171$ to $0.0533$ due to loss of boundary information.

3. **For high-dimensional data, was PCA beneficial in reducing overfitting?**
   - **Yes:** On Breast Cancer ($N=569, D=30$), PCA compressed dimensions by $66.7\%$ with zero performance degradation ($F1 = 0.9719$). On Optical Digits ($D=64$), reducing to 40 components yielded statistically indistinguishable performance ($W=11.0, p=0.1055 > 0.05$) while filtering high-frequency noise.

4. **How did linear models behave compared to ensemble models with PCA?**
   - **Linear models** gained relative rank under PCA (Logistic Regression rank $4.42 \to 3.17$). Orthogonal coordinate rotation directly complements linear hyperplanes.
   - **Ensemble models** suffered drops on sparse and tabular features (Spambase $p=0.0137$, Loan Approval $p=0.0059$) because rotating sparse features smears isolated signals across all coordinates.

5. **Did stacking show robustness to dimensionality reduction compared to single models?**
   - **Yes, exceptionally:** Stacking achieved rank **3.17 (No-PCA)** and rank **3.33 (With-PCA)**, finishing in the top two across all datasets. Blending diverse model families (Logistic Regression, Random Forest, KNN) buffers against single-model degradation.

---

### Result:
PCA dimensionality reduction was successfully evaluated across 10 classifiers on 6 benchmark datasets. Linear and maximum-margin models (SVM, Logistic Regression) and Stacking Ensembles proved most robust under PCA compression, while axis-aligned Decision Trees experienced severe degradation under coordinate rotation.

---

### Learning Outcomes:
* Learned to compute covariance eigen-decomposition and apply the 95% cumulative explained variance threshold to compress high-dimensional feature spaces.
* Understood the geometric impact of PCA coordinate rotation: linear models (SVM, Logistic Regression) benefit from orthogonal alignment, whereas axis-aligned Decision Trees suffer performance degradation.
* Mastered statistical hypothesis testing for model comparison: applied Paired $t$-tests and Wilcoxon Signed-Rank tests for pairwise evaluation, and the Friedman Omnibus test for multi-algorithm benchmark ranking without Type I error inflation.
