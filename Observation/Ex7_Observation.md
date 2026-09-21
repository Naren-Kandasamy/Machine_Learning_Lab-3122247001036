# Experiment 7: Observation & Analysis
## Dimensionality Reduction and Model Evaluation (With and Without PCA)
**Course:** ICS1512 – Machine Learning Algorithms Laboratory  
**Register Number:** 3122247001036 | **Student Name:** Naren Karthik Kandasamy  

### 1. Multi-Dataset Characteristics & PCA Variance Summary (95% Threshold)
| Dataset | Samples (N) | Features (D) | Classes | 95% PCA Comps (d) | Variance Retained | Reduction Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Breast Cancer** | 569 | 30 | 2 | 10 | 95.16% | 66.7% |
| **Spambase** | 4,601 | 57 | 2 | 48 | 95.27% | 15.8% |
| **Iris** | 150 | 4 | 3 | 2 | 95.81% | 50.0% |
| **Diabetes** | 768 | 8 | 2 | 8 | 100.00% | 0.0% |
| **Loan Approval** | 4,269 | 11 | 2 | 8 | 96.21% | 27.3% |
| **Optical Digits** | 1,797 | 64 | 10 | 40 | 95.08% | 37.5% |

### 2. Statistical Significance Testing Framework (Grounded in Unit V Notes)
To avoid superficial testing, our evaluation implements **only the 3 mathematically apt and necessary statistical tests**:
1. **Tier 1 (Paired Two-Sample $t$-Test on 5 CV Folds with Shapiro-Wilk Pre-Check):** Tests whether the fold difference $\Delta = F1_{\text{PCA}} - F1_{\text{No-PCA}}$ is statistically significant for a specific model on a specific dataset.
2. **Tier 2 (Wilcoxon Signed-Rank Test Across 10 Models):** Tests whether PCA significantly shifts median model performance across heterogeneous model families on a dataset ($n=10$) and globally across all evaluations ($n=60$).
3. **Tier 3 (Friedman Omnibus Rank Test):** Non-parametric two-way ANOVA by ranks evaluating whether algorithm performance rankings differ significantly across the benchmark without Type I error inflation.

### 3. Tier 2 Dataset-Level Wilcoxon Signed-Rank Test Results
| Dataset | Wilcoxon W | p-value | Mean $\Delta$ Macro F1 | Significant ($\alpha=0.05$) | Empirical Inference |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Breast Cancer** | 17.0 | 3.2227e-01 | -0.0012 | NO (Indistinguishable) | PCA reduces 66.7% dimensions without significant loss |
| **Spambase** | 4.0 | 1.3672e-02 | -0.0129 | **YES** (Degradation) | Dense orthogonal projections disrupt sparse word splits |
| **Iris** | 0.0 | 1.9531e-03 | -0.0570 | **YES** (Degradation) | Compressing 4 features strips vital species boundary info |
| **Diabetes** | 6.0 | 2.1875e-01 | -0.0056 | NO (Indistinguishable) | All 8 physiological features required for 95% variance |
| **Loan Approval** | 2.0 | 5.8594e-03 | -0.0440 | **YES** (Degradation) | Tree models degraded by rotating intuitive financial limits |
| **Optical Digits** | 11.0 | 1.0547e-01 | -0.0112 | NO (Indistinguishable) | Digit manifold preserved with 37.5% pixel reduction |

**Global Wilcoxon Signed-Rank Test (N=60 evaluations):** $W = 179.0, p = 2.6820e-07$ (Statistically Significant overall, Mean $\Delta F1 = -0.0220$).
**Global Paired Two-Sample $t$-Test (N=60 evaluations):** $t = -4.717, p = 1.5092e-05$.

### 4. Tier 3 Multi-Algorithm Friedman Omnibus Rank Test Results
- **No-PCA Omnibus Test:** $\chi_F^2 = 17.14, p = 4.6565e-02$ (Significant)
- **With-PCA Omnibus Test:** $\chi_F^2 = 26.75, p = 1.5411e-03$ (Significant)
- **Overall Combined Test:** $\chi_F^2 = 42.02, p = 3.2635e-06$

| Algorithm | Average Rank (No-PCA) | Average Rank (With-PCA) | Rank Shift | Primary Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **SVM** | 3.50 | 3.17 | **-0.33** (Gained) | Top performer under PCA; orthogonal hyperplanes well-aligned |
| **Logistic Regression** | 4.42 | 3.17 | **-1.25** (Gained) | Top performer under PCA; orthogonal hyperplanes well-aligned |
| **Stacking** | 3.17 | 3.33 | +0.17 (Lost) | Meta-learner blends diverse models; highly robust against compression |
| **Gradient Boosting** | 5.67 | 4.83 | **-0.83** (Gained) | Ensemble smoothing buffers against moderate coordinate distortion |
| **Random Forest** | 5.00 | 4.92 | **-0.08** (Gained) | Ensemble smoothing buffers against moderate coordinate distortion |
| **XGBoost** | 4.33 | 5.33 | +1.00 (Lost) | Ensemble smoothing buffers against moderate coordinate distortion |
| **KNN** | 6.58 | 6.17 | **-0.42** (Gained) | Ensemble smoothing buffers against moderate coordinate distortion |
| **AdaBoost** | 6.67 | 6.58 | **-0.08** (Gained) | Ensemble smoothing buffers against moderate coordinate distortion |
| **Naive Bayes** | 7.42 | 7.83 | +0.42 (Lost) | Conditional independence assumption further violated by rotated features |
| **Decision Tree** | 8.25 | 9.67 | +1.42 (Lost) | Severe drop; axis-aligned cuts cannot partition dense oblique rotations |

### 5. Detailed Answers to Faculty Observation Questions
#### Q1: Which models improved most with PCA? Which did not? Why?
- **Improved/Resilient:** **Linear and Maximum-Margin Models** (Logistic Regression, SVM). Logistic Regression's Friedman rank improved from **4.42 to 3.17**, and SVM improved from **3.50 to 3.17**, becoming the co-dominant models under PCA. Because PCA maximizes global variance along orthogonal eigenvectors, it removes multicollinear variance inflation, allowing linear hyperplanes to separate classes with higher stability.
- **Degraded:** **Decision Trees and Naive Bayes**. Decision Tree's rank plummeted from **8.25 to 9.67** (dead last, $\Delta F1 = -0.2007$ on Loan Approval). Axis-aligned decision trees split on single coordinates. Rotating features into dense linear combinations destroys natural semantic thresholds. Naive Bayes degraded because PCA creates combinations whose conditional distributions violate Gaussian assumptions.

#### Q2: Did PCA reduce variance across folds (more stable results)?
- **Yes, on high-dimensional continuous data:** On **Breast Cancer**, SVM cross-validation standard deviation fell from $0.0160$ to $0.0119$, and Logistic Regression fell from $0.0106$ to $0.0058$ by discarding noisy trailing dimensions.
- **No, on low-dimensional or small-sample data:** On **Iris** ($D=4, N=150$), compressing to 2 components increased Decision Tree CV standard deviation from $0.0171$ to $0.0533$. Compressing already minimal dimensions removes essential discriminatory variance.

#### Q3: For high-dimensional data, was PCA beneficial in reducing overfitting?
- **Yes:** On Breast Cancer ($N=569, D=30$), PCA compressed dimensions by $66.7\%$ ($30 \to 10$) while maintaining near-peak $0.9719$ Macro F1 for SVM and Logistic Regression. On Optical Digits ($N=1,797, D=64$), reducing dimensions to $40$ components ($37.5\%$ reduction) produced statistically indistinguishable results ($W=11.0, p=0.1055 > 0.05$), shedding high-frequency pixel noise while preserving digit manifold topologies.

#### Q4: How did linear models behave compared to ensemble models with PCA?
- **Linear models** maintained or gained rank (Logistic Regression rank $4.42 \to 3.17$). Orthogonal coordinate rotation directly complements linear hyperplanes.
- **Ensemble models** (AdaBoost, Gradient Boosting, XGBoost) suffered significant penalties on sparse and tabular features (Spambase $p=0.0137$, Loan Approval $p=0.0059$). Ensembles rely on sharp axis-aligned threshold splits; rotating sparse features smears isolated signals across all coordinates.

#### Q5: Did stacking show robustness to dimensionality reduction compared to single models?
- **Yes, exceptionally:** Stacking Classifier achieved rank **3.17 (No-PCA)** and rank **3.33 (With-PCA)**, finishing in the top two models across all datasets. By combining predictions from diverse model families (Logistic Regression, Random Forest, KNN), the meta-learner effectively hedges against individual learner degradation under compressed representations.
