# Experiment 9
**Perceptron vs Multilayer Perceptron (A/B Experiment) with Hyperparameter Tuning**

**Aim:**
To implement and empirically compare the classification performance of a Single-Layer Perceptron (PLA) against a Multilayer Perceptron (MLP) with hyperparameter tuning on the English Handwritten Characters dataset.

**Dataset Characteristics:**
| Dataset | Total Samples | Classes | Samples Per Class | Image Dimensions | Train-Test Split |
| :--- | :--- | :--- | :--- | :--- | :--- |
| English Handwritten Characters | 3,410 | 62 (0–9, A–Z, a–z) | 55 (Balanced) | $28 \times 28$ Grayscale (784 features) | 80% Train (2,728), 20% Test (682) |

---

### Table 1: MLP Systematic Hyperparameter Exploration
| Hidden Architecture | Activation | Optimizer | Learning Rate | Validation Accuracy | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| $(64,)$ | ReLU | ADAM | 0.001 | 1.47% | 2.05s |
| $(128, 64)$ | ReLU | ADAM | 0.001 | 29.85% | 9.83s |
| $(128, 64)$ | Tanh | ADAM | 0.001 | 29.12% | 7.72s |
| $(128, 64)$ | ReLU | SGD | 0.010 | 17.95% | 5.56s |
| **$(256, 128)$** | **ReLU** | **ADAM** | **0.001** | **34.80%** | **15.77s** |

---

### Table 2: A/B Performance Comparison (PLA vs Tuned MLP)
| Model | Architecture | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Training Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Single-Layer Perceptron (PLA)** | Single Layer (OvR 62 Step Units) | 12.61% | 26.73% | 12.61% | 11.53% | 15.09s |
| **Tuned Multilayer Perceptron (MLP)** | Input(784) $\rightarrow$ Dense(256, ReLU) $\rightarrow$ Dense(128, ReLU) $\rightarrow$ Softmax(62) | **41.35%** | **43.98%** | **41.35%** | **40.65%** | 20.26s |

---

### Observation Questions & Answers:
1. **Why does PLA underperform compared to MLP?**
   - **Answer:** PLA is strictly a linear decision machine constrained to linear hyperplanes in 784-dimensional space. Complex handwritten character glyphs across 62 categories exhibit high intra-class variance and non-linear boundaries. MLP utilizes non-linear activation functions (ReLU) and multi-level hidden layers to project inputs into latent manifolds where non-linear patterns become separable.

2. **Which hyperparameters had the most impact on MLP performance?**
   - **Answer:** The **hidden layer architecture (depth and width)** and **optimizer choice** had the most pronounced effect. Single-layer narrow models ($64$ units) suffered from severe underfitting on 62 classes, while expanding to $(256, 128)$ allowed sufficient capacity to distinguish delicate character strokes.

3. **Did optimizer choice (SGD vs Adam) affect convergence?**
   - **Answer:** Yes. **Adam** converged far faster and reached almost double the validation accuracy ($29.85\%$) compared to basic SGD ($17.95\%$) over the same epoch budget, thanks to adaptive per-parameter learning rate scaling and momentum estimation.

4. **Did adding more hidden layers always improve results? Why or why not?**
   - **Answer:** Adding depth improves representational power up to a point where capacity matches dataset complexity. Moving from $(64,)$ to $(128, 64)$ and $(256, 128)$ produced major gains, but further depth on 3,410 samples increases parameter count and risks vanishing gradients without batch normalization or residual connections.

5. **Did MLP show overfitting? How could it be mitigated?**
   - **Answer:** Yes, training loss consistently decreased while validation plateaued around epoch 45. Overfitting can be mitigated using Dropout ($p=0.2–0.5$), $L_2$ weight decay (regularization), Early Stopping, and data augmentation (affine rotations, shears).

---

### Result:
The A/B experiment demonstrated that the Single-Layer Perceptron is fundamentally limited by linear separability, whereas the Multilayer Perceptron with backpropagation, ReLU activations, and Adam optimization successfully captured non-linear character patterns, achieving over $3\times$ higher classification accuracy.
