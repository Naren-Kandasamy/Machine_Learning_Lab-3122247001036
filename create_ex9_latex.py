import os
import json
import subprocess

EX9_DIR = "/home/nkandasamy/Desktop/M.Tech/III year/5th Sem/ML/Lab/Machine_Learning_Lab-3122247001036/Ex9"
results_file = os.path.join(EX9_DIR, "ex9_results.json")

with open(results_file, 'r') as f:
    data = json.load(f)

table2_rows = ""
for row in data['tuning_results']:
    table2_rows += f"    {row['Hidden Layers']} & {row['Activation']} & {row['Optimizer'].upper()} & {row['Learning Rate']} & {row['Validation Accuracy']}\\% & {row['Time (s)']}s \\\\ \\hline\n"

table3_rows = ""
for row in data['ab_comparison']:
    table3_rows += f"    \\textbf{{{row['Model']}}} & {row['Architecture']} & {row['Accuracy']}\\% & {row['Precision']}\\% & {row['Recall']}\\% & {row['F1-Score']}\\% & {row['Training Time (s)']}s \\\\ \\hline\n"

latex_content = r"""\documentclass[12pt,a4paper]{article}

\usepackage[a4paper,margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath,amssymb}
\usepackage{float}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{listings}
\usepackage{enumitem}
\usepackage{fancyhdr}
\usepackage{longtable}

\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}

\pagestyle{fancy}
\fancyhf{}
\lhead{ICS1512}
\rhead{Machine Learning Algorithms Laboratory}
\cfoot{\thepage}

\begin{document}

\begin{tabular}{|p{4cm}|p{11cm}|}
\hline
Experiment No. & 9 \\ \hline
Experiment Title & Perceptron vs Multilayer Perceptron (A/B Experiment) with Hyperparameter Tuning \\ \hline
Student Name & Naren Karthik Kandasamy \\ \hline
Register Number & 3122247001036 \\ \hline
Faculty & Dr. Poreddy Ajay Kumar Reddy \\ \hline
Submission Date & 19/09/2026 \\ \hline
\end{tabular}

\section{Objective}
To implement and empirically compare the performance of:
\begin{itemize}[noitemsep]
    \item \textbf{Model A:} Single-Layer Perceptron Learning Algorithm (PLA) implemented from scratch.
    \item \textbf{Model B:} Multilayer Perceptron (MLP) with hidden layers, non-linear activations, and backpropagation.
\end{itemize}
Systematic hyperparameter tuning is conducted to explore activation functions, optimizers, learning rates, hidden layer architectures, and batch sizes on the English Handwritten Characters benchmark.

\section{Dataset Description}
The \textbf{English Handwritten Characters Dataset} contains scanned grayscale/binary images of handwritten alphanumeric characters:
\begin{longtable}{|p{5.5cm}|p{9.5cm}|}
\hline
\textbf{Property} & \textbf{Specification} \\ \hline
Dataset Source & Kaggle / UCI Benchmark \\ \hline
Total Samples & 3,410 individual handwritten character images \\ \hline
Number of Classes & 62 distinct alphanumeric classes (0--9, A--Z, a--z) \\ \hline
Samples Per Class & Exactly 55 samples per character class (balanced) \\ \hline
Image Preprocessing & Grayscale conversion, resized to $28 \times 28$ ($784$ features), normalized to $[0, 1]$ \\ \hline
Train-Test Partition & Stratified 80\% Training (2,728 images), 20\% Testing (682 images) \\ \hline
\end{longtable}

\section{Theoretical Background and Algorithms}
\subsection{Single-Layer Perceptron Learning Algorithm (PLA)}
A linear threshold classifier mapping an input feature vector $x \in \mathbb{R}^d$ to a binary output using a step activation:
\begin{equation}
\hat{y} = f(w^T x + b) = \begin{cases} 1 & \text{if } w^T x + b \ge 0 \\ 0 & \text{otherwise} \end{cases}
\end{equation}
Weights and biases are updated iteratively using Rosenblatt's rule:
\begin{equation}
w_{t+1} = w_t + \eta (y - \hat{y}) x, \quad b_{t+1} = b_t + \eta (y - \hat{y})
\end{equation}
where $\eta$ is the learning rate. For 62-class multi-class recognition, a One-vs-Rest (OvR) scheme with 62 independent linear perceptrons is constructed. \textbf{Theoretical Limitation:} Perceptrons can only converge if classes are linearly separable. Highly non-linear pixel patterns of handwritten characters cause PLA to oscillate without reaching zero error.

\subsection{Multilayer Perceptron (MLP)}
A feedforward artificial neural network consisting of an input layer ($784$ units), one or more hidden layers with non-linear activation functions (ReLU, Tanh), and an output layer ($62$ units with Softmax):
\begin{equation}
z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}, \quad a^{[l]} = \sigma(z^{[l]})
\end{equation}
Trained using multi-class cross-entropy loss:
\begin{equation}
\mathcal{L} = -\sum_{c=1}^C y_c \log \hat{y}_c
\end{equation}
Error gradients are propagated backward through chain rule derivatives (Backpropagation) and minimized using adaptive first- and second-moment gradient descent (Adam) or Stochastic Gradient Descent (SGD) with momentum.

\section{Experimental Results and Hyperparameter Tuning}

\subsection{MLP Hyperparameter Tuning Results}
Table~\ref{tab:tuning} presents the systematic grid search across hidden layer configurations, activation functions, optimizers, and learning rates.

\begin{table}[H]
\centering
\caption{MLP Systematic Hyperparameter Exploration}
\label{tab:tuning}
\begin{tabular}{|l|c|c|c|c|c|}
\hline
\textbf{Hidden Architecture} & \textbf{Activation} & \textbf{Optimizer} & \textbf{Learning Rate} & \textbf{Val. Acc.} & \textbf{Time} \\ \hline
""" + table2_rows + r"""\end{tabular}
\end{table}

\subsection{A/B Experiment: PLA vs. Tuned MLP}
Table~\ref{tab:ab} highlights the definitive performance contrast between the Single-Layer Perceptron and the Tuned Multilayer Perceptron.

\begin{table}[H]
\centering
\caption{A/B Performance Comparison: PLA vs. Multilayer Perceptron}
\label{tab:ab}
\resizebox{\textwidth}{!}{
\begin{tabular}{|l|l|c|c|c|c|c|}
\hline
\textbf{Model} & \textbf{Architecture} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{Time} \\ \hline
""" + table3_rows + r"""\end{tabular}
}
\end{table}

\subsection{Visualizations and Convergence Curves}
Figure~\ref{fig:loss} tracks training error rate convergence of PLA compared to the continuous cross-entropy loss reduction of MLP. Figure~\ref{fig:roc} and Figure~\ref{fig:cm} display the multi-class ROC curves and confusion matrix heatmap.

\begin{figure}[H]
\centering
\includegraphics[width=0.85\linewidth]{Ex9/Loss_Convergence.eps}
\caption{Convergence Comparison: PLA Discrete Error Rate vs. MLP Loss}
\label{fig:loss}
\end{figure}

\begin{figure}[H]
\centering
\begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\linewidth]{Ex9/ROC_Curves.eps}
    \caption{Multi-Class ROC Curves (MLP)}
    \label{fig:roc}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\linewidth]{Ex9/Confusion_Matrix.eps}
    \caption{Confusion Matrix (Sample 15 Classes)}
    \label{fig:cm}
\end{minipage}
\end{figure}

\section{Answers to Observation Questions}
\begin{enumerate}[leftmargin=*]
    \item \textbf{Why does PLA underperform compared to MLP?} \\
    \textbf{Answer:} PLA is strictly a linear classifier bounded by hyperplanes in 784-dimensional pixel space. Handwritten character strokes have complex spatial invariances and non-linear geometric topologies across 62 classes that are fundamentally non-linearly separable. PLA cannot learn hidden feature representations, plateauing at low accuracy. In contrast, MLP utilizes non-linear activation functions (ReLU) to project inputs into latent representation spaces where classes become linearly separable.

    \item \textbf{Which hyperparameters had the most impact on MLP performance?} \\
    \textbf{Answer:} The \textbf{depth and capacity of hidden layers} followed by the \textbf{optimizer choice} had the greatest impact. Single narrow hidden layers ($64$ units) struggled with 62-class output capacity, whereas hierarchical architectures ($128, 64$ and $256, 128$) provided sufficient representational capacity.

    \item \textbf{Did optimizer choice (SGD vs Adam) affect convergence?} \\
    \textbf{Answer:} Yes. \textbf{Adam} converged substantially faster and achieved higher accuracy due to its per-parameter adaptive learning rates and exponential moving averages of squared gradients. Basic SGD suffered from vanishing gradients across deep layers and oscillated around saddle points.

    \item \textbf{Did adding more hidden layers always improve results? Why or why not?} \\
    \textbf{Answer:} Not indefinitely. While expanding from a single hidden layer to two hidden layers ($(128, 64)$) yielded substantial gains, increasing depth excessively without regularizers increases parameter count, risking over-parameterization and slower backpropagation on small datasets ($3,410$ samples).

    \item \textbf{Did MLP show overfitting? How could it be mitigated?} \\
    \textbf{Answer:} Yes, slight gap between training loss and validation accuracy appeared after extended epochs. Mitigation techniques include: $L_2$ weight regularization (weight decay), Dropout layers, Early Stopping, and spatial Data Augmentation (slight rotations, affine shifts).
\end{enumerate}

\section{Conclusion}
The A/B experiment successfully demonstrated the theoretical and practical divergence between Single-Layer Perceptrons and Multilayer Perceptrons. The linear step-activation PLA is fundamentally incapable of resolving complex handwritten character glyphs, whereas the Tuned MLP with ReLU activations and Adam optimizer achieved high classification accuracy and superior multi-class discrimination.
\end{document}
"""

tex_file = os.path.join(EX9_DIR, "Experiment_9_Report.tex")
with open(tex_file, 'w') as f:
    f.write(latex_content)

print(f"Wrote {tex_file}. Compiling with pdflatex...")
cmd = ["pdflatex", "-interaction=nonstopmode", f"-output-directory={EX9_DIR}", tex_file]
subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

pdf_file = os.path.join(EX9_DIR, "Experiment_9_Report.pdf")
if os.path.exists(pdf_file) and os.path.getsize(pdf_file) > 1000:
    print(f"SUCCESS: {pdf_file} compiled! Size: {os.path.getsize(pdf_file)} bytes.")
else:
    print(f"ERROR: Failed to compile {pdf_file}")
