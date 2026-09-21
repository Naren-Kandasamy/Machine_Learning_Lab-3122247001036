import os
import json
import subprocess

EX8_DIR = "/home/nkandasamy/Desktop/M.Tech/III year/5th Sem/ML/Lab/Machine_Learning_Lab-3122247001036/Ex8"
results_file = os.path.join(EX8_DIR, "ex8_results.json")

with open(results_file, 'r') as f:
    data = json.load(f)

table1_rows = ""
for row in data['table1_elbow']:
    table1_rows += f"    {row['k']} & {row['WCSS']} & {row['Silhouette']} \\\\ \\hline\n"

table2_rows = ""
for row in data['db_tuning']:
    table2_rows += f"    {row['eps']} & {row['min_samples']} & {row['n_clusters']} & {row['n_noise']} & {row['noise_pct']}\\% & {row['silhouette']} \\\\ \\hline\n"

table3_rows = ""
for method, metrics in data['hac_linkages'].items():
    table3_rows += f"    {method.capitalize()} & {metrics['silhouette']} & {metrics['davies_bouldin']} & {metrics['calinski_harabasz']} \\\\ \\hline\n"

table4_rows = ""
for row in data['comparison_metrics']:
    table4_rows += f"    {row['Algorithm']} & {row['Silhouette Score']} & {row['Davies-Bouldin Index']} & {row['Calinski-Harabasz Index']} & {row['Adjusted Rand Index (ARI)']} & {row['Normalized Mutual Info (NMI)']} \\\\ \\hline\n"

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
Experiment No. & 8 \\ \hline
Experiment Title & Clustering Human Activity Recognition Data using K-Means, DBSCAN, and Hierarchical Clustering \\ \hline
Student Name & Naren Karthik Kandasamy \\ \hline
Register Number & 3122247001036 \\ \hline
Faculty & Dr. Poreddy Ajay Kumar Reddy \\ \hline
Submission Date & 19/09/2026 \\ \hline
\end{tabular}

\section{Objective}
To implement and analyze the performance of unsupervised clustering algorithms on the Human Activity Recognition (HAR) dataset:
\begin{itemize}[noitemsep]
    \item \textbf{Model A:} K-Means Clustering with Elbow and Silhouette optimization.
    \item \textbf{Model B:} DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
    \item \textbf{Model C:} Hierarchical Agglomerative Clustering (HAC) exploring diverse linkage criteria.
\end{itemize}
The objective is to evaluate clusters formed by each algorithm using both internal validation metrics and external ground-truth activity comparisons.

\section{Dataset Description}
The \textbf{Human Activity Recognition Using Smartphones} dataset consists of sensory signals recorded from 30 participants (aged 19--48 years) performing six daily activities wearing a Samsung Galaxy S II smartphone on their waist.
\begin{longtable}{|p{5.5cm}|p{9.5cm}|}
\hline
\textbf{Property} & \textbf{Specification} \\ \hline
Dataset Source & UCI Machine Learning Repository \\ \hline
Sensor Modalities & 3-Axis Accelerometer and 3-Axis Gyroscope (50 Hz) \\ \hline
Windowing & 2.56-second sliding windows (128 readings, 50\% overlap) \\ \hline
Number of Samples & 10,299 instances (7,352 train + 2,947 test) \\ \hline
Number of Features & 561 time- and frequency-domain extracted metrics \\ \hline
Ground-Truth Activities (6) & WALKING, WALKING\_UPSTAIRS, WALKING\_DOWNSTAIRS, SITTING, STANDING, LAYING \\ \hline
\end{longtable}

\section{Theory and Methodology}
\subsection{K-Means Clustering}
Partitions $n$ data points into $k$ disjoint clusters $S = \{S_1, \dots, S_k\}$ to minimize Within-Cluster Sum of Squares (WCSS):
\begin{equation}
\text{WCSS} = \sum_{i=1}^k \sum_{x \in S_i} \|x - \mu_i\|^2
\end{equation}
where $\mu_i$ represents the centroid of cluster $S_i$. We determine the optimal $k$ using the \textbf{Elbow Method} (where rate of WCSS reduction diminishes) combined with the \textbf{Silhouette Score}.

\subsection{DBSCAN}
A density-based clustering algorithm characterized by parameters $\epsilon$ (neighborhood radius) and $minPts$ (density threshold). Points are categorized into core points, density-reachable border points, and noise ($ -1 $). DBSCAN discovers arbitrary geometries without pre-specifying $k$ and isolates outlier transitions.

\subsection{Hierarchical Agglomerative Clustering (HAC)}
Constructs a nested bottom-up dendrogram by iteratively merging the pair of clusters minimizing a linkage criterion:
\begin{itemize}[noitemsep]
    \item \textbf{Ward's Method:} Minimizes the total within-cluster variance increase.
    \item \textbf{Complete Linkage:} Maximum pairwise Euclidean distance between cluster points.
    \item \textbf{Average Linkage:} Mean pairwise distance between cluster members.
    \item \textbf{Single Linkage:} Minimum pairwise distance (susceptible to chaining).
\end{itemize}

\section{Experimental Results and Evaluation}

\subsection{K-Means Elbow Method Results}
Table~\ref{tab:elbow} and Figures~\ref{fig:elbow} and~\ref{fig:silhouette} summarize the Elbow Method and Silhouette evaluations across cluster counts $k \in [2, 8]$.

\begin{table}[H]
\centering
\caption{K-Means Elbow Method Results}
\label{tab:elbow}
\begin{tabular}{|c|c|c|}
\hline
\textbf{Number of Clusters ($k$)} & \textbf{WCSS (Inertia)} & \textbf{Silhouette Score} \\ \hline
""" + table1_rows + r"""\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\linewidth]{Ex8/Elbow_Curve.eps}
    \caption{K-Means Elbow Curve (WCSS vs $k$)}
    \label{fig:elbow}
\end{minipage}\hfill
\begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\linewidth]{Ex8/Silhouette_Curve.eps}
    \caption{K-Means Silhouette Score vs $k$}
    \label{fig:silhouette}
\end{minipage}
\end{figure}

\subsection{DBSCAN Parameter Tuning}
Table~\ref{tab:dbscan} documents the exploration across spatial radius $\epsilon$ and $minPts$. At $\epsilon = 15.0$ and $minPts = 15$, DBSCAN achieves a balance with a representative noise threshold.

\begin{table}[H]
\centering
\caption{DBSCAN Parameter Exploration and Noise Detection}
\label{tab:dbscan}
\begin{tabular}{|c|c|c|c|c|c|}
\hline
\textbf{$\epsilon$} & \textbf{$minPts$} & \textbf{Clusters Found} & \textbf{Noise Points} & \textbf{Noise \%} & \textbf{Silhouette} \\ \hline
""" + table2_rows + r"""\end{tabular}
\end{table}

\subsection{Hierarchical Linkage Comparison}
Table~\ref{tab:hac} summarizes internal validation metrics for HAC with $k=6$ across four linkage criteria. Figure~\ref{fig:dendro} illustrates the hierarchical merging tree.

\begin{table}[H]
\centering
\caption{HAC Linkage Methods Comparison ($k=6$)}
\label{tab:hac}
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{Linkage} & \textbf{Silhouette Score} & \textbf{Davies-Bouldin} & \textbf{Calinski-Harabasz} \\ \hline
""" + table3_rows + r"""\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.75\linewidth]{Ex8/Dendrogram.eps}
\caption{Hierarchical Clustering Dendrogram (Ward's Linkage)}
\label{fig:dendro}
\end{figure}

\subsection{Comparative Cluster Visualizations and Final Evaluation}
Figure~\ref{fig:clusters} compares the 2D cluster projections (PCA subspace) alongside ground-truth activities. Table~\ref{tab:final} and Figure~\ref{fig:comp} provide the comparative internal and external validation scores.

\begin{figure}[H]
\centering
\includegraphics[width=0.95\linewidth]{Ex8/Cluster_Scatter.eps}
\caption{2D PCA Visualizations: Ground Truth vs K-Means, DBSCAN, and HAC}
\label{fig:clusters}
\end{figure}

\begin{table}[H]
\centering
\caption{Final Clustering Performance Comparison across Algorithms}
\label{tab:final}
\begin{tabular}{|l|c|c|c|c|c|}
\hline
\textbf{Algorithm} & \textbf{Silhouette} & \textbf{Davies-Bouldin} & \textbf{Calinski-H.} & \textbf{ARI} & \textbf{NMI} \\ \hline
""" + table4_rows + r"""\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.8\linewidth]{Ex8/Clustering_Comparison.eps}
\caption{Performance Comparison Bar Plot Across Clustering Algorithms}
\label{fig:comp}
\end{figure}

\section{Answers to Observation Questions}
\begin{enumerate}[leftmargin=*]
    \item \textbf{Which algorithm produced the most meaningful clusters? Why?} \\
    \textbf{Answer:} \textbf{K-Means ($k=6$)} produced the most semantically meaningful clusters, achieving the highest external alignment metrics ($\text{ARI} = 0.4034$, $\text{NMI} = 0.5582$, and $\text{Calinski-Harabasz} = 741.37$). It successfully separated dynamic kinetic activities (walking, walking upstairs, downstairs) from static postural states (sitting, standing, laying).

    \item \textbf{How sensitive was K-Means to the choice of $k$?} \\
    \textbf{Answer:} Highly sensitive. At $k=2$, the Silhouette score peaks ($0.3815$) by cleanly splitting dynamic movements from stationary postures. Moving towards $k=6$, inertia drops significantly from $963,418$ down to $751,980$, but boundaries between postural activities (sitting vs standing) exhibit overlapping feature densities.

    \item \textbf{Did DBSCAN detect noise or small clusters effectively?} \\
    \textbf{Answer:} Yes. In high-dimensional spaces ($561$ dimensions), DBSCAN suffers from the curse of dimensionality where Euclidean distances become uniform. However, tuned with $\epsilon=15.0$ and $minPts=15$, it effectively filtered outlier transitions ($34.57\%$ noise), capturing dense core movement profiles.

    \item \textbf{How does linkage choice (single/complete/ward) affect hierarchical clustering?} \\
    \textbf{Answer:} Ward's linkage proved optimal for clustering accuracy by minimizing within-cluster variance ($\text{CH} = 705.32$, $\text{ARI} = 0.2913$). In contrast, Single linkage suffered from chaining artifacts (merging outliers into one giant cluster with artificially high silhouette but near-zero Calinski-Harabasz of $11.81$).

    \item \textbf{Which internal metric best matched your visual intuition of cluster quality?} \\
    \textbf{Answer:} The \textbf{Calinski-Harabasz Index} and \textbf{Davies-Bouldin Index} best matched visual intuition on 2D PCA plots. While silhouette scores favor elongated densities, Calinski-Harabasz rewards well-separated, compact spherical clusters corresponding to distinct kinetic behaviors.
\end{enumerate}

\section{Conclusion}
We successfully implemented and benchmarked K-Means, DBSCAN, and Hierarchical Agglomerative Clustering on the Human Activity Recognition dataset. The experiments validate that sensor-based activity spaces possess natural geometric clusters reflecting physical movements. K-Means with $k=6$ and Ward's HAC yielded the most accurate groupings aligning with ground-truth labels.
\end{document}
"""

tex_file = os.path.join(EX8_DIR, "Experiment_8_Report.tex")
with open(tex_file, 'w') as f:
    f.write(latex_content)

print(f"Wrote {tex_file}. Compiling with pdflatex...")
cmd = ["pdflatex", "-interaction=nonstopmode", f"-output-directory={EX8_DIR}", tex_file]
subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

pdf_file = os.path.join(EX8_DIR, "Experiment_8_Report.pdf")
if os.path.exists(pdf_file) and os.path.getsize(pdf_file) > 1000:
    print(f"SUCCESS: {pdf_file} compiled! Size: {os.path.getsize(pdf_file)} bytes.")
else:
    print(f"ERROR: Failed to compile {pdf_file}")
