# Experiment 8
**Clustering Human Activity Recognition Data using K-Means, DBSCAN, and Hierarchical Clustering**

**Aim:**
To implement and analyze the performance of clustering algorithms (K-Means, DBSCAN, and Hierarchical Agglomerative Clustering) on the Human Activity Recognition dataset and compare formed clusters against ground-truth activities.

**Dataset Characteristics:**
| Dataset | Number of Samples | Number of Features | Number of Classes | Sensory Modalities |
| :--- | :--- | :--- | :--- | :--- |
| UCI Human Activity Recognition (HAR) | 10,299 | 561 | 6 Activities | 3-axis Accelerometer & Gyroscope (50 Hz) |

---

### Table 1: K-Means Elbow Method Results
| Number of Clusters ($k$) | WCSS (Inertia) | Silhouette Score |
| :---: | :---: | :---: |
| 2 | 963418.29 | 0.3815 |
| 3 | 851205.12 | 0.3042 |
| 4 | 812358.47 | 0.2460 |
| 5 | 774596.03 | 0.1194 |
| 6 | 751980.97 | 0.1198 |
| 7 | 730037.78 | 0.0969 |
| 8 | 716024.77 | 0.0759 |

**Selection and Justification of Best $k$:**
- **Mathematical Elbow:** The reduction in WCSS slows down noticeably after $k=2$ and $k=6$.
- **Silhouette Analysis:** $k=2$ captures the major topological division between active dynamic movements (walking) and passive stationary states (laying, sitting, standing).
- **Domain Justification:** $k=6$ matches the exact physical activity classes defined in the ground-truth annotations with competitive internal compactness.

---

### Table 2: Clustering Performance Comparison Across Algorithms
| Algorithm | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index | Adjusted Rand Index (ARI) | Normalized Mutual Info (NMI) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **K-Means ($k=6$)** | 0.1198 | 2.3170 | 741.37 | 0.4034 | 0.5582 |
| **DBSCAN ($\epsilon=15, minPts=15$)** | 0.3333 | 1.4767 | 737.52 | 0.2840 | 0.4306 |
| **Hierarchical (Ward's, $k=6$)** | 0.1279 | 2.6018 | 705.32 | 0.2913 | 0.4614 |

---

### Hierarchical Clustering Linkage Comparison ($k=6$)
| Linkage Method | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index |
| :--- | :---: | :---: | :---: |
| **Ward** | 0.1279 | 2.6018 | 705.32 |
| **Complete** | 0.3061 | 1.8475 | 605.26 |
| **Average** | 0.3368 | 1.2790 | 55.59 |
| **Single** | 0.5310 | 0.2969 | 11.81 |

---

### Observation Questions & Answers:
1. **Which algorithm produced the most meaningful clusters? Why?**
   - **Answer:** K-Means ($k=6$) produced the most meaningful clusters with highest ARI ($0.4034$) and NMI ($0.5582$). It cleanly divided dynamic activities from static postural states while maintaining high between-cluster separation.
2. **How sensitive was K-Means to the choice of $k$?**
   - **Answer:** Highly sensitive. At $k=2$, the silhouette score peaks at $0.3815$ separating dynamic from static behaviors. Beyond $k=6$, inertia decreases marginally while silhouette degrades to $<0.08$.
3. **Did DBSCAN detect noise or small clusters effectively?**
   - **Answer:** Yes, tuned with $\epsilon=15.0$ and $minPts=15$, DBSCAN effectively filtered transition outliers ($34.57\%$ noise points) while isolating dominant core movement clusters.
4. **How does linkage choice affect hierarchical clustering?**
   - **Answer:** Ward's linkage produces balanced, compact clusters minimizing variance ($\text{CH} = 705.32$). Single linkage suffers severely from chaining, leaving solitary points attached to a giant cluster.
5. **Which internal metric best matched your visual intuition of cluster quality?**
   - **Answer:** Calinski-Harabasz Index and Davies-Bouldin Index best matched visual inspection of 2D PCA cluster projections, accurately reflecting geometric compactness and separation.

---

### Result:
The clustering algorithms were successfully implemented and evaluated on the Human Activity Recognition dataset. K-Means demonstrated superior correspondence with ground-truth labels ($\text{NMI} = 0.5582$), while DBSCAN effectively identified transitional noise points.
