import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
)
from scipy.cluster.hierarchy import dendrogram, linkage

# Styling for academic publication quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 13,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.titlesize': 15
})

EX8_DIR = "/home/nkandasamy/Desktop/M.Tech/III year/5th Sem/ML/Lab/Machine_Learning_Lab-3122247001036/Ex8"
DATA_DIR = "/home/nkandasamy/Desktop/M.Tech/III year/5th Sem/ML/Lab/Machine_Learning_Lab-3122247001036/Datasets/UCI_HAR"
os.makedirs(EX8_DIR, exist_ok=True)

def load_har_data(sample_size=3000, random_state=42):
    print("Loading UCI HAR Dataset...")
    # Load feature names
    features_path = os.path.join(DATA_DIR, "features.txt")
    with open(features_path, 'r') as f:
        features = [line.strip().split(None, 1)[1] for line in f if line.strip()]
    
    # Load activity labels
    activity_path = os.path.join(DATA_DIR, "activity_labels.txt")
    activity_map = {}
    with open(activity_path, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                activity_map[int(parts[0])] = parts[1]
                
    # Load train data
    X_train_path = os.path.join(DATA_DIR, "train", "X_train.txt")
    y_train_path = os.path.join(DATA_DIR, "train", "y_train.txt")
    X_test_path = os.path.join(DATA_DIR, "test", "X_test.txt")
    y_test_path = os.path.join(DATA_DIR, "test", "y_test.txt")
    
    X_tr = np.loadtxt(X_train_path)
    y_tr = np.loadtxt(y_train_path, dtype=int)
    X_te = np.loadtxt(X_test_path)
    y_te = np.loadtxt(y_test_path, dtype=int)
    
    X = np.vstack((X_tr, X_te))
    y = np.concatenate((y_tr, y_te))
    
    print(f"Full HAR dataset: {X.shape[0]} samples, {X.shape[1]} features, 6 activities.")
    
    # Stratified sampling for efficient distance-matrix computation (HAC/DBSCAN)
    if sample_size and sample_size < len(y):
        np.random.seed(random_state)
        indices = []
        samples_per_class = sample_size // len(np.unique(y))
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            chosen = np.random.choice(cls_idx, size=min(samples_per_class, len(cls_idx)), replace=False)
            indices.extend(chosen)
        indices = np.array(indices)
        X = X[indices]
        y = y[indices]
        print(f"Stratified sample for clustering analysis: {X.shape[0]} samples.")
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, activity_map, features

def run_kmeans_elbow(X, k_range=range(2, 9)):
    print("\n--- Running K-Means Elbow & Silhouette Analysis ---")
    wcss = []
    sil_scores = []
    
    table1_data = []
    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        cluster_labels = km.fit_predict(X)
        inertia = km.inertia_
        sil = silhouette_score(X, cluster_labels)
        wcss.append(inertia)
        sil_scores.append(sil)
        table1_data.append({
            'k': k,
            'WCSS': round(inertia, 2),
            'Silhouette': round(sil, 4)
        })
        print(f"k={k}: WCSS={inertia:.2f}, Silhouette={sil:.4f}")
        
    # Plot Elbow Curve
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), wcss, marker='o', color='#1f77b4', linewidth=2.5, markersize=8)
    plt.title('K-Means Elbow Method (WCSS vs k)', pad=12)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS (Inertia)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    elbow_path = os.path.join(EX8_DIR, 'Elbow_Curve.eps')
    plt.savefig(elbow_path, format='eps', dpi=300)
    plt.close()
    
    # Plot Silhouette Curve
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), sil_scores, marker='s', color='#2ca02c', linewidth=2.5, markersize=8)
    plt.title('K-Means Silhouette Score vs k', pad=12)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    sil_path = os.path.join(EX8_DIR, 'Silhouette_Curve.eps')
    plt.savefig(sil_path, format='eps', dpi=300)
    plt.close()
    
    # Choose optimal k: k=6 (ground truth activity count) and k=2 (highest silhouette separation between active/passive)
    best_k = 6
    return table1_data, best_k

def run_dbscan(X, eps_candidates=[10.0, 15.0, 20.0, 25.0], min_samples_candidates=[5, 10, 15]):
    print("\n--- Running DBSCAN Parameter Exploration ---")
    tuning_results = []
    best_db = None
    best_sil = -1
    best_params = {}
    
    for eps in eps_candidates:
        for ms in min_samples_candidates:
            db = DBSCAN(eps=eps, min_samples=ms)
            labels = db.fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            noise_ratio = n_noise / len(labels)
            
            if n_clusters > 1 and n_noise < len(labels):
                non_noise_mask = labels != -1
                if len(np.unique(labels[non_noise_mask])) > 1:
                    sil = silhouette_score(X[non_noise_mask], labels[non_noise_mask])
                else:
                    sil = -1
            else:
                sil = -1
                
            tuning_results.append({
                'eps': eps,
                'min_samples': ms,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_pct': round(noise_ratio * 100, 2),
                'silhouette': round(sil, 4) if sil != -1 else 0.0
            })
            
            if n_clusters >= 2 and sil > best_sil:
                best_sil = sil
                best_db = labels
                best_params = {'eps': eps, 'min_samples': ms, 'n_clusters': n_clusters}
                
    if best_db is None: # fallback
        db = DBSCAN(eps=15.0, min_samples=10)
        best_db = db.fit_predict(X)
        best_params = {'eps': 15.0, 'min_samples': 10, 'n_clusters': len(set(best_db)) - (1 if -1 in best_db else 0)}
        
    print(f"Optimal DBSCAN: eps={best_params['eps']}, min_samples={best_params['min_samples']}, clusters={best_params['n_clusters']}")
    return best_db, best_params, tuning_results

def run_hierarchical(X, n_clusters=6):
    print("\n--- Running Hierarchical Agglomerative Clustering (HAC) ---")
    linkage_methods = ['ward', 'complete', 'average', 'single']
    linkage_results = {}
    
    for method in linkage_methods:
        hac = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
        labels = hac.fit_predict(X)
        sil = silhouette_score(X, labels)
        db_score = davies_bouldin_score(X, labels)
        ch_score = calinski_harabasz_score(X, labels)
        linkage_results[method] = {
            'labels': labels,
            'silhouette': round(sil, 4),
            'davies_bouldin': round(db_score, 4),
            'calinski_harabasz': round(ch_score, 2)
        }
        print(f"HAC [{method}]: Silhouette={sil:.4f}, DB={db_score:.4f}, CH={ch_score:.2f}")
        
    # Dendrogram using a small subset for clear visualization
    subset_X = X[:80]
    Z = linkage(subset_X, method='ward')
    plt.figure(figsize=(10, 5))
    dendrogram(Z, truncate_mode='lastp', p=20, leaf_rotation=45, leaf_font_size=10, show_contracted=True)
    plt.title("Hierarchical Clustering Dendrogram (Ward's Linkage)", pad=12)
    plt.xlabel('Sample Cluster Index')
    plt.ylabel('Euclidean Distance')
    plt.tight_layout()
    dendro_path = os.path.join(EX8_DIR, 'Dendrogram.eps')
    plt.savefig(dendro_path, format='eps', dpi=300)
    plt.close()
    
    return linkage_results

def evaluate_and_compare(X, y, km_labels, db_labels, hac_labels, activity_map):
    print("\n--- Computing Clustering Evaluation Metrics ---")
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    
    # 2D Scatter visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Ground Truth
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=[activity_map[label] for label in y],
                    palette='tab10', s=25, alpha=0.75, ax=axes[0, 0])
    axes[0, 0].set_title('Ground Truth Activities (PCA 2D)', weight='bold')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # K-Means
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=[f"Cluster {c}" for c in km_labels],
                    palette='Set2', s=25, alpha=0.75, ax=axes[0, 1])
    axes[0, 1].set_title('K-Means Clusters (k=6)', weight='bold')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # DBSCAN
    db_clean_names = [f"Cluster {c}" if c != -1 else "Noise (-1)" for c in db_labels]
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=db_clean_names,
                    palette='tab20', s=25, alpha=0.75, ax=axes[1, 0])
    axes[1, 0].set_title('DBSCAN Clusters & Noise', weight='bold')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Hierarchical Ward
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=[f"Cluster {c}" for c in hac_labels],
                    palette='Paired', s=25, alpha=0.75, ax=axes[1, 1])
    axes[1, 1].set_title("Hierarchical Agglomerative (Ward's, k=6)", weight='bold')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    for ax in axes.flat:
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.grid(True, linestyle=':', alpha=0.6)
        
    plt.tight_layout()
    scatter_path = os.path.join(EX8_DIR, 'Cluster_Scatter.eps')
    plt.savefig(scatter_path, format='eps', dpi=300)
    plt.close()
    
    # Metrics calculation
    models = {
        'K-Means (k=6)': km_labels,
        'DBSCAN': db_labels,
        'Hierarchical (Ward)': hac_labels
    }
    
    comparison_metrics = []
    for name, labels in models.items():
        # Mask noise for DBSCAN internal metrics if needed
        valid_mask = labels != -1 if -1 in labels else np.ones(len(labels), dtype=bool)
        if len(np.unique(labels[valid_mask])) > 1:
            sil = silhouette_score(X[valid_mask], labels[valid_mask])
            db_score = davies_bouldin_score(X[valid_mask], labels[valid_mask])
            ch_score = calinski_harabasz_score(X[valid_mask], labels[valid_mask])
        else:
            sil, db_score, ch_score = 0.0, 0.0, 0.0
            
        ari = adjusted_rand_score(y, labels)
        nmi = normalized_mutual_info_score(y, labels)
        
        comparison_metrics.append({
            'Algorithm': name,
            'Silhouette Score': round(sil, 4),
            'Davies-Bouldin Index': round(db_score, 4),
            'Calinski-Harabasz Index': round(ch_score, 2),
            'Adjusted Rand Index (ARI)': round(ari, 4),
            'Normalized Mutual Info (NMI)': round(nmi, 4)
        })
        
    # Bar plot of comparison
    df_metrics = pd.DataFrame(comparison_metrics)
    plt.figure(figsize=(10, 5))
    metrics_to_plot = ['Silhouette Score', 'Adjusted Rand Index (ARI)', 'Normalized Mutual Info (NMI)']
    df_melt = df_metrics.melt(id_vars='Algorithm', value_vars=metrics_to_plot, var_name='Metric', value_name='Score')
    sns.barplot(data=df_melt, x='Algorithm', y='Score', hue='Metric', palette='viridis')
    plt.title('Clustering Algorithms Performance Comparison', pad=12)
    plt.ylabel('Score')
    plt.ylim(-0.1, 1.0)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.tight_layout()
    bar_path = os.path.join(EX8_DIR, 'Clustering_Comparison.eps')
    plt.savefig(bar_path, format='eps', dpi=300)
    plt.close()
    
    return comparison_metrics, df_metrics

def main():
    X, y, activity_map, features = load_har_data(sample_size=3000)
    
    # 1. K-Means
    table1_data, best_k = run_kmeans_elbow(X)
    km = KMeans(n_clusters=best_k, init='k-means++', n_init=10, random_state=42)
    km_labels = km.fit_predict(X)
    
    # 2. DBSCAN
    db_labels, best_db_params, db_tuning = run_dbscan(X)
    
    # 3. Hierarchical
    hac_results = run_hierarchical(X, n_clusters=6)
    hac_labels = hac_results['ward']['labels']
    
    # 4. Evaluation & Visualizations
    comp_metrics, df_metrics = evaluate_and_compare(X, y, km_labels, db_labels, hac_labels, activity_map)
    
    # Save results to JSON
    ex8_data = {
        'table1_elbow': table1_data,
        'db_tuning': db_tuning[:6],
        'hac_linkages': {k: {m: v for m, v in val.items() if m != 'labels'} for k, val in hac_results.items()},
        'comparison_metrics': comp_metrics,
        'best_k': best_k,
        'db_params': best_db_params
    }
    
    with open(os.path.join(EX8_DIR, 'ex8_results.json'), 'w') as f:
        json.dump(ex8_data, f, indent=2)
        
    print("\nExperiment 8 computation and graphics generated successfully!")
    print(df_metrics)

if __name__ == '__main__':
    main()
