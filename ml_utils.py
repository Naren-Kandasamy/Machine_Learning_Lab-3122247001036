import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# Global plotting configuration per strictly defined guidelines
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 15
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['legend.fontsize'] = 15
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
# Note: Matplotlib doesn't have a direct 'bold' setting for all axes labels globally in a single line, 
# but axes.labelweight handles it for the x and y labels.

def generate_eda_subplots(df, output_path, target_col=None):
    """
    Generates one consolidated EDA figure containing exactly 12 different subplots
    arranged on a single page, saved in .eps format at 600 DPI.
    """
    # Select up to 12 numeric features to plot
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    features_to_plot = numeric_cols[:12]
    
    # If there are fewer than 12, we will pad with dummy empty plots or duplicate variations
    # But typically ML datasets will have enough features or we'll plot boxplots and histograms
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    axes = axes.flatten()
    
    for i in range(12):
        if i < len(features_to_plot):
            sns.histplot(data=df, x=features_to_plot[i], kde=True, ax=axes[i], color='blue')
            axes[i].set_title(f'Distribution of {features_to_plot[i]}', fontname='Times New Roman', fontsize=15, fontweight='bold')
            axes[i].set_xlabel(features_to_plot[i], fontname='Times New Roman', fontsize=15, fontweight='bold')
            axes[i].set_ylabel('Count', fontname='Times New Roman', fontsize=15, fontweight='bold')
        elif i < len(features_to_plot) * 2 and (i - len(features_to_plot)) < len(features_to_plot):
            # Boxplots if we ran out of unique features to reach 12
            feat = features_to_plot[i - len(features_to_plot)]
            sns.boxplot(data=df, y=feat, ax=axes[i], color='orange')
            axes[i].set_title(f'Boxplot of {feat}', fontname='Times New Roman', fontsize=15, fontweight='bold')
            axes[i].set_xlabel(feat, fontname='Times New Roman', fontsize=15, fontweight='bold')
            axes[i].set_ylabel('Value', fontname='Times New Roman', fontsize=15, fontweight='bold')
        else:
            axes[i].set_visible(False)
            
    plt.tight_layout()
    plt.savefig(output_path, format='eps', dpi=600)
    plt.close()

def train_evaluate_classification(model, X_train, y_train, X_test, y_test):
    """
    Reusable function for training and evaluating classification models.
    """
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
    
    start_pred = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_pred
    
    return y_pred, train_time, pred_time

def train_evaluate_regression(model, X_train, y_train, X_test, y_test):
    """
    Reusable function for training and evaluating regression models.
    """
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
    
    start_pred = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_pred
    
    return y_pred, train_time, pred_time

def compute_classification_metrics(y_true, y_pred):
    """
    Computes and displays classification performance metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

def compute_regression_metrics(y_true, y_pred):
    """
    Computes and displays regression performance metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
