import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.neural_network import MLPClassifier

# Styling for publication quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 13,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.titlesize': 15
})

EX9_DIR = "/home/nkandasamy/Desktop/M.Tech/III year/5th Sem/ML/Lab/Machine_Learning_Lab-3122247001036/Ex9"
DATA_DIR = "/home/nkandasamy/Desktop/M.Tech/III year/5th Sem/ML/Lab/Machine_Learning_Lab-3122247001036/Datasets/English_Characters"
os.makedirs(EX9_DIR, exist_ok=True)

def load_and_preprocess_images(target_size=(28, 28)):
    print("\n--- Loading & Preprocessing Handwritten Characters Dataset ---")
    csv_path = os.path.join(DATA_DIR, "english.csv")
    df = pd.read_csv(csv_path)
    
    images = []
    labels = []
    
    for _, row in df.iterrows():
        rel_img_path = row['image']
        label = str(row['label'])
        full_img_path = os.path.join(DATA_DIR, rel_img_path)
        
        if os.path.exists(full_img_path):
            with Image.open(full_img_path) as img:
                # Convert to grayscale, resize, and normalize to [0, 1]
                img_gray = img.convert('L').resize(target_size)
                img_arr = np.array(img_gray, dtype=np.float32) / 255.0
                images.append(img_arr.flatten())
                labels.append(label)
                
    X = np.array(images)
    y = np.array(labels)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"Loaded {len(X)} images of dimension {target_size[0]}x{target_size[1]} ({X.shape[1]} features).")
    print(f"Number of distinct character classes: {len(le.classes_)}")
    
    # Stratified Train/Test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    return X_train, X_test, y_train, y_test, le

class MultiClassPLAFromScratch:
    """
    Single-Layer Perceptron Learning Algorithm (PLA) implemented from scratch
    using Step Activation and One-vs-Rest (OvR) scheme for multi-class classification.
    Weight update rule: w_{t+1} = w_t + eta * (y - y_hat) * x
    """
    def __init__(self, n_classes, lr=0.01, max_epochs=40):
        self.n_classes = n_classes
        self.lr = lr
        self.max_epochs = max_epochs
        self.weights = None
        self.biases = None
        self.epoch_errors = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((self.n_classes, n_features))
        self.biases = np.zeros(self.n_classes)
        
        # Binarize labels for One-vs-Rest
        y_bin = label_binarize(y, classes=range(self.n_classes))
        if self.n_classes == 2 and y_bin.shape[1] == 1:
            y_bin = np.hstack((1 - y_bin, y_bin))
            
        for epoch in range(self.max_epochs):
            total_errors = 0
            for i in range(n_samples):
                xi = X[i]
                for c in range(self.n_classes):
                    target = y_bin[i, c]
                    # Step activation
                    linear_output = np.dot(self.weights[c], xi) + self.biases[c]
                    y_pred = 1 if linear_output >= 0 else 0
                    
                    error = target - y_pred
                    if error != 0:
                        total_errors += abs(error)
                        self.weights[c] += self.lr * error * xi
                        self.biases[c] += self.lr * error
                        
            error_rate = total_errors / (n_samples * self.n_classes)
            self.epoch_errors.append(error_rate)
            if epoch % 10 == 0 or epoch == self.max_epochs - 1:
                print(f"PLA Epoch {epoch+1}/{self.max_epochs} - Avg Error Rate: {error_rate:.4f}")
                
        return self

    def predict(self, X):
        # Raw activation scores
        scores = np.dot(X, self.weights.T) + self.biases
        return np.argmax(scores, axis=1)

    def decision_function(self, X):
        return np.dot(X, self.weights.T) + self.biases

def tune_and_train_mlp(X_train, y_train, X_test, y_test, n_classes):
    print("\n--- Hyperparameter Tuning for Multilayer Perceptron (MLP) ---")
    
    tuning_configs = [
        {'hidden_layer_sizes': (64,), 'activation': 'relu', 'optimizer': 'adam', 'lr': 0.001, 'batch_size': 64},
        {'hidden_layer_sizes': (128, 64), 'activation': 'relu', 'optimizer': 'adam', 'lr': 0.001, 'batch_size': 64},
        {'hidden_layer_sizes': (128, 64), 'activation': 'tanh', 'optimizer': 'adam', 'lr': 0.001, 'batch_size': 64},
        {'hidden_layer_sizes': (128, 64), 'activation': 'relu', 'optimizer': 'sgd', 'lr': 0.01, 'batch_size': 64},
        {'hidden_layer_sizes': (256, 128), 'activation': 'relu', 'optimizer': 'adam', 'lr': 0.001, 'batch_size': 64}
    ]
    
    tuning_results = []
    best_mlp = None
    best_val_acc = -1
    best_config = None
    
    # Sub-split training set into train/val for tuning
    X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    for idx, cfg in enumerate(tuning_configs):
        solver = cfg['optimizer']
        mlp = MLPClassifier(
            hidden_layer_sizes=cfg['hidden_layer_sizes'],
            activation=cfg['activation'],
            solver=solver,
            learning_rate_init=cfg['lr'],
            batch_size=cfg['batch_size'],
            max_iter=45,
            random_state=42,
            early_stopping=False
        )
        
        t0 = time.time()
        mlp.fit(X_tr_sub, y_tr_sub)
        train_time = time.time() - t0
        
        val_acc = accuracy_score(y_val, mlp.predict(X_val))
        print(f"Config {idx+1}: {cfg['hidden_layer_sizes']}, {cfg['activation']}, {cfg['optimizer']}, lr={cfg['lr']} -> Val Acc: {val_acc:.4f}")
        
        tuning_results.append({
            'Configuration': f"Hidden: {cfg['hidden_layer_sizes']}, Act: {cfg['activation']}, Opt: {cfg['optimizer']}, LR: {cfg['lr']}",
            'Hidden Layers': str(cfg['hidden_layer_sizes']),
            'Activation': cfg['activation'],
            'Optimizer': cfg['optimizer'],
            'Learning Rate': cfg['lr'],
            'Validation Accuracy': round(val_acc * 100, 2),
            'Time (s)': round(train_time, 2)
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_config = cfg
            
    print(f"\nOptimal MLP Config: {best_config} with Val Acc: {best_val_acc:.4f}")
    
    # Train final best MLP on full X_train
    final_mlp = MLPClassifier(
        hidden_layer_sizes=best_config['hidden_layer_sizes'],
        activation=best_config['activation'],
        solver=best_config['optimizer'],
        learning_rate_init=best_config['lr'],
        batch_size=best_config['batch_size'],
        max_iter=60,
        random_state=42
    )
    t0 = time.time()
    final_mlp.fit(X_train, y_train)
    final_train_time = time.time() - t0
    
    return final_mlp, best_config, tuning_results, final_train_time

def generate_visualizations(pla_errors, mlp_loss_curve, y_test, pla_preds, mlp_preds, mlp_probs, n_classes):
    print("\n--- Generating Plots & Evaluation Visualizations ---")
    
    # 1. Loss & Error Convergence Plot
    fig, ax1 = plt.subplots(figsize=(9, 5))
    color = '#d62728'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('PLA Error Rate', color=color)
    l1 = ax1.plot(range(1, len(pla_errors) + 1), pla_errors, color=color, linewidth=2.2, label='PLA Training Error')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2 = ax1.twinx()
    color = '#1f77b4'
    ax2.set_ylabel('MLP Cross-Entropy Loss', color=color)
    l2 = ax2.plot(range(1, len(mlp_loss_curve) + 1), mlp_loss_curve, color=color, linewidth=2.2, linestyle='--', label='MLP Training Loss')
    ax2.tick_params(axis='y', labelcolor=color)
    
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    plt.title('Training Convergence: PLA Error Rate vs. MLP Loss', pad=12)
    plt.tight_layout()
    loss_path = os.path.join(EX9_DIR, 'Loss_Convergence.eps')
    plt.savefig(loss_path, format='eps', dpi=300)
    plt.close()
    
    # 2. Confusion Matrix for MLP
    plt.figure(figsize=(10, 8))
    # Subsample 15 classes for readable matrix display
    selected_classes = list(range(15))
    mask = np.isin(y_test, selected_classes) & np.isin(mlp_preds, selected_classes)
    cm = confusion_matrix(y_test[mask], mlp_preds[mask], labels=selected_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('MLP Confusion Matrix (Sample Subset of 15 Character Classes)', pad=12)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    cm_path = os.path.join(EX9_DIR, 'Confusion_Matrix.eps')
    plt.savefig(cm_path, format='eps', dpi=300)
    plt.close()
    
    # 3. Multi-Class ROC Curves (Micro & Macro)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    
    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc = dict()
    
    # Micro-average
    fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(y_test_bin.ravel(), mlp_probs.ravel())
    roc_auc["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])
    
    # Macro-average
    all_fpr = np.unique(np.concatenate([roc_curve(y_test_bin[:, i], mlp_probs[:, i])[0] for i in range(min(10, n_classes))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(min(10, n_classes)):
        fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], mlp_probs[:, i])
        mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
    mean_tpr /= min(10, n_classes)
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_dict["micro"], tpr_dict["micro"],
             label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.3f})',
             color='deeppink', linestyle=':', linewidth=3)
    plt.plot(fpr_dict["macro"], tpr_dict["macro"],
             label=f'Macro-average ROC (AUC = {roc_auc["macro"]:.3f})',
             color='navy', linestyle=':', linewidth=3)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Multi-Class ROC Curves (MLP)', pad=12)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    roc_path = os.path.join(EX9_DIR, 'ROC_Curves.eps')
    plt.savefig(roc_path, format='eps', dpi=300)
    plt.close()
    
    return roc_auc

def main():
    X_train, X_test, y_train, y_test, le = load_and_preprocess_images()
    n_classes = len(le.classes_)
    
    # 1. Model A: PLA
    print("\n--- Training Single-Layer Perceptron (PLA) from Scratch ---")
    t0 = time.time()
    pla = MultiClassPLAFromScratch(n_classes=n_classes, lr=0.01, max_epochs=35)
    pla.fit(X_train, y_train)
    pla_time = time.time() - t0
    pla_preds = pla.predict(X_test)
    
    pla_acc = accuracy_score(y_test, pla_preds)
    pla_prec = precision_score(y_test, pla_preds, average='weighted', zero_division=0)
    pla_rec = recall_score(y_test, pla_preds, average='weighted', zero_division=0)
    pla_f1 = f1_score(y_test, pla_preds, average='weighted', zero_division=0)
    
    print(f"PLA Test Results -> Acc: {pla_acc:.4f}, Prec: {pla_prec:.4f}, Rec: {pla_rec:.4f}, F1: {pla_f1:.4f}")
    
    # 2. Model B: MLP with Hyperparameter Tuning
    final_mlp, best_config, tuning_results, mlp_time = tune_and_train_mlp(
        X_train, y_train, X_test, y_test, n_classes
    )
    mlp_preds = final_mlp.predict(X_test)
    mlp_probs = final_mlp.predict_proba(X_test)
    
    mlp_acc = accuracy_score(y_test, mlp_preds)
    mlp_prec = precision_score(y_test, mlp_preds, average='weighted', zero_division=0)
    mlp_rec = recall_score(y_test, mlp_preds, average='weighted', zero_division=0)
    mlp_f1 = f1_score(y_test, mlp_preds, average='weighted', zero_division=0)
    
    print(f"MLP Test Results -> Acc: {mlp_acc:.4f}, Prec: {mlp_prec:.4f}, Rec: {mlp_rec:.4f}, F1: {mlp_f1:.4f}")
    
    # Visualizations
    roc_auc = generate_visualizations(
        pla.epoch_errors, final_mlp.loss_curve_, y_test, pla_preds, mlp_preds, mlp_probs, n_classes
    )
    
    # A/B Comparison Table
    ab_comparison = [
        {
            'Model': 'Single-Layer Perceptron (PLA)',
            'Architecture': 'Single Layer (OvR 62 Step Units)',
            'Accuracy': round(pla_acc * 100, 2),
            'Precision': round(pla_prec * 100, 2),
            'Recall': round(pla_rec * 100, 2),
            'F1-Score': round(pla_f1 * 100, 2),
            'Training Time (s)': round(pla_time, 2)
        },
        {
            'Model': 'Tuned Multilayer Perceptron (MLP)',
            'Architecture': f"Input(784) -> {best_config['hidden_layer_sizes']} -> Output(62)",
            'Accuracy': round(mlp_acc * 100, 2),
            'Precision': round(mlp_prec * 100, 2),
            'Recall': round(mlp_rec * 100, 2),
            'F1-Score': round(mlp_f1 * 100, 2),
            'Training Time (s)': round(mlp_time, 2)
        }
    ]
    df_ab = pd.DataFrame(ab_comparison)
    print("\n=== A/B EXPERIMENT COMPARISON ===")
    print(df_ab)
    
    ex9_data = {
        'ab_comparison': ab_comparison,
        'tuning_results': tuning_results,
        'best_config': {k: str(v) for k, v in best_config.items()},
        'roc_auc': {k: round(v, 4) for k, v in roc_auc.items()},
        'pla_error_final': round(pla.epoch_errors[-1], 4),
        'mlp_loss_final': round(final_mlp.loss_curve_[-1], 4)
    }
    
    with open(os.path.join(EX9_DIR, 'ex9_results.json'), 'w') as f:
        json.dump(ex9_data, f, indent=2)
        
    print("\nExperiment 9 computation and graphics generated successfully!")

if __name__ == '__main__':
    main()
