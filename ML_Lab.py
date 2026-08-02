#!/usr/bin/env python
# coding: utf-8

# # Experiment 1: Automated Machine Learning Workflow
# This notebook implements a complete end-to-end automated machine learning data pipeline.
# 
# ### Associated Machine Learning Tasks:
# 1. **Iris Dataset:** Supervised Learning -> Multi-class Classification (predicting Iris flower species).
# 2. **Diabetes Dataset:** Supervised Learning -> Binary Classification (predicting diabetes onset).
# 3. **Loan Dataset:** Supervised Learning -> Binary Classification (predicting loan approval status).
# 4. **Email Spam Dataset:** Supervised Learning -> Binary Classification / NLP (identifying spam vs. ham).
# 5. **MNIST Dataset:** Supervised Learning -> Multi-class Classification / Image Recognition (recognizing handwritten digits 0-9).

# In[1]:


import pandas as pd
import numpy as np
import os
import struct
from sklearn.preprocessing import StandardScaler

def smart_loader(data_source):
    # Identifies the file type and loads it appropriately
    
    # 1. Handle Partitioned Datasets (Dictionaries)
    if isinstance(data_source, dict):
        print("Partitioned dataset detected. Loading splits...")
        loaded_splits = {}
        for split_name, file_path in data_source.items():
            # Recursively call the loader for each file
            data, data_type = smart_loader(file_path)
            loaded_splits[split_name] = data
        return loaded_splits, f"{data_type}_split"

    # 2. File Existence Check (Fails fast if the path is wrong)
    if not os.path.exists(data_source):
        raise FileNotFoundError(f"Local file not found: {data_source}")

    # Extract extension/filename for single files
    _, ext = os.path.splitext(data_source)
    ext = ext.lower()
    basename = os.path.basename(data_source).lower()

    # 3. Handle Known Tabular Formats
    if ext in ['.csv', '.data']:
        print(f"Loading CSV/data file: {os.path.basename(data_source)}")
        if 'email' in basename or 'spam' in basename:
            print(f"Loading NLP/Text data file: {os.path.basename(data_source)}")
            return pd.read_csv(data_source), "nlp_vectorized" # <-- New Tag!
            
        if 'iris' in basename:
            # Iris dataset doesn't have a header in the raw file
            return pd.read_csv(data_source, header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']), "tabular"
        return pd.read_csv(data_source), "tabular"
        
    elif ext in ['.xls', '.xlsx']:
        print(f"Loading Excel file: {os.path.basename(data_source)}")
        return pd.read_excel(data_source), "tabular"

    # 4. Handle Raw Binary MNIST Files (.ubyte)
    elif 'ubyte' in ext or 'ubyte' in data_source:
        print(f"Detected raw byte file: {os.path.basename(data_source)}")
        with open(data_source, 'rb') as f:
            magic_number = struct.unpack('>I', f.read(4))[0]
            if magic_number == 2051: # Images
                num_items, rows, cols = struct.unpack('>III', f.read(12))
                data = np.fromfile(f, dtype=np.uint8).reshape(num_items, rows, cols)
                return data, "mnist_images"
            elif magic_number == 2049: # Labels
                num_items = struct.unpack('>I', f.read(4))[0]
                data = np.fromfile(f, dtype=np.uint8)
                return data, "mnist_labels"
            else:
                raise ValueError(f"Unknown MNIST magic number: {magic_number}")

    # 5. Total Failure
    else:
        raise ValueError(f"Completely unsupported local format: {ext}")


# In[2]:


# --- Test Ingestion of All Datasets ---
print("=== Loading Iris ===")
iris_data, iris_flag = smart_loader('Datasets/iris/bezdekIris.data')

print("\n=== Loading Diabetes ===")
diabetes_data, diabetes_flag = smart_loader('Datasets/Diabetes_Dataset/diabetes.csv')

print("\n=== Loading Loan Dataset (Dictionary Split) ===")
loan_splits = {
    'train': 'Datasets/Loan_Amount_Dataset/loan-train.csv',
    'test': 'Datasets/Loan_Amount_Dataset/loan-test.csv'
}
loan_data, loan_flag = smart_loader(loan_splits)

print("\n=== Loading Email Spam ===")
email_data, email_flag = smart_loader('Datasets/Email_Spam_Dataset/emails.csv')

print("\n=== Loading MNIST (Dictionary Split) ===")
mnist_splits = {
    'train_images': 'Datasets/MNIST_Dataset/train-images.idx3-ubyte',
    'train_labels': 'Datasets/MNIST_Dataset/train-labels.idx1-ubyte',
    'test_images': 'Datasets/MNIST_Dataset/t10k-images.idx3-ubyte',
    'test_labels': 'Datasets/MNIST_Dataset/t10k-labels.idx1-ubyte'
}
mnist_data, mnist_flag = smart_loader(mnist_splits)


# In[16]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

def automated_eda(data, data_type):
    """
    Generates summary statistics and visualizations based on data type.
    """
    if "split" in data_type:
        print(f"Partitioned dataset detected with flag: {data_type}")
        has_images = any("image" in k for k in data.keys())
        has_labels = any("label" in k for k in data.keys())
        if "mnist" in data_type and has_images and has_labels:
            img_key = next((k for k in data.keys() if "images" in k or "image" in k), None)
            lbl_key = next((k for k in data.keys() if "labels" in k or "label" in k), None)
            if img_key and lbl_key:
                images, labels = data[img_key], data[lbl_key]
                print(f"\n=== Combined MNIST Visualization ({img_key} & {lbl_key}) ===")
                print(f"Images Shape: {images.shape}, Labels Shape: {labels.shape}")
                print(f"Pixel Range: [{images.min()}, {images.max()}]")
                
                fig = plt.figure(figsize=(15, 6))
                gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1.0])
                
                # Left: Label Distribution
                ax_lbl = fig.add_subplot(gs[0])
                unique, counts = np.unique(labels, return_counts=True)
                sns.barplot(x=unique, y=counts, palette='viridis', ax=ax_lbl)
                ax_lbl.set_title("MNIST Class Distribution")
                ax_lbl.set_xlabel("Digit Class")
                ax_lbl.set_ylabel("Frequency")
                
                # Right: Grid of 3x3 digits
                gs_right = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[1])
                for i in range(9):
                    ax_img = fig.add_subplot(gs_right[i])
                    idx = np.random.randint(0, len(images))
                    ax_img.imshow(images[idx], cmap='gray')
                    ax_img.axis('off')
                    ax_img.set_title(f"Idx: {idx}", fontsize=8)
                
                plt.suptitle("MNIST Dataset Summary Dashboard", fontsize=16)
                plt.tight_layout()
                plt.show()
                return
        
        for name, split in data.items():
            print(f"\n--- Exploring Split: {name} ---")
            if isinstance(split, pd.DataFrame):
                automated_eda(split, "tabular")
            elif isinstance(split, np.ndarray):
                m_type = "mnist_images" if len(split.shape) == 3 else "mnist_labels"
                automated_eda(split, m_type)

    elif "tabular" in data_type:
        print("=== TABULAR SUMMARY STATISTICS ===")
        print(data.info())
        print("\n--- Descriptive Statistics ---")
        print(data.describe())
        
        # Isolate numerical columns
        num_cols = data.select_dtypes(include=[np.number]).columns
        
        print("\n=== TABULAR VISUALIZATIONS ===")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Subplot 1: Missing Data Profile
        sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis', ax=axes[0])
        axes[0].set_title("Missing Data Profile (Yellow = Missing)")
        
        # Subplot 2: Correlation Heatmap
        if len(num_cols) > 0:
            sns.heatmap(data[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1])
            axes[1].set_title("Feature Correlation Matrix")
        else:
            axes[1].text(0.5, 0.5, 'No Numerical Columns', ha='center', va='center')
            axes[1].set_title("Feature Correlation Matrix")
            
        # Subplot 3: Boxplot of Scaled Numerical Features
        if len(num_cols) > 0:
            # Scale features temporarily for clean boxplot representation
            temp_scaler = StandardScaler()
            temp_scaled = temp_scaler.fit_transform(data[num_cols].dropna())
            temp_df = pd.DataFrame(temp_scaled, columns=num_cols)
            sns.boxplot(data=temp_df, ax=axes[2], palette='Set2')
            axes[2].set_title("Boxplot of Scaled Numerical Features")
            axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha='right')
        else:
            axes[2].text(0.5, 0.5, 'No Numerical Columns', ha='center', va='center')
            axes[2].set_title("Boxplot of Features")
            
        plt.tight_layout()
        plt.show()
        
        # Scatter Plot (First two numerical features if they exist)
        if len(num_cols) >= 2:
            plt.figure(figsize=(8, 4.5))
            sns.scatterplot(data=data, x=num_cols[0], y=num_cols[1], hue=data.columns[-1], palette='viridis')
            plt.title(f"Scatter Plot: {num_cols[0]} vs {num_cols[1]}")
            plt.tight_layout()
            plt.show()
        
        # Histograms
        if len(num_cols) > 0:
            data[num_cols].hist(figsize=(12, 8), bins=20, edgecolor='black', color='skyblue')
            plt.suptitle("Numerical Feature Distributions")
            plt.tight_layout()
            plt.show()

    elif "mnist" in data_type:
        print("=== MNIST DATA STATISTICS ===")
        print(f"Data type: {data_type}")
        print(f"Data shape: {data.shape}")
        print(f"Data values range: [{data.min()}, {data.max()}]")
        
        if data_type == "mnist_images":
            print("\n=== MNIST IMAGE VISUALIZATION ===")
            fig, axes = plt.subplots(3, 3, figsize=(6, 6))
            for i, ax in enumerate(axes.flat):
                idx = np.random.randint(0, len(data))
                ax.imshow(data[idx], cmap='gray')
                ax.axis('off')
                ax.set_title(f"Idx: {idx}")
            plt.suptitle("Sample Digits Grid")
            plt.tight_layout()
            plt.show()
            
        elif data_type == "mnist_labels":
            print("\n=== MNIST LABEL VISUALIZATION ===")
            unique, counts = np.unique(data, return_counts=True)
            plt.figure(figsize=(8, 4))
            sns.barplot(x=unique, y=counts, palette='viridis')
            plt.title("MNIST Class Distribution")
            plt.xlabel("Digit Class")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()
    elif "nlp_vectorized" in data_type:
        print("=== NLP DATA SUMMARY ===")
        print(f"Total Documents (Emails): {data.shape[0]}")
        print(f"Total Vocabulary (Words): {data.shape[1] - 2}") # Subtracting ID and Target
        
        # 1. Find the target column (usually 'Prediction' in spam datasets)
        target_col = 'Prediction' if 'Prediction' in data.columns else data.columns[-1]
        
        # Plot Class Distribution
        plt.figure(figsize=(6, 4))
        sns.countplot(data=data, x=target_col, palette='Set2')
        plt.title("Class Distribution (0 = Ham, 1 = Spam)")
        plt.xlabel("Email Type")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()
        
        # 2. Top 20 Most Frequent Words
        print("\n=== TOP WORD FREQUENCIES ===")
        
        # Identify ID columns and the Target column
        junk_cols = [c for c in data.columns if 'id' in c.lower() or 'no.' in c.lower()] + [target_col]
        
        # NEW: Identify tiny garbage columns (1 or 2 letters long)
        tiny_words = [c for c in data.columns if len(str(c)) <= 2]
        
        # Combine the lists and drop them
        cols_to_drop = junk_cols + tiny_words
        word_data = data.drop(columns=cols_to_drop, errors='ignore')
        
        # Sum all the columns to find out how many times each word was used overall
        word_sums = word_data.sum().sort_values(ascending=False).head(20)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=word_sums.values, y=word_sums.index, palette='viridis')
        plt.title("Top 20 Most Frequent Words (Filtered)")
        plt.xlabel("Total Occurrences")
        plt.ylabel("Word")
        plt.tight_layout()
        plt.show()


# In[17]:


# --- Run EDA on Datasets ---
print("=== Running EDA on Iris Dataset ===")
automated_eda(iris_data, iris_flag)

print("\n=== Running EDA on Diabetes Dataset ===")
automated_eda(diabetes_data, diabetes_flag)

print("\n=== Running EDA on MNIST (Train Images) ===")
automated_eda(mnist_data['train_images'], 'mnist_images')

print("\n=== Running EDA on MNIST (Train Labels) ===")
automated_eda(mnist_data['train_labels'], 'mnist_labels')

print("\n=== Running Combined MNIST Dashboard ===")
automated_eda(mnist_data, mnist_flag)

print("\n=== Running EDA on Email Spam Dataset ===")
automated_eda(email_data, email_flag)


# In[18]:


from sklearn.preprocessing import StandardScaler
def preprocess_dataset(data, data_type, dataset_name=None, target_col=None):
    """
    Performs data cleaning, imputation, categorical encoding, and scaling.
    """
    if "split" in data_type:
        processed_splits = {}
        for name, split in data.items():
            print(f"\n--- Preprocessing Split: {name} ---")
            if isinstance(split, pd.DataFrame):
                processed_splits[name] = preprocess_dataset(split, "tabular", dataset_name=dataset_name, target_col=target_col)
            elif isinstance(split, np.ndarray):
                m_type = "mnist_images" if len(split.shape) == 3 else "mnist_labels"
                processed_splits[name] = preprocess_dataset(split, m_type, target_col=target_col)
        return processed_splits
    elif "tabular" in data_type:
        df = data.copy()
        
        # 0. Drop rows with missing target variable
        if target_col and target_col in df.columns:
            initial_len = len(df)
            df = df.dropna(subset=[target_col])
            dropped_len = initial_len - len(df)
            if dropped_len > 0:
                print(f"Dropped {dropped_len} rows with missing target variable '{target_col}'.")
        
        # 1. Handle dataset-specific medical placeholders
        if dataset_name and "diabetes" in dataset_name.lower():
            # In diabetes, zeros in certain columns are invalid
            invalid_zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
            for col in invalid_zero_cols:
                if col in df.columns:
                    df[col] = df[col].replace(0, np.nan)
            print("Diabetes: Replaced invalid zeros with NaN.")
            
        # 2. Impute missing values
        # Numeric imputation (median)
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"Imputed missing numeric values in '{col}' with median: {median_val}")
                
        # Categorical imputation (mode)
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                print(f"Imputed missing categorical values in '{col}' with mode: {mode_val}")
        print("--- Outlier Detection ---")
        original_row_count = df.shape[0]
        target_col_name = df.columns[-1] # Identify target so we don't drop target classes
        
        # We only check for outliers in continuous numerical columns
        for col in num_cols:
            if not col.lower().endswith('_id') and col != target_col_name:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Filter the dataframe to keep ONLY rows within the boundaries
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                
        rows_dropped = original_row_count - df.shape[0]
        print(f"Dropped {rows_dropped} rows containing extreme outliers.\n")
        if rows_dropped > 0:
            print(f">>> {rows_dropped} outliers detected and removed.\n")
        else:
            print(">>> 0 outliers detected.\n")
        # 3. Categorical encoding
        # Refresh cat_cols after imputation
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        cols_to_encode = [c for c in cat_cols if not c.lower().endswith('_id')]
        for col in cols_to_encode:
            df[col] = df[col].astype('category').cat.codes
            print(f"Encoded categorical column '{col}' to category codes.")
            
        # 4. Feature Scaling (Standardization)
        target_col = df.columns[-1]
        scale_cols = [c for c in num_cols if c != target_col and not c.lower().endswith('_id')]
        
        if len(scale_cols) > 0:
            scaler = StandardScaler()
            df[scale_cols] = scaler.fit_transform(df[scale_cols])
            print(f"Standardized numerical columns: {scale_cols}")
            
        return df
        
    elif "mnist" in data_type:
        if data_type == "mnist_images":
            # 1. Normalize pixel values [0, 255] -> [0.0, 1.0]
            processed_data = data.astype(np.float32) / 255.0
            print("MNIST Images: Normalized pixel values to [0.0, 1.0]")
            
            # 2. Flatten 2D image arrays into 1D vectors (N x 784)
            num_images, rows, cols = processed_data.shape
            processed_data = processed_data.reshape(num_images, rows * cols)
            print(f"MNIST Images: Flattened 2D images {rows}x{cols} to 1D vectors of size {rows*cols}")
            return processed_data
            
        elif data_type == "mnist_labels":
            print("MNIST Labels: No preprocessing needed.")
            return data
    elif "nlp_vectorized" in data_type:
        df = data.copy()
        
        # 0. Drop rows with missing target variable
        if target_col and target_col in df.columns:
            initial_len = len(df)
            df = df.dropna(subset=[target_col])
            dropped_len = initial_len - len(df)
            if dropped_len > 0:
                print(f"Dropped {dropped_len} rows with missing target variable '{target_col}'.")
        print("NLP Data: Skipping standard imputation and scaling to preserve sparsity.")
        
        # NLP datasets often have an ID column we need to drop immediately 
        id_cols = [c for c in df.columns if 'id' in c.lower() or 'no.' in c.lower()]
        if id_cols:
            df = df.drop(columns=id_cols)
            print(f"Dropped ID columns: {id_cols}")
        return df


# In[19]:


# --- Run Preprocessing on All Datasets ---
print("=== Preprocessing Iris Dataset ===")
processed_iris = preprocess_dataset(iris_data, iris_flag)
print("Processed Iris Shape:", processed_iris.shape)
print(processed_iris.head(2))

print("\n=== Preprocessing Diabetes Dataset ===")
processed_diabetes = preprocess_dataset(diabetes_data, diabetes_flag, dataset_name="Diabetes")
print("Processed Diabetes Shape:", processed_diabetes.shape)
print(processed_diabetes.head(2))

print("\n=== Preprocessing Loan Splits ===")
processed_loan = preprocess_dataset(loan_data, loan_flag)
print("Processed Loan Train Shape:", processed_loan['train'].shape)
print("Processed Loan Test Shape:", processed_loan['test'].shape)
print(processed_loan['train'].head(2))

print("\n=== Preprocessing Email Spam ===")
processed_email = preprocess_dataset(email_data, email_flag)
print("Processed Email Shape:", processed_email.shape)

print("\n=== Preprocessing MNIST Splits ===")
processed_mnist = preprocess_dataset(mnist_data, mnist_flag)
print("Processed MNIST Train Images Shape:", processed_mnist['train_images'].shape)
print("Processed MNIST Train Images Min/Max:", processed_mnist['train_images'].min(), "/", processed_mnist['train_images'].max())


# In[29]:


from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

def select_features(X, y, k=4):
    """
    Selects top k features using SelectKBest with ANOVA F-value (f_classif).
    """
    selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
    X_new = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)
    selected_features = X.columns[selected_indices] if isinstance(X, pd.DataFrame) else list(range(X.shape[1]))
    scores = selector.scores_[selected_indices]
    
    print("--- Feature Selection Scores ---")
    for f, s in zip(selected_features, scores):
        print(f"Feature: {f:<25} | F-Score: {s:.2f}")
        
    return X_new, list(selected_features)

def split_dataset(X, y, val_size=0.15, test_size=0.15, is_regression=False):
    """
    Splits dataset into Train, Validation, and Test sets.
    """
    rem_size = val_size + test_size
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=rem_size, random_state=42, stratify=None if is_regression else y)
    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=test_size/rem_size, random_state=42, stratify=None if is_regression else y_rem)
    
    print(f"Split completed - Train: {X_train.shape[0]} samples, Val: {X_val.shape[0]} samples, Test: {X_test.shape[0]} samples")
    return X_train, X_val, X_test, y_train, y_val, y_test


# In[22]:


# --- Run Feature Selection and Data Splitting ---

# 1. Iris dataset
print("=== Processing Iris Dataset ===")
X_iris = processed_iris.iloc[:, :-1]
y_iris = processed_iris.iloc[:, -1]
X_iris_sel, iris_features = select_features(X_iris, y_iris, k=3)
X_iris_tr, X_iris_val, X_iris_te, y_iris_tr, y_iris_val, y_iris_te = split_dataset(X_iris_sel, y_iris)

# 2. Diabetes dataset
print("\n=== Processing Diabetes Dataset ===")
X_diab = processed_diabetes.iloc[:, :-1]
y_diab = processed_diabetes.iloc[:, -1]
X_diab_sel, diab_features = select_features(X_diab, y_diab, k=5)
X_diab_tr, X_diab_val, X_diab_te, y_diab_tr, y_diab_val, y_diab_te = split_dataset(X_diab_sel, y_diab)

# 3. Loan dataset (using train split for manual partition)
print("\n=== Processing Loan Dataset ===")
X_loan_train = processed_loan['train'].drop(columns=['Loan_ID', 'Loan_Status'])
y_loan_train = processed_loan['train']['Loan_Status']
X_loan_tr, X_loan_val, X_loan_te, y_loan_tr, y_loan_val, y_loan_te = split_dataset(X_loan_train, y_loan_train)

# 4. Email Spam dataset
print("\n=== Processing Email Spam Dataset ===")
X_email = processed_email.drop(columns=['Email No.', 'Prediction'], errors='ignore')
y_email = processed_email['Prediction']
X_email_sel, email_features = select_features(X_email, y_email, k=10)
X_email_tr, X_email_val, X_email_te, y_email_tr, y_email_val, y_email_te = split_dataset(X_email_sel, y_email)

# 5. MNIST dataset
print("\n=== Processing MNIST Dataset ===")
X_mnist_train_full = processed_mnist['train_images']
y_mnist_train_full = processed_mnist['train_labels']
X_mnist_te = processed_mnist['test_images']
y_mnist_te = processed_mnist['test_labels']
X_mnist_tr, X_mnist_val, y_mnist_tr, y_mnist_val = train_test_split(
    X_mnist_train_full, y_mnist_train_full, test_size=0.15, random_state=42, stratify=y_mnist_train_full
)
print(f"MNIST splits - Train: {X_mnist_tr.shape[0]}, Val: {X_mnist_val.shape[0]}, Test: {X_mnist_te.shape[0]}")


# In[23]:


def run_data_pipeline(data_source, dataset_name="Dataset", target_col=None, drop_cols=None, k_features=5):
    """
    End-to-end master pipeline: Loads, explores, cleans, 
    selects features, and splits the data.
    """
    print(f"========== STARTING PIPELINE: {dataset_name} ==========\n")
    
    # 1. Ingestion
    print("[Step 1: Loading Data]")
    raw_data, data_type = smart_loader(data_source)
    
    # 2. Exploration (EDA)
    print("\n[Step 2: Exploratory Data Analysis]")
    automated_eda(raw_data, data_type)
    
    # 3. Preprocessing (Cleaning, Encoding, Scaling)
    print("\n[Step 3: Preprocessing]")
    processed_data = preprocess_dataset(raw_data, data_type, dataset_name=dataset_name, target_col=target_col)
    
    # 4 & 5. Feature Selection & Splitting
    if "tabular" in data_type:
        print("\n[Step 4: Feature Selection]")
        
        # Handle custom drops and target variables
        X = processed_data.copy()
        if drop_cols:
            X = X.drop(columns=drop_cols, errors='ignore')
        if target_col:
            X = X.drop(columns=[target_col], errors='ignore')
            y = processed_data[target_col]
        else:
            X = X.iloc[:, :-1]
            y = processed_data.iloc[:, -1]
            
        # Select top features
        X_selected, selected_features = select_features(X, y, k=k_features)
        
        print("\n[Step 5: Train/Val/Test Split]")
        X_tr, X_val, X_te, y_tr, y_val, y_te = split_dataset(X_selected, y, is_regression=(dataset_name == 'Loan Dataset'))
        
        print(f"\n========== PIPELINE COMPLETE: {dataset_name} ==========")
        
        return {
            'X_train': X_tr, 'X_val': X_val, 'X_test': X_te,
            'y_train': y_tr, 'y_val': y_val, 'y_test': y_te,
            'selected_features': selected_features
        }
        
    elif "nlp_vectorized" in data_type:
        print("\n[Step 4: NLP Feature Routing]")
        print("Skipping SelectKBest. NLP models perform better with full vocabularies.")
        
        # Isolate target (assuming it's the last column, or defined by target_col)
        if target_col:
            X = processed_data.drop(columns=[target_col], errors='ignore')
            y = processed_data[target_col]
        else:
            X = processed_data.iloc[:, :-1]
            y = processed_data.iloc[:, -1]
            
        print("\n[Step 5: Train/Val/Test Split]")
        X_tr, X_val, X_te, y_tr, y_val, y_te = split_dataset(X, y)
        
        print(f"\n========== PIPELINE COMPLETE: {dataset_name} ==========")
        return {
            'X_train': X_tr, 'X_val': X_val, 'X_test': X_te,
            'y_train': y_tr, 'y_val': y_val, 'y_test': y_te,
            'selected_features': "Full Vocabulary"
        }
        
    else:
        print("\n[Step 4 & 5: Routing Complex/Image Data]")
        print(f"Note: Standard tabular feature selection skipped for {data_type}.")
        print(f"\n========== PIPELINE COMPLETE: {dataset_name} ==========")
        return processed_data


# In[24]:


# Iris (Default behavior, target is last column)
iris_ready = run_data_pipeline('Datasets/iris/bezdekIris.data', dataset_name="Iris", k_features=3)

# Email Spam (Explicitly dropping the ID and targeting 'Prediction')
email_ready = run_data_pipeline(
    'Datasets/Email_Spam_Dataset/emails.csv', 
    dataset_name="Email Spam", 
    target_col='Prediction', 
    drop_cols=['Email No.'], 
    k_features=10
)


# In[9]:


from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, dataset_name):
    """
    Trains a model and evaluates it on Train, Val, and Test splits.
    """
    model.fit(X_train, y_train)
    
    # Predict
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Calculate Metrics
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    print(f"=== Performance: {dataset_name} ===")
    print(f"Validation Accuracy: {val_acc:.4f} | Validation F1-Score: {val_f1:.4f}")
    print(f"Test Accuracy:       {test_acc:.4f} | Test F1-Score:       {test_f1:.4f}")
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_test_pred))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Confusion Matrix - {dataset_name} (Test)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()
    
    return {
        'Dataset': dataset_name,
        'Val Accuracy': val_acc,
        'Val F1': val_f1,
        'Test Accuracy': test_acc,
        'Test F1': test_f1
    }


# In[28]:


import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

email_data_dict = run_data_pipeline(
    'Datasets/Email_Spam_Dataset/emails.csv', 
    dataset_name="Email Spam", 
    target_col='Prediction'
)

X_train_raw = email_data_dict['X_train']
X_test_raw = email_data_dict['X_test']
y_train = email_data_dict['y_train']
y_test = email_data_dict['y_test']

results = {}

def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    # Track Training Time
    start_train = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - start_train
    
    # Track Prediction Time
    start_pred = time.time()
    y_pred = model.predict(X_te)
    pred_time = time.time() - start_pred
    
    # Calculate Metrics
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    
    results[name] = {
        'Accuracy': acc,
        'F1-Score': f1,
        'Train Time (s)': train_time,
        'Predict Time (s)': pred_time
    }
    print(f"{name:<15} | Acc: {acc:.4f} | F1: {f1:.4f} | Train: {train_time:.4f}s | Pred: {pred_time:.4f}s")

evaluate_model("Gaussian NB", GaussianNB(), X_train_raw, y_train, X_test_raw, y_test)
evaluate_model("Multinomial NB", MultinomialNB(), X_train_raw, y_train, X_test_raw, y_test)
evaluate_model("Bernoulli NB", BernoulliNB(binarize=0.0), X_train_raw, y_train, X_test_raw, y_test)

print("\nScaling features specifically for KNN...")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# Running baseline KNN with k=5
evaluate_model("KNN (Baseline)", KNeighborsClassifier(n_neighbors=5), X_train_scaled, y_train, X_test_scaled, y_test)
print("\n=== FINAL BASELINE COMPARISON ===")
comparison_df = pd.DataFrame(results).T
print(comparison_df.to_string())


# In[ ]:





# # Experiment 2: Email Spam/Ham Classification
# Comparing Naïve Bayes Variations and K-Nearest Neighbors Optimization using modular pipeline functions.

# In[ ]:


from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_naive_bayes(X_train, y_train, X_test, y_test, dataset_name):
    print(f"\n=== Evaluating Naïve Bayes Models for {dataset_name} ===")
    models = {
        "Gaussian NB": GaussianNB(),
        "Multinomial NB": MultinomialNB(),
        "Bernoulli NB": BernoulliNB(binarize=0.5)
    }
    
    results = {}
    best_f1 = 0
    best_model = None
    
    for name, model in models.items():
        # Train
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        
        # Predict
        t1 = time.time()
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        pred_time = time.time() - t1
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1, "ROC-AUC": roc_auc,
            "Train Time (s)": train_time, "Predict Time (s)": pred_time,
            "Model": model, "y_pred": y_pred, "y_prob": y_prob
        }
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            
    # Print comparison
    comparison_df = pd.DataFrame(results).T.drop(columns=["Model", "y_pred", "y_prob"])
    print(comparison_df.to_string())
    return results, best_model


# In[ ]:


def optimize_knn(X_train, y_train, X_test, y_test, dataset_name):
    print(f"\n=== Optimizing KNN for {dataset_name} ===")
    results = {}
    
    # 0. Varying k
    print("\n=== KNN Comparison (varying k) ===")
    print(f"{'k':<5} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
    print("-" * 55)
    k_vals = [1, 3, 5, 7, 9, 11]
    k_accs = []
    for k in k_vals:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred_k = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred_k)
        k_accs.append(acc)
        print(f"{k:<5} | {acc:<10.4f} | {precision_score(y_test, y_pred_k):<10.4f} | {recall_score(y_test, y_pred_k):<10.4f} | {f1_score(y_test, y_pred_k):<10.4f}")
    
    # 1. Grid Search
    param_grid = {'n_neighbors': np.arange(1, 22, 2), 'weights': ['uniform', 'distance']}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    
    t0 = time.time()
    grid_search.fit(X_train, y_train)
    grid_time = time.time() - t0
    print(f"\nGridSearchCV best params: {grid_search.best_params_} (Found in {grid_time:.2f}s)")
    
    # 2. Randomized Search
    random_search = RandomizedSearchCV(KNeighborsClassifier(), param_grid, n_iter=10, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)
    
    t0 = time.time()
    random_search.fit(X_train, y_train)
    random_time = time.time() - t0
    print(f"RandomizedSearchCV best params: {random_search.best_params_} (Found in {random_time:.2f}s)")
    
    # Best Model overall
    best_knn = grid_search.best_estimator_
    
    # 3. KDTree vs BallTree Comparison
    trees = {'KDTree': KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'], algorithm='kd_tree'),
             'BallTree': KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'], algorithm='ball_tree')}
    
    for name, tree_model in trees.items():
        t0 = time.time()
        tree_model.fit(X_train, y_train)
        train_time = time.time() - t0
        
        t1 = time.time()
        tree_model.predict(X_test)
        pred_time = time.time() - t1
        
        results[name] = {"Train Time (s)": train_time, "Predict Time (s)": pred_time}
    
    tree_df = pd.DataFrame(results).T
    print("\nTree Structure Timing Comparison:")
    print(tree_df.to_string())
    
    # Evaluate best KNN
    t0 = time.time()
    best_knn.fit(X_train, y_train)
    best_knn_train_time = time.time() - t0
    
    t1 = time.time()
    y_pred = best_knn.predict(X_test)
    y_prob = best_knn.predict_proba(X_test)[:, 1]
    best_knn_pred_time = time.time() - t1
    
    knn_final_results = {
        "Optimized KNN": {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_prob),
            "Train Time (s)": best_knn_train_time,
            "Predict Time (s)": best_knn_pred_time,
            "Model": best_knn,
            "y_pred": y_pred,
            "y_prob": y_prob
        }
    }
    
    # Extract Grid Search info for plotting
    grid_scores = pd.DataFrame(grid_search.cv_results_)
    random_scores = pd.DataFrame(random_search.cv_results_)
    
    return knn_final_results, best_knn, grid_scores, random_scores, k_vals, k_accs



# In[ ]:


def run_cross_validation(X, y, best_nb, best_knn, dataset_name):
    print(f"\n=== 5-Fold Cross Validation for {dataset_name} ===")
    
    # Re-evaluate best models using Cross Validation
    nb_scores = cross_val_score(best_nb, X, y, cv=5, scoring='accuracy')
    knn_scores = cross_val_score(best_knn, X, y, cv=5, scoring='accuracy')
    
    print(f"{'Fold':<10} | {'Naïve Bayes':<15} | {'Best KNN':<15}")
    print("-" * 45)
    for i in range(5):
        print(f"{i+1:<10} | {nb_scores[i]:<15.4f} | {knn_scores[i]:<15.4f}")
    print(f"{'Average':<10} | {nb_scores.mean():<15.4f} | {knn_scores.mean():<15.4f}")
    
    return {"NB_CV_Scores": nb_scores, "KNN_CV_Scores": knn_scores}



# In[ ]:


from sklearn.metrics import precision_recall_curve

def generate_evaluation_plots(nb_results, knn_results, grid_scores, random_scores, k_vals, k_accs, cv_results, y_test, dataset_name):
    print(f"\n=== Generating Evaluation Plots for {dataset_name} ===")
    
    # Accuracy vs k
    plt.figure(figsize=(6,4))
    plt.plot(k_vals, k_accs, marker='o', linestyle='-', color='b')
    plt.title('Accuracy vs k (KNN)')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 1. ROC Curves
    plt.figure(figsize=(8, 6))
    for name, res in nb_results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        plt.plot(fpr, tpr, label=f'{name} (AUC = {res["ROC-AUC"]:.2f})')
        
    knn_res = knn_results["Optimized KNN"]
    fpr, tpr, _ = roc_curve(y_test, knn_res["y_prob"])
    plt.plot(fpr, tpr, label=f'Optimized KNN (AUC = {knn_res["ROC-AUC"]:.2f})', linewidth=2, linestyle='--')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curves Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    
    # PR Curves
    plt.figure(figsize=(8, 6))
    for name, res in nb_results.items():
        p, r, _ = precision_recall_curve(y_test, res["y_prob"])
        plt.plot(r, p, label=f'{name}')
    p, r, _ = precision_recall_curve(y_test, knn_res["y_prob"])
    plt.plot(r, p, label='Optimized KNN', linewidth=2, linestyle='--')
    plt.title("Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 2. Confusion Matrices (2x2 Grid)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    
    all_models = list(nb_results.items()) + list(knn_results.items())
    for idx, (name, res) in enumerate(all_models):
        if idx >= 4: break # Only show up to 4
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[idx])
        axes[idx].set_title(f"{name} Confusion Matrix")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("True")
        
    plt.tight_layout()
    plt.show()
    
    # 3. GridSearchCV Heatmap (Accuracy vs k and weights)
    plt.figure(figsize=(8, 5))
    heatmap_data = grid_scores.pivot(index='param_weights', columns='param_n_neighbors', values='mean_test_score')
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt=".3f")
    plt.title("GridSearchCV Mean F1-Score (k vs weights)")
    plt.tight_layout()
    plt.show()
    
    # RandomizedSearchCV score distribution
    plt.figure(figsize=(6,4))
    sns.histplot(random_scores['mean_test_score'], bins=10, kde=True, color='purple')
    plt.title("RandomizedSearchCV Score Distribution")
    plt.xlabel("Mean Test F1-Score")
    plt.tight_layout()
    plt.show()
    
    # 4. Cross Validation Boxplot
    cv_df = pd.DataFrame({
        'Best Naïve Bayes': cv_results["NB_CV_Scores"],
        'Optimized KNN': cv_results["KNN_CV_Scores"]
    })
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=cv_df, palette='Set2')
    plt.title("5-Fold Cross Validation Accuracy Distribution")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()

    # Time Comparisons
    model_names = [name for name, _ in nb_results.items()] + ['Optimized KNN']
    train_times = [res['Train Time (s)'] for _, res in nb_results.items()] + [knn_res['Train Time (s)']]
    pred_times = [res['Predict Time (s)'] for _, res in nb_results.items()] + [knn_res['Predict Time (s)']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    sns.barplot(x=model_names, y=train_times, ax=ax1, palette='mako')
    ax1.set_title("Training Time Comparison")
    ax1.set_ylabel("Seconds")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    
    sns.barplot(x=model_names, y=pred_times, ax=ax2, palette='rocket')
    ax2.set_title("Prediction Time Comparison")
    ax2.set_ylabel("Seconds")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()



# In[ ]:


from sklearn.preprocessing import MinMaxScaler

def run_experiment_2(csv_path, dataset_name="Dataset", target_col=None, drop_cols=None):
    """
    Master pipeline that runs Experiment 2: loads data, runs EDA, 
    evaluates Naïve Bayes, optimizes KNN, validates, and plots results.
    """
    print(f"\n" + "="*50)
    print(f"=== LAUNCHING EXPERIMENT 2 PIPELINE: {dataset_name} ===")
    print("="*50)
    
    # Phase 1: Reusing Exp1 Pipeline
    data_dict = run_data_pipeline(
        data_source=csv_path, 
        dataset_name=dataset_name, 
        target_col=target_col, 
        drop_cols=drop_cols,
        k_features=57 
    )
    
    if not isinstance(data_dict, dict):
        return
        
    X_train, X_test = data_dict['X_train'], data_dict['X_test']
    y_train, y_test = data_dict['y_train'], data_dict['y_test']
    
    # Scale Data specifically for KNN
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Phase 2: Naïve Bayes Evaluation
    nb_results, best_nb_model = evaluate_naive_bayes(X_train_scaled, y_train, X_test_scaled, y_test, dataset_name)
    
    # Phase 3: KNN Optimization
    knn_results, best_knn_model, grid_scores, random_scores, k_vals, k_accs = optimize_knn(X_train_scaled, y_train, X_test_scaled, y_test, dataset_name)
    
    # Phase 4: Cross Validation and Plotting
    X_full = np.vstack((X_train_scaled, X_test_scaled))
    y_full = pd.concat([y_train, y_test]).reset_index(drop=True)
    
    cv_results = run_cross_validation(X_full, y_full, best_nb_model, best_knn_model, dataset_name)
    
    generate_evaluation_plots(nb_results, knn_results, grid_scores, random_scores, k_vals, k_accs, cv_results, y_test, dataset_name)
    
    print("\n=== EXPERIMENT 2 PIPELINE COMPLETE ===")



# In[ ]:


# Execute the Pipeline on the Ex2 Spambase Dataset
run_experiment_2(
    csv_path="Ex2/spambase_csv.csv", 
    dataset_name="Kaggle Spambase Dataset", 
    target_col="class"
)


# In[ ]:


# ==========================================
# EXPERIMENT 3: REGRESSION ANALYSIS
# ==========================================
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np

def evaluate_linear_regression(X_train, y_train, X_test, y_test, dataset_name):
    print(f"\n=== Evaluating Linear Regression for {dataset_name} ===")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics = {
        'Train MAE': mean_absolute_error(y_train, y_train_pred),
        'Train RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'Train R2': r2_score(y_train, y_train_pred),
        'Test MAE': mean_absolute_error(y_test, y_test_pred),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'Test R2': r2_score(y_test, y_test_pred)
    }
    
    print(f"{'Metric':<12} | {'Train':<10} | {'Test':<10}")
    print("-"*38)
    for m in ['MAE', 'RMSE', 'R2']:
        print(f"{m:<12} | {metrics['Train '+m]:<10.4f} | {metrics['Test '+m]:<10.4f}")
    
    return model, metrics

def optimize_regularized_models(X_train, y_train, X_test, y_test, dataset_name):
    print(f"\n=== Optimizing Regularized Models for {dataset_name} ===")
    
    # Search Spaces as per lab manual
    ridge_param = {'alpha': [0.01, 0.1, 1, 10, 100]}
    lasso_param = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
    elastic_param = {'alpha': [0.01, 0.1, 1, 10], 'l1_ratio': [0.2, 0.5, 0.8]}
    
    best_models = {}
    test_metrics = {}
    
    models = {
        'Ridge': (Ridge(), ridge_param),
        'Lasso': (Lasso(), lasso_param),
        'ElasticNet': (ElasticNet(), elastic_param)
    }
    
    for name, (model, params) in models.items():
        grid = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        best_models[name] = best_model
        
        y_pred = best_model.predict(X_test)
        test_metrics[name] = {
            'Best Params': grid.best_params_,
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred)
        }
        
        print(f"{name:<11} | Best Params: {str(grid.best_params_):<45} | Test R2: {test_metrics[name]['R2']:.4f}")
        
    return best_models, test_metrics


# In[ ]:


from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

def run_regression_cross_validation(X, y, models_dict):
    print("\n=== 5-Fold Cross Validation for Regression Models ===")
    
    for name, model in models_dict.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
        print(f"{name:<15} | CV R2 Mean: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
def generate_regression_plots(y_test, predictions_dict, test_metrics, dataset_name):
    print(f"\n=== Generating Evaluation Plots for {dataset_name} ===")
    
    # 1. Bar Plot for Test R2 Scores
    models = list(test_metrics.keys())
    r2_scores = [test_metrics[m]['R2'] for m in models]
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=models, y=r2_scores, palette='viridis')
    plt.title(f"{dataset_name}: Test R² Score Comparison")
    plt.ylabel("R² Score")
    # plt.ylim(0, 1) # Optional: comment out because R2 can technically be negative if worse than baseline
    plt.show()
    
    # 2. Scatter plots of Actual vs Predicted
    num_models = len(models)
    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 5))
    if num_models == 1: axes = [axes]
    
    for ax, name in zip(axes, models):
        sns.scatterplot(x=y_test, y=predictions_dict[name], ax=ax, alpha=0.6)
        # Perfect prediction line
        min_val = min(y_test.min(), predictions_dict[name].min())
        max_val = max(y_test.max(), predictions_dict[name].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        ax.set_title(f"{name}: Actual vs Predicted")
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        
    plt.tight_layout()
    plt.show()


# In[ ]:


def run_experiment_3(dataset_path, target_col):
    print("="*50)
    print(f"=== LAUNCHING EXPERIMENT 3 PIPELINE: {dataset_path} ===")
    print("="*50)
    
    # 1. Data Loading & Preprocessing
    results = run_data_pipeline(dataset_path, target_col=target_col, dataset_name='Loan Dataset')
    X_train, X_test = results['X_train'], results['X_test']
    y_train, y_test = results['y_train'], results['y_test']
    
    # 2. Linear Regression Evaluation
    lr_model, lr_metrics = evaluate_linear_regression(X_train, y_train, X_test, y_test, "Loan Dataset")
    
    # 3. Regularized Models Optimization
    best_models, test_metrics = optimize_regularized_models(X_train, y_train, X_test, y_test, "Loan Dataset")
    
    # 4. Combine all models for CV and plots
    all_models = {'LinearRegression': lr_model}
    all_models.update(best_models)
    
    all_metrics = {'LinearRegression': {'R2': lr_metrics['Test R2']}}
    all_metrics.update(test_metrics)
    
    # 5. Cross Validation
    run_regression_cross_validation(X_train, y_train, all_models)
    
    # 6. Evaluation Plots
    predictions_dict = {}
    for name, model in all_models.items():
        predictions_dict[name] = model.predict(X_test)
        
    generate_regression_plots(y_test, predictions_dict, all_metrics, "Loan Dataset")
    print("\n=== EXPERIMENT 3 PIPELINE COMPLETE ===")

# Execute Experiment 3
run_experiment_3("Datasets/loan_approval_dataset/loan_approval_dataset.csv", " loan_amount")


# In[ ]:


def evaluate_baseline_classifiers(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import StandardScaler
    import time
    
    # Scale features for SVM and Logistic Regression
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n=== Logistic Regression Performance ===")
    print(f"{'Metric':<20} | {'Value':<10}")
    print("-" * 35)
    
    # Train Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    start_time = time.time()
    lr.fit(X_train_scaled, y_train)
    lr_time = time.time() - start_time
    
    y_pred_lr = lr.predict(X_test_scaled)
    print(f"{'Accuracy':<20} | {accuracy_score(y_test, y_pred_lr):.4f}")
    print(f"{'Precision':<20} | {precision_score(y_test, y_pred_lr):.4f}")
    print(f"{'Recall':<20} | {recall_score(y_test, y_pred_lr):.4f}")
    print(f"{'F1 Score':<20} | {f1_score(y_test, y_pred_lr):.4f}")
    print(f"{'Training Time (s)':<20} | {lr_time:.4f}")
    
    print("\n=== SVM Kernel-wise Performance ===")
    print(f"{'Kernel':<15} | {'Accuracy':<10} | {'F1 Score':<10} | {'Training Time (s)':<15}")
    print("-" * 60)
    
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    for k in kernels:
        svm = SVC(kernel=k, random_state=42)
        start_time = time.time()
        svm.fit(X_train_scaled, y_train)
        svm_time = time.time() - start_time
        
        y_pred_svm = svm.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred_svm)
        f1 = f1_score(y_test, y_pred_svm)
        
        print(f"{k.capitalize():<15} | {acc:<10.4f} | {f1:<10.4f} | {svm_time:<15.4f}")
        
    return X_train_scaled, X_test_scaled


# In[ ]:


def optimize_classification_models(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.metrics import accuracy_score
    
    print("\n=== Hyperparameter Tuning Results ===")
    print(f"{'Model':<20} | {'Search Method':<15} | {'Best Parameters':<65} | {'Test Accuracy'}")
    print("-" * 120)
    
    # 1. Tune Logistic Regression
    lr_param_grid = [
        {'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear']},
        {'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100], 'solver': ['saga']}
    ]
    lr = LogisticRegression(max_iter=1000, random_state=42)
    # RandomizedSearchCV for speed
    lr_search = RandomizedSearchCV(lr, lr_param_grid, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
    lr_search.fit(X_train, y_train)
    
    best_lr = lr_search.best_estimator_
    lr_test_acc = accuracy_score(y_test, best_lr.predict(X_test))
    lr_params_str = str(lr_search.best_params_)
    print(f"{'Logistic Regression':<20} | {'Randomized':<15} | {lr_params_str:<65} | {lr_test_acc:.4f}")
    
    # 2. Tune SVM
    svm_param_grid = [
        {'kernel': ['linear', 'rbf', 'sigmoid'], 'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto']},
        {'kernel': ['poly'], 'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto'], 'degree': [2, 3, 4]}
    ]
    svm = SVC(random_state=42)
    # RandomizedSearchCV for SVM is highly recommended to save time in lab environments
    svm_search = RandomizedSearchCV(svm, svm_param_grid, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
    svm_search.fit(X_train, y_train)
    
    best_svm = svm_search.best_estimator_
    svm_test_acc = accuracy_score(y_test, best_svm.predict(X_test))
    svm_params_str = str(svm_search.best_params_)
    print(f"{'SVM':<20} | {'Randomized':<15} | {svm_params_str:<65} | {svm_test_acc:.4f}")
    
    return best_lr, best_svm


# In[ ]:


def run_classification_cross_validation(X, y, best_lr, best_svm):
    from sklearn.model_selection import cross_validate
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    print("\n=== 5-Fold Cross Validation for Best Models ===")
    
    # Scale the full X for cross-validation
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X)
    
    scoring = {'accuracy': 'accuracy', 'f1': 'f1'}
    
    lr_cv = cross_validate(best_lr, X_scaled, y, cv=5, scoring=scoring)
    svm_cv = cross_validate(best_svm, X_scaled, y, cv=5, scoring=scoring)
    
    print(f"{'Model':<20} | {'CV Accuracy':<25} | {'CV F1 Score':<25}")
    print("-" * 75)
    
    print(f"{'Logistic Regression':<20} | {np.mean(lr_cv['test_accuracy']):.4f} (+/- {np.std(lr_cv['test_accuracy']):.4f}) | {np.mean(lr_cv['test_f1']):.4f} (+/- {np.std(lr_cv['test_f1']):.4f})")
    print(f"{'SVM':<20} | {np.mean(svm_cv['test_accuracy']):.4f} (+/- {np.std(svm_cv['test_accuracy']):.4f}) | {np.mean(svm_cv['test_f1']):.4f} (+/- {np.std(svm_cv['test_f1']):.4f})")
    
    print("\n=== EXPERIMENT 4 PIPELINE COMPLETE ===")


# In[ ]:


def run_experiment_4(dataset_path="Ex2/spambase_csv.csv", target_col="class"):
    print("="*50)
    print(f"=== LAUNCHING EXPERIMENT 4 PIPELINE: {dataset_path} ===")
    print("="*50)
    
    # Phase 1: Pipeline Hookup
    results = run_data_pipeline(dataset_path, target_col=target_col, dataset_name="Experiment 4 Spambase")
    X_train, X_test = results['X_train'], results['X_test']
    y_train, y_test = results['y_train'], results['y_test']
    
    # Reconstruct X and y for Cross-Validation
    import pandas as pd
    X = pd.concat([X_train, results['X_val'], X_test])
    y = pd.concat([y_train, results['y_val'], y_test])
    
    print("\nPhase 1: Data loaded successfully for Experiment 4!")
    
    # Phase 2: Baseline Models
    X_train_scaled, X_test_scaled = evaluate_baseline_classifiers(X_train, y_train, X_test, y_test)
    
    # Phase 3: Hyperparameter Optimization
    best_lr, best_svm = optimize_classification_models(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Phase 4: Cross-Validation & Execution
    run_classification_cross_validation(X, y, best_lr, best_svm)
    
run_experiment_4()

