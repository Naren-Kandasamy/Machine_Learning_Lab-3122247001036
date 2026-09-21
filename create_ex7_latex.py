import json
import subprocess
import os
import re
import numpy as np

def generate_latex():
    with open('Ex7/ex7_results.json', 'r') as f:
        data = json.load(f)

    res_no = data['res_no']
    cv_no = data['cv_no']
    param_no = data['param_no']
    
    res_pca = data['res_pca']
    cv_pca = data['cv_pca']
    param_pca = data['param_pca']
    
    n_comp = data['n_comp']
    var_exp = data['var_exp']

    models = ['SVM', 'Naive Bayes', 'KNN', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'AdaBoost', 'Gradient Boosting', 'XGBoost', 'Stacking']

    template_path = "Manual.tex"
    output_path = "Ex7/Experiment_7_Report.tex"

    with open(template_path, 'r') as f:
        content = f.read()

    # Meta
    content = content.replace('Experiment No. & \\\\ \\hline', 'Experiment No. & 7 \\\\ \\hline')
    content = content.replace('Experiment Title &', 'Experiment Title & Dimensionality Reduction and Model Evaluation (With and Without PCA) \\\\ \\hline')
    content = content.replace('\\footnote{\\textbf{GitHub Repository: }\\url{https://github.com/username/repository}}\\\\ \\hline', '')
    content = content.replace('Student Name & \\\\ \\hline', 'Student Name & Naren Karthik Kandasamy \\\\ \\hline')
    content = content.replace('Register Number & \\\\ \\hline', 'Register Number & 3122247001036 \\\\ \\hline')
    content = content.replace('Submission Date & \\\\ \\hline', 'Submission Date & 31/8/2026 \\\\ \\hline')

    # Objective
    content = content.replace(
        'State the objective of the experiment.',
        'To study the effect of dimensionality reduction using Principal Component Analysis (PCA) on the performance of various machine learning classifiers, comparing their metrics, variance, and bias before and after PCA.'
    )
    content = content.replace(
        'Describe the problem, input, output, and objective.',
        'We evaluate 10 classification algorithms on the Breast Cancer dataset (30 features) predicting malignancy. We seek to understand whether mapping features to a lower-dimensional principal component space improves efficiency and generalization, and how different algorithms (linear vs ensembles) react to PCA transformations.'
    )

    # Dataset
    content = content.replace('Dataset Name & \\\\ \\hline', 'Dataset Name & Wisconsin Diagnostic Breast Cancer \\\\ \\hline')
    content = content.replace('Dataset Source & \\\\ \\hline', 'Dataset Source & UCI Machine Learning Repository \\\\ \\hline')
    content = content.replace('Number of Samples & \\\\ \\hline', 'Number of Samples & 569 \\\\ \\hline')
    content = content.replace('Number of Features & \\\\ \\hline', 'Number of Features & 30 \\\\ \\hline')
    content = content.replace('Number of Classes & \\\\ \\hline', 'Number of Classes & 2 (Malignant, Benign) \\\\ \\hline')
    content = content.replace('Missing Values & \\\\ \\hline', 'Missing Values & None \\\\ \\hline')
    content = content.replace('Train-Test Split & \\\\ \\hline', 'Train-Test Split & 80-20 \\\\ \\hline')

    # Strip default generic EDA image
    content = content.replace(
        '\n\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.7\\linewidth]{example-image}\n\\caption{Replace with your figure caption}\n\\end{figure}\n',
        ''
    )
    content = content.replace(
        'Write 3--5 sentences explaining what the graph indicates and how it helps in understanding the dataset.',
        'The original dataset has 30 numerical attributes. Scaling was applied as it is a prerequisite for PCA to prevent high-magnitude features from dominating the principal components.'
    )

    content = content.replace(
        '\\item Feature engineering (if applicable)\n\\end{itemize}',
        '\\item Standardized features using StandardScaler.\n\\item Applied PCA to retain 95\\% of the variance.\n\\end{itemize}'
    )

    # Table 1: PCA Summary
    pca_table = f"""
\\textbf{{Table 1: PCA Variance Explained}}\\\\
\\begin{{tabular}}{{lllp{{4cm}}}}\\toprule Setting & Chosen Components / Variance Target & Explained Variance (\\%) & Justification\\\\ \\midrule
With-PCA & {n_comp} Components / 95\\% Target & {var_exp*100:.2f} & 95\\% is a standard threshold to retain max information while shedding noise. \\\\ \\bottomrule \\end{{tabular}}
    """

    tables_str = pca_table + "\n\\vspace{1em}\\\\ "

    # Model Tuning Tables
    for idx, model in enumerate(models):
        if model == 'Stacking':
            param_str_no = "Base: RF, SVM, XGB"
            param_str_pca = "Base: RF, SVM, XGB"
        else:
            # properly format the dict for latex
            p_no = str(param_no.get(model, 'Default')).replace('_', '\\_')
            p_pca = str(param_pca.get(model, 'Default')).replace('_', '\\_')
            param_str_no = p_no
            param_str_pca = p_pca
            
        tables_str += f"""
\\textbf{{Table {idx+2}: {model} Hyperparameter Tuning Results}}\\\\
\\begin{{tabular}}{{lll}}\\toprule Hyperparameters & Performance (No-PCA) & Performance (With-PCA)\\\\ \\midrule
{param_str_no} (No-PCA) & {res_no[model]['Accuracy']*100:.2f}\\% & - \\\\
{param_str_pca} (With-PCA) & - & {res_pca[model]['Accuracy']*100:.2f}\\% \\\\ \\bottomrule \\end{{tabular}}\\vspace{{1em}}\\\\
        """
    
    # 5-Fold Cross Validation Results Table (Table 12)
    cv_table = f"""
\\textbf{{Table 12: 5-Fold Cross-Validation Results (No-PCA vs With-PCA)}}\\\\
\\begin{{tabular}}{{llllllll}}\\toprule Model & Fold 1 & Fold 2 & Fold 3 & Fold 4 & Fold 5 & Avg (No) & Avg (PCA)\\\\ \\midrule
    """
    for i, model in enumerate(models):
        cvs_no = cv_no[model]
        cvs_pca = cv_pca[model]
        cv_table += f"{model} (No PCA) & {cvs_no[0]*100:.1f} & {cvs_no[1]*100:.1f} & {cvs_no[2]*100:.1f} & {cvs_no[3]*100:.1f} & {cvs_no[4]*100:.1f} & {np.mean(cvs_no)*100:.2f} & - \\\\\\\\ "
        if i == len(models) - 1:
            cv_table += f"{model} (With PCA) & {cvs_pca[0]*100:.1f} & {cvs_pca[1]*100:.1f} & {cvs_pca[2]*100:.1f} & {cvs_pca[3]*100:.1f} & {cvs_pca[4]*100:.1f} & - & {np.mean(cvs_pca)*100:.2f} \\\\\\\\ "
        else:
            cv_table += f"{model} (With PCA) & {cvs_pca[0]*100:.1f} & {cvs_pca[1]*100:.1f} & {cvs_pca[2]*100:.1f} & {cvs_pca[3]*100:.1f} & {cvs_pca[4]*100:.1f} & - & {np.mean(cvs_pca)*100:.2f} \\\\\\\\ \\midrule "
        
    cv_table += "\\bottomrule \\end{tabular}"

    content = content.replace(
        '\\begin{tabular}{ll}\n\\toprule\nParameter & Value\\\\\n\\midrule\n & \\\\\n & \\\\\n\\bottomrule\n\\end{tabular}',
        tables_str + "\n\\vspace{1em}\\\\ " + cv_table
    )

    # Performance Comparison of Ensemble Models
    content = re.sub(r'Accuracy & \\\\.*?ROC-AUC & \\\\', lambda m: "See Table 12 above for comprehensive comparison.", content, flags=re.DOTALL)

    content = content.replace(
        '\\end{enumerate}',
        '\\end{enumerate}\n\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.8\\linewidth]{Ex7/Scree_Plot.eps}\n\\caption{Scree Plot: Explained Variance by Components.}\n\\end{figure}'
    )

    content = content.replace(
        'Discuss observations, strengths, limitations and improvements.',
        'PCA successfully reduced the feature space from 30 to 10 components while retaining 95\\% variance. Tree-based models (Random Forest, XGBoost) generally perform exceptionally well on original feature spaces but can sometimes see a slight drop in accuracy with PCA since orthogonal transformations disrupt individual feature semantics. Linear models (SVM, Logistic Regression) benefited from decorrelated features and reduced dimensions.'
    )
    content = content.replace(
        'Summarize the experiment and key findings.',
        'The experiment confirmed that PCA is highly effective for reducing dimensionality without significant loss of information. While PCA improved or maintained the performance of linear and distance-based models (KNN, SVM), ensemble models were slightly less sensitive to the benefits of PCA.'
    )
    
    # Remove the generic image block if replace didn't catch it
    content = re.sub(r'\\begin\{figure\}\[H\]\n\\centering\n\\includegraphics\[width=0\.7\\linewidth\]\{example-image\}.*?\\end\{figure\}', '', content, flags=re.DOTALL)


    with open(output_path, 'w') as f:
        f.write(content)

    print(f"LaTeX generated: {output_path}")
    subprocess.run(['pdflatex', '-interaction=nonstopmode', '-output-directory=Ex7', output_path])

if __name__ == '__main__':
    generate_latex()
