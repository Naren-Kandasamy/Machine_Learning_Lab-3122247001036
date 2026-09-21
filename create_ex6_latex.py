import json
import re
import subprocess

def create_latex():
    with open('Ex6/ex6_results.json', 'r') as f:
        data = json.load(f)
        
    metrics = data['metrics']
    cv_results = data['cv_metrics']
    bagging_best = data['bagging_best']
    adaboost_best = data['adaboost_best']
    
    template_path = "Manual.tex"
    output_path = "Ex6/Experiment_6_Report.tex"
    
    with open(template_path, 'r') as f:
        content = f.read()
        
    content = content.replace('Experiment No. & \\\\ \\hline', 'Experiment No. & 6 \\\\ \\hline')
    content = content.replace('Experiment Title &', 'Experiment Title & Bagging, Boosting, and Stacked Ensemble Models \\\\ \\hline')
    content = content.replace('\\footnote{\\textbf{GitHub Repository: }\\url{https://github.com/username/repository}}\\\\ \\hline', '')
    content = content.replace('Student Name & \\\\ \\hline', 'Student Name & Naren Karthik Kandasamy \\\\ \\hline')
    content = content.replace('Register Number & \\\\ \\hline', 'Register Number & 3122247001036 \\\\ \\hline')
    content = content.replace('Submission Date & \\\\ \\hline', 'Submission Date & 31/8/2026 \\\\ \\hline')
    
    content = content.replace(
        'State the objective of the experiment.',
        'To understand and implement ensemble learning strategies including Bagging, Boosting, and Stacking. To compare these models in terms of accuracy, stability, and generalization, and analyze their effect on bias and variance.'
    )
    content = content.replace(
        'Describe the problem, input, output, and objective.',
        'The problem involves classifying tumor samples from the Wisconsin Diagnostic Breast Cancer dataset. We employ heterogeneous ensemble models to combine multiple base learners and theoretically compare how Bagging reduces variance while Boosting addresses model bias.'
    )
    
    content = content.replace('Dataset Name & \\\\ \\hline', 'Dataset Name & Wisconsin Diagnostic Breast Cancer \\\\ \\hline')
    content = content.replace('Dataset Source & \\\\ \\hline', 'Dataset Source & UCI Machine Learning Repository \\\\ \\hline')
    content = content.replace('Number of Samples & \\\\ \\hline', 'Number of Samples & 569 \\\\ \\hline')
    content = content.replace('Number of Features & \\\\ \\hline', 'Number of Features & 30 \\\\ \\hline')
    content = content.replace('Number of Classes & \\\\ \\hline', 'Number of Classes & 2 (Malignant, Benign) \\\\ \\hline')
    content = content.replace('Missing Values & \\\\ \\hline', 'Missing Values & None \\\\ \\hline')
    content = content.replace('Train-Test Split & \\\\ \\hline', 'Train-Test Split & 80-20 \\\\ \\hline')

    content = re.sub(r'\\begin\{figure\}\[H\]\n\\centering\n\\includegraphics\[width=0\.7\\linewidth\]\{example-image\}.*?\\end\{figure\}', '', content, flags=re.DOTALL)
    content = content.replace(
        'Write 3--5 sentences explaining what the graph indicates and how it helps in understanding the dataset.',
        'Exploratory Data Analysis was thoroughly conducted in the previous experiment. The data distributions and correlations remain identical, so we proceed directly to modeling.'
    )
    
    content = content.replace(
        '\\item Feature engineering (if applicable)\\n\\end{itemize}',
        '\\item Feature engineering (if applicable)\\n\\end{itemize}\\nStandard scaling was applied as models like SVM and Logistic Regression (meta-learner) are sensitive to feature scales.'
    )
    
    bagging_table = f"\\textbf{{Table 1: Bagging Hyperparameter Evaluation}}\\\\\\\\ \\begin{{tabular}}{{llll}}\\toprule n\\_estimators & max\\_samples & Avg CV Accuracy (\\%) & Avg CV F1 Score\\\\ \\midrule {bagging_best['n_estimators']} & {bagging_best['max_samples']} & {cv_results['Bagging']['CV_Accuracy']*100:.2f} & {cv_results['Bagging']['CV_F1']:.4f}\\\\ \\bottomrule \\end{{tabular}}"
    
    boosting_table = f"\\textbf{{Table 2: Boosting Hyperparameter Evaluation}}\\\\\\\\ \\begin{{tabular}}{{llll}}\\toprule n\\_estimators & learning\\_rate & Avg CV Accuracy (\\%) & Avg CV F1 Score\\\\ \\midrule {adaboost_best['n_estimators']} & {adaboost_best['learning_rate']} & {cv_results['AdaBoost']['CV_Accuracy']*100:.2f} & {cv_results['AdaBoost']['CV_F1']:.4f}\\\\ \\bottomrule \\end{{tabular}}"
    
    stacked_table = f"\\textbf{{Table 3: Stacked Ensemble Evaluation}}\\\\\\\\ \\begin{{tabular}}{{llll}}\\toprule Base Models & Meta Learner & Avg CV Accuracy (\\%) & Avg CV F1 Score\\\\ \\midrule DT, SVM, NB & Logistic Reg & {cv_results['Stacked Ensemble']['CV_Accuracy']*100:.2f} & {cv_results['Stacked Ensemble']['CV_F1']:.4f}\\\\ \\bottomrule \\end{{tabular}}"
    
    perf_table = f"\\textbf{{Table 4: Performance Comparison of Ensemble Models}}\\\\\\\\ \\begin{{tabular}}{{lllll}}\\toprule Model & Accuracy (\\%) & Precision & Recall & F1 Score\\\\ \\midrule Bagging & {metrics['Bagging']['Accuracy']*100:.2f} & {metrics['Bagging']['Precision']:.4f} & {metrics['Bagging']['Recall']:.4f} & {metrics['Bagging']['F1']:.4f}\\\\ AdaBoost & {metrics['AdaBoost']['Accuracy']*100:.2f} & {metrics['AdaBoost']['Precision']:.4f} & {metrics['AdaBoost']['Recall']:.4f} & {metrics['AdaBoost']['F1']:.4f}\\\\ Gradient Boosting & {metrics['Gradient Boosting']['Accuracy']*100:.2f} & {metrics['Gradient Boosting']['Precision']:.4f} & {metrics['Gradient Boosting']['Recall']:.4f} & {metrics['Gradient Boosting']['F1']:.4f}\\\\ Stacked Ensemble & {metrics['Stacked Ensemble']['Accuracy']*100:.2f} & {metrics['Stacked Ensemble']['Precision']:.4f} & {metrics['Stacked Ensemble']['Recall']:.4f} & {metrics['Stacked Ensemble']['F1']:.4f}\\\\ \\bottomrule \\end{{tabular}}"

    content = content.replace(
        '\\begin{tabular}{ll}\\n\\toprule\\nParameter & Value\\\\\\n\\midrule\\n & \\\\\\n & \\\\\\n\\bottomrule\\n\\end{tabular}',
        bagging_table + "\\vspace{1em}\\\\ " + boosting_table + "\\vspace{1em}\\\\ " + stacked_table
    )
    
    content = re.sub(r'Accuracy & \\\\.*?ROC-AUC & \\\\', lambda m: perf_table, content, flags=re.DOTALL)
    
    content = content.replace(
        '\\end{enumerate}',
        '\\end{enumerate}\n\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.8\\linewidth]{Ex6/ROC_Curves.eps}\n\\caption{ROC Curves for Ensemble Models.}\n\\end{figure}\n\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.8\\linewidth]{Ex6/Confusion_Matrices.eps}\n\\caption{Confusion Matrices comparing errors.}\n\\end{figure}'
    )
    
    content = content.replace(
        'Discuss observations, strengths, limitations and improvements.',
        'Bagging effectively reduced the variance of the base Decision Tree. Boosting models (AdaBoost and Gradient Boosting) improved upon bias by sequentially focusing on misclassified samples. The Stacked Ensemble leveraged diverse base estimators (SVM, Naive Bayes, Decision Tree) and optimally combined their predictive power using a Logistic Regression meta-learner.'
    )
    content = content.replace(
        'Summarize the experiment and key findings.',
        'Ensemble techniques consistently outperformed standalone estimators. Gradient Boosting and Stacking demonstrated superior generalization on the test set, exhibiting the lowest false negative rates, which is critical for cancer diagnosis. The experiment practically illustrated the theoretical trade-offs between variance reduction (Bagging) and bias reduction (Boosting).'
    )
    
    with open(output_path, 'w') as f:
        f.write(content)
        
    print(f"LaTeX generated: {output_path}")
    subprocess.run(['pdflatex', '-interaction=nonstopmode', '-output-directory=Ex6', output_path])

if __name__ == '__main__':
    create_latex()
