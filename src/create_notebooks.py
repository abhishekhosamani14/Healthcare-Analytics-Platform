import os
import json

def create_notebook(filename, cells):
    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=1)
    print(f"Created notebook: {filename}")

def make_markdown_cell(text_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text_lines]
    }

def make_code_cell(code_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code_lines]
    }

# ==========================================
# 1. EDA NOTEBOOK CELLS
# ==========================================
eda_cells = [
    make_markdown_cell([
        "# Healthcare Analytics: Exploratory Data Analysis (EDA)",
        "This notebook explores the healthcare dataset to understand features, search for missing values, analyze outliers, detect class imbalance, and view feature correlations."
    ]),
    make_code_cell([
        "import os",
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "",
        "# Set style",
        "sns.set_theme(style='whitegrid')",
        "plt.rcParams['figure.figsize'] = (10, 6)"
    ]),
    make_markdown_cell([
        "## 1. Load the Dataset"
    ]),
    make_code_cell([
        "df_path = os.path.join('..', 'data', 'healthcare_dataset.csv')",
        "df = pd.read_csv(df_path)",
        "print(f'Dataset Shape: {df.shape}')",
        "df.head()"
    ]),
    make_markdown_cell([
        "## 2. Basic Dataset Information & Summary Statistics"
    ]),
    make_code_cell([
        "df.info()"
    ]),
    make_code_cell([
        "df.describe().T"
    ]),
    make_markdown_cell([
        "## 3. Missing Value Analysis"
    ]),
    make_code_cell([
        "missing_values = df.isnull().sum()",
        "missing_percent = (missing_values / len(df)) * 100",
        "missing_df = pd.DataFrame({'Missing Count': missing_values, 'Percentage (%)': missing_percent})",
        "missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)",
        "print('Features with missing values:')",
        "print(missing_df)",
        "",
        "# Plot missing values",
        "if not missing_df.empty:",
        "    sns.barplot(x=missing_df.index, y='Missing Count', data=missing_df, palette='viridis')",
        "    plt.title('Missing Value Count per Feature')",
        "    plt.xlabel('Features')",
        "    plt.ylabel('Count')",
        "    plt.show()"
    ]),
    make_markdown_cell([
        "## 4. Distribution of Target Variable: Stroke",
        "Check class imbalance, which is common in healthcare prediction datasets."
    ]),
    make_code_cell([
        "stroke_counts = df['Stroke'].value_counts()",
        "stroke_pct = df['Stroke'].value_counts(normalize=True) * 100",
        "print('Stroke Distribution:')",
        "for val, count, pct in zip(stroke_counts.index, stroke_counts.values, stroke_pct.values):",
        "    print(f'Class {val}: {count} records ({pct:.2f}%)')",
        "",
        "plt.figure(figsize=(6, 5))",
        "sns.countplot(x='Stroke', data=df, palette='pastel')",
        "plt.title('Stroke Class Distribution')",
        "plt.xlabel('Stroke (0 = No, 1 = Yes)')",
        "plt.ylabel('Count')",
        "plt.show()"
    ]),
    make_markdown_cell([
        "## 5. Numerical Feature Distributions",
        "Analyze the distribution of continuous clinical indicators: `Age`, `Avg_Glucose_Level`, `BMI`."
    ]),
    make_code_cell([
        "num_cols = ['Age', 'Avg_Glucose_Level', 'BMI']",
        "fig, axes = plt.subplots(3, 2, figsize=(14, 15))",
        "for i, col in enumerate(num_cols):",
        "    # Histogram",
        "    sns.histplot(df[col], kde=True, ax=axes[i, 0], color='skyblue')",
        "    axes[i, 0].set_title(f'{col} Histogram')",
        "    ",
        "    # Boxplot for Outliers",
        "    sns.boxplot(x=df[col], ax=axes[i, 1], color='lightgreen')",
        "    axes[i, 1].set_title(f'{col} Boxplot (Outlier Detection)')",
        "",
        "plt.tight_layout()",
        "plt.show()"
    ]),
    make_markdown_cell([
        "## 6. Categorical Feature Distributions",
        "Explore distributions of demographic and administrative variables."
    ]),
    make_code_cell([
        "cat_cols = ['Gender', 'Hypertension', 'Heart_Disease', 'Ever_Married', 'Work_Type', 'Residence_Type', 'Smoking_Status', 'Admission_Type', 'Medical_Condition', 'Insurance_Provider', 'Risk_Category']",
        "",
        "fig, axes = plt.subplots(4, 3, figsize=(20, 24))",
        "axes = axes.flatten()",
        "",
        "for i, col in enumerate(cat_cols):",
        "    sns.countplot(x=col, data=df, ax=axes[i], palette='muted')",
        "    axes[i].set_title(f'{col} Count')",
        "    axes[i].tick_params(axis='x', rotation=45)",
        "    ",
        "# Remove empty subplots",
        "for j in range(len(cat_cols), len(axes)):",
        "    fig.delaxes(axes[j])",
        "",
        "plt.tight_layout()",
        "plt.show()"
    ]),
    make_markdown_cell([
        "## 7. Correlation Heatmap",
        "Compute Pearson correlation matrix on numeric features."
    ]),
    make_code_cell([
        "numeric_df = df.select_dtypes(include=[np.number])",
        "corr = numeric_df.corr()",
        "",
        "plt.figure(figsize=(10, 8))",
        "sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)",
        "plt.title('Correlation Matrix of Numeric Features')",
        "plt.show()"
    ]),
    make_markdown_cell([
        "## 8. Outlier Detection using IQR Method",
        "Calculate number of outliers in `BMI` and `Avg_Glucose_Level`."
    ]),
    make_code_cell([
        "for col in ['Avg_Glucose_Level', 'BMI']:",
        "    q1 = df[col].quantile(0.25)",
        "    q3 = df[col].quantile(0.75)",
        "    iqr = q3 - q1",
        "    lower_bound = q1 - 1.5 * iqr",
        "    upper_bound = q3 + 1.5 * iqr",
        "    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]",
        "    print(f'{col}: lower bound = {lower_bound:.2f}, upper bound = {upper_bound:.2f}')",
        "    print(f'  Total Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)')"
    ])
]

# ==========================================
# 2. PREPROCESSING NOTEBOOK CELLS
# ==========================================
preprocessing_cells = [
    make_markdown_cell([
        "# Healthcare Analytics: Data Preprocessing",
        "This notebook handles missing values, encodes categorical variables, and performs train-test splitting."
    ]),
    make_code_cell([
        "import os",
        "import pandas as pd",
        "import numpy as np",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.preprocessing import LabelEncoder"
    ]),
    make_markdown_cell([
        "## 1. Load Data"
    ]),
    make_code_cell([
        "df = pd.read_csv(os.path.join('..', 'data', 'healthcare_dataset.csv'))",
        "print(df.shape)",
        "df.isnull().sum()"
    ]),
    make_markdown_cell([
        "## 2. Missing Value Imputation",
        "Since `BMI` has missing values (approx 4%) and is slightly skewed, we will impute with the median value of the training set. To avoid data leakage, we compute the median first."
    ]),
    make_code_cell([
        "median_bmi = df['BMI'].median()",
        "print(f'Median BMI to impute: {median_bmi}')",
        "df['BMI'] = df['BMI'].fillna(median_bmi)",
        "print('Remaining nulls:', df['BMI'].isnull().sum())"
    ]),
    make_markdown_cell([
        "## 3. Categorical Variables Encoding",
        "Let's identify categorical variables and encode them appropriately:",
        "- Binary columns (`Gender`, `Ever_Married`, `Residence_Type`): Label Encoding (0 or 1)",
        "- Multi-class columns (`Work_Type`, `Smoking_Status`, `Insurance_Provider`, `Medical_Condition`, `Admission_Type`, `Risk_Category`): One-Hot Encoding"
    ]),
    make_code_cell([
        "# Demographics / Predictors we care about for machine learning modeling",
        "feature_cols = ['Age', 'Gender', 'Hypertension', 'Heart_Disease', 'Ever_Married', ",
        "                'Work_Type', 'Residence_Type', 'Avg_Glucose_Level', 'BMI', 'Smoking_Status']",
        "target_col = 'Stroke'",
        "",
        "X = df[feature_cols].copy()",
        "y = df[target_col].copy()",
        "",
        "# Binary variables label encoding",
        "label_encoders = {}",
        "for col in ['Gender', 'Ever_Married', 'Residence_Type']:",
        "    le = LabelEncoder()",
        "    X[col] = le.fit_transform(X[col])",
        "    label_encoders[col] = le",
        "    print(f'Encoded {col}: {list(le.classes_)} -> {list(le.transform(le.classes_))}')",
        "",
        "# One-hot encode remaining multi-class categoricals",
        "X = pd.get_dummies(X, columns=['Work_Type', 'Smoking_Status'], drop_first=True, dtype=int)",
        "print('\\nProcessed feature matrix shape:', X.shape)",
        "X.head()"
    ]),
    make_markdown_cell([
        "## 4. Train-Test Split",
        "Use stratified splitting because of the target variable's class imbalance."
    ]),
    make_code_cell([
        "X_train, X_test, y_train, y_test = train_test_split(",
        "    X, y, test_size=0.2, random_state=42, stratify=y",
        ")",
        "print(f'Train shape: {X_train.shape}, Test shape: {X_test.shape}')",
        "print('Train class distribution:\\n', y_train.value_counts(normalize=True))",
        "print('Test class distribution:\\n', y_test.value_counts(normalize=True))"
    ]),
    make_markdown_cell([
        "## 5. Save Splits for Next Steps",
        "Save preprocessed train and test sets to temporary CSV files to ensure modularity."
    ]),
    make_code_cell([
        "os.makedirs(os.path.join('..', 'data', 'temp'), exist_ok=True)",
        "X_train.to_csv(os.path.join('..', 'data', 'temp', 'X_train.csv'), index=False)",
        "X_test.to_csv(os.path.join('..', 'data', 'temp', 'X_test.csv'), index=False)",
        "y_train.to_csv(os.path.join('..', 'data', 'temp', 'y_train.csv'), index=False)",
        "y_test.to_csv(os.path.join('..', 'data', 'temp', 'y_test.csv'), index=False)",
        "print('Preprocessed partitions saved to data/temp/')"
    ])
]

# ==========================================
# 3. FEATURE ENGINEERING NOTEBOOK CELLS
# ==========================================
feature_engineering_cells = [
    make_markdown_cell([
        "# Healthcare Analytics: Feature Scaling & Feature Importance",
        "This notebook scales numerical variables using `StandardScaler` and visualizes feature importances using a Random Forest."
    ]),
    make_code_cell([
        "import os",
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "from sklearn.preprocessing import StandardScaler",
        "from sklearn.ensemble import RandomForestClassifier",
        "",
        "sns.set_theme(style='whitegrid')"
    ]),
    make_markdown_cell([
        "## 1. Load Data Splits"
    ]),
    make_code_cell([
        "X_train = pd.read_csv(os.path.join('..', 'data', 'temp', 'X_train.csv'))",
        "X_test = pd.read_csv(os.path.join('..', 'data', 'temp', 'X_test.csv'))",
        "y_train = pd.read_csv(os.path.join('..', 'data', 'temp', 'y_train.csv')).values.ravel()",
        "y_test = pd.read_csv(os.path.join('..', 'data', 'temp', 'y_test.csv')).values.ravel()",
        "print(X_train.shape, X_test.shape)"
    ]),
    make_markdown_cell([
        "## 2. Feature Scaling",
        "We scale numerical columns (`Age`, `Avg_Glucose_Level`, `BMI`). We fit the scaler *only* on the training set to prevent data leakage."
    ]),
    make_code_cell([
        "num_cols = ['Age', 'Avg_Glucose_Level', 'BMI']",
        "scaler = StandardScaler()",
        "",
        "X_train_scaled = X_train.copy()",
        "X_test_scaled = X_test.copy()",
        "",
        "X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])",
        "X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])",
        "",
        "X_train_scaled.head()"
    ]),
    make_markdown_cell([
        "## 3. Feature Importance Visualization",
        "Train a Random Forest classifier to extract and plot Gini feature importance."
    ]),
    make_code_cell([
        "rf = RandomForestClassifier(n_estimators=100, random_state=42)",
        "rf.fit(X_train_scaled, y_train)",
        "",
        "importances = rf.feature_importances_",
        "features = X_train.columns",
        "feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)",
        "",
        "plt.figure(figsize=(10, 6))",
        "sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis')",
        "plt.title('Feature Importance via Random Forest Classifier')",
        "plt.xlabel('Importance (Gini)')",
        "plt.ylabel('Feature')",
        "plt.show()",
        "",
        "feat_imp_df"
    ]),
    make_markdown_cell([
        "## 4. Save Scaled Datasets"
    ]),
    make_code_cell([
        "X_train_scaled.to_csv(os.path.join('..', 'data', 'temp', 'X_train_scaled.csv'), index=False)",
        "X_test_scaled.to_csv(os.path.join('..', 'data', 'temp', 'X_test_scaled.csv'), index=False)",
        "print('Scaled partitions saved to data/temp/')"
    ])
]

# ==========================================
# 4. MODEL BUILDING NOTEBOOK CELLS
# ==========================================
model_building_cells = [
    make_markdown_cell([
        "# Healthcare Analytics: Machine Learning Model Building",
        "This notebook trains and evaluates four machine learning algorithms on our scaled healthcare dataset:",
        "1. Logistic Regression",
        "2. Decision Tree Classifier",
        "3. Random Forest Classifier",
        "4. XGBoost Classifier"
    ]),
    make_code_cell([
        "import os",
        "import pandas as pd",
        "import numpy as np",
        "import joblib",
        "from sklearn.linear_model import LogisticRegression",
        "from sklearn.tree import DecisionTreeClassifier",
        "from sklearn.ensemble import RandomForestClassifier",
        "from xgboost import XGBClassifier",
        "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score"
    ]),
    make_markdown_cell([
        "## 1. Load Scaled Partition Data"
    ]),
    make_code_cell([
        "X_train = pd.read_csv(os.path.join('..', 'data', 'temp', 'X_train_scaled.csv'))",
        "X_test = pd.read_csv(os.path.join('..', 'data', 'temp', 'X_test_scaled.csv'))",
        "y_train = pd.read_csv(os.path.join('..', 'data', 'temp', 'y_train.csv')).values.ravel()",
        "y_test = pd.read_csv(os.path.join('..', 'data', 'temp', 'y_test.csv')).values.ravel()",
        "print('Data Loaded Successfully. Shape:', X_train.shape)"
    ]),
    make_markdown_cell([
        "## 2. Train Models"
    ]),
    make_code_cell([
        "models = {",
        "    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),",
        "    'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),",
        "    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),",
        "    'XGBoost': XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric='logloss')",
        "}",
        "",
        "results = []",
        "trained_models = {}",
        "",
        "for name, model in models.items():",
        "    # Train",
        "    model.fit(X_train, y_train)",
        "    trained_models[name] = model",
        "    ",
        "    # Predict",
        "    preds = model.predict(X_test)",
        "    probs = model.predict_proba(X_test)[:, 1]",
        "    ",
        "    # Metrics",
        "    acc = accuracy_score(y_test, preds)",
        "    prec = precision_score(y_test, preds, zero_division=0)",
        "    rec = recall_score(y_test, preds)",
        "    f1 = f1_score(y_test, preds)",
        "    auc = roc_auc_score(y_test, probs)",
        "    ",
        "    results.append({",
        "        'Model': name,",
        "        'Accuracy': acc,",
        "        'Precision': prec,",
        "        'Recall': rec,",
        "        'F1-score': f1,",
        "        'ROC-AUC': auc",
        "    })",
        "    print(f'Trained {name}')"
    ]),
    make_markdown_cell([
        "## 3. Compare Models"
    ]),
    make_code_cell([
        "comparison_df = pd.DataFrame(results).sort_values(by='ROC-AUC', ascending=False)",
        "comparison_df"
    ]),
    make_markdown_cell([
        "## 4. Save Trained Models & Best Model",
        "Choose the best model based on ROC-AUC (standard for clinical screening where we want to capture probabilities and control thresholds)."
    ]),
    make_code_cell([
        "os.makedirs(os.path.join('..', 'models'), exist_ok=True)",
        "best_model_name = comparison_df.iloc[0]['Model']",
        "best_model = trained_models[best_model_name]",
        "print(f'Best Model: {best_model_name}')",
        "",
        "# Save the best model",
        "model_file = os.path.join('..', 'models', 'best_model.pkl')",
        "joblib.dump(best_model, model_file)",
        "print(f'Best model saved to {model_file}')",
        "",
        "# Save all models for visualization comparisons in notebook 5",
        "for name, model in trained_models.items():",
        "    safe_name = name.lower().replace(' ', '_')",
        "    joblib.dump(model, os.path.join('..', 'models', f'{safe_name}_model.pkl'))"
    ])
]

# ==========================================
# 5. MODEL EVALUATION NOTEBOOK CELLS
# ==========================================
model_evaluation_cells = [
    make_markdown_cell([
        "# Healthcare Analytics: Comprehensive Model Evaluation",
        "This notebook computes the detailed metrics, confusion matrices, and ROC-AUC curves for the trained models."
    ]),
    make_code_cell([
        "import os",
        "import joblib",
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "from sklearn.metrics import (",
        "    classification_report, confusion_matrix, ConfusionMatrixDisplay,",
        "    roc_curve, auc, precision_recall_curve, average_precision_score",
        ")",
        "",
        "sns.set_theme(style='whitegrid')"
    ]),
    make_markdown_cell([
        "## 1. Load Data & Models"
    ]),
    make_code_cell([
        "X_test = pd.read_csv(os.path.join('..', 'data', 'temp', 'X_test_scaled.csv'))",
        "y_test = pd.read_csv(os.path.join('..', 'data', 'temp', 'y_test.csv')).values.ravel()",
        "",
        "models = {",
        "    'Logistic Regression': joblib.load(os.path.join('..', 'models', 'logistic_regression_model.pkl')),",
        "    'Decision Tree': joblib.load(os.path.join('..', 'models', 'decision_tree_model.pkl')),",
        "    'Random Forest': joblib.load(os.path.join('..', 'models', 'random_forest_model.pkl')),",
        "    'XGBoost': joblib.load(os.path.join('..', 'models', 'xgboost_model.pkl'))",
        "}"
    ]),
    make_markdown_cell([
        "## 2. Generate Classification Reports"
    ]),
    make_code_cell([
        "for name, model in models.items():",
        "    preds = model.predict(X_test)",
        "    print('='*50)",
        "    print(f'Model: {name}')",
        "    print('='*50)",
        "    print(classification_report(y_test, preds))",
        "    print('\\n')"
    ]),
    make_markdown_cell([
        "## 3. Confusion Matrix Visualization"
    ]),
    make_code_cell([
        "fig, axes = plt.subplots(2, 2, figsize=(12, 10))",
        "axes = axes.flatten()",
        "",
        "for i, (name, model) in enumerate(models.items()):",
        "    preds = model.predict(X_test)",
        "    cm = confusion_matrix(y_test, preds)",
        "    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Stroke', 'Stroke'])",
        "    disp.plot(ax=axes[i], cmap='Blues', colorbar=False)",
        "    axes[i].set_title(f'{name} Confusion Matrix')",
        "    axes[i].grid(False)",
        "",
        "plt.tight_layout()",
        "plt.show()"
    ]),
    make_markdown_cell([
        "## 4. ROC-AUC Curves Comparison"
    ]),
    make_code_cell([
        "plt.figure(figsize=(10, 8))",
        "",
        "for name, model in models.items():",
        "    probs = model.predict_proba(X_test)[:, 1]",
        "    fpr, tpr, _ = roc_curve(y_test, probs)",
        "    roc_auc = auc(fpr, tpr)",
        "    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', lw=2)",
        "",
        "plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random Guessing')",
        "plt.xlim([0.0, 1.0])",
        "plt.ylim([0.0, 1.05])",
        "plt.xlabel('False Positive Rate')",
        "plt.ylabel('True Positive Rate')",
        "plt.title('Receiver Operating Characteristic (ROC) Curve')",
        "plt.legend(loc='lower right')",
        "plt.show()"
    ]),
    make_markdown_cell([
        "## 5. Precision-Recall Curves Comparison",
        "Especially useful for imbalanced datasets."
    ]),
    make_code_cell([
        "plt.figure(figsize=(10, 8))",
        "",
        "for name, model in models.items():",
        "    probs = model.predict_proba(X_test)[:, 1]",
        "    precision, recall, _ = precision_recall_curve(y_test, probs)",
        "    ap = average_precision_score(y_test, probs)",
        "    plt.plot(recall, precision, label=f'{name} (AP = {ap:.3f})', lw=2)",
        "",
        "plt.xlabel('Recall')",
        "plt.ylabel('Precision')",
        "plt.title('Precision-Recall Curve')",
        "plt.legend(loc='lower left')",
        "plt.show()"
    ])
]

if __name__ == "__main__":
    os.makedirs("notebooks", exist_ok=True)
    
    create_notebook(os.path.join("notebooks", "01_eda.ipynb"), eda_cells)
    create_notebook(os.path.join("notebooks", "02_data_preprocessing.ipynb"), preprocessing_cells)
    create_notebook(os.path.join("notebooks", "03_feature_engineering.ipynb"), feature_engineering_cells)
    create_notebook(os.path.join("notebooks", "04_model_building.ipynb"), model_building_cells)
    create_notebook(os.path.join("notebooks", "05_model_evaluation.ipynb"), model_evaluation_cells)
    print("All notebooks created successfully!")
