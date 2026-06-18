import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import project modules
from data_loader import HealthcareDataLoader
from preprocessing import HealthcarePreprocessor
from utils import setup_directories, save_confusion_matrix, save_roc_curve, save_feature_importance

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_training_pipeline(data_path: str, models_dir: str, reports_dir: str):
    # 1. Setup Directories
    setup_directories([models_dir, reports_dir])
    
    # 2. Load Data
    loader = HealthcareDataLoader(data_path)
    df = loader.load_data()
    
    # 3. Extract Features and Target
    # Demographics and clinical markers used for predicting Stroke risk
    feature_cols = [
        "Age", "Gender", "Hypertension", "Heart_Disease", "Ever_Married", 
        "Work_Type", "Residence_Type", "Avg_Glucose_Level", "BMI", "Smoking_Status"
    ]
    target_col = "Stroke"
    
    X = df[feature_cols]
    y = df[target_col]
    
    # 4. Train-Test Split (Stratified to handle class balance)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Split completed. Training samples: {len(X_train_raw)}, Test samples: {len(X_test_raw)}")
    
    # 5. Fit Preprocessor on Training Features & Save
    preprocessor = HealthcarePreprocessor()
    preprocessor.fit(X_train_raw)
    
    preprocessor_path = os.path.join(models_dir, "preprocessor.pkl")
    preprocessor.save(preprocessor_path)
    
    # 6. Transform Features
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    logger.info(f"Preprocessed features shape: {X_train.shape}")
    
    # 7. Define Models
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision_Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric="logloss")
    }
    
    results = []
    trained_models = {}
    
    # 8. Train and Evaluate Each Model
    for name, model in models.items():
        logger.info(f"Training model: {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]
        
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_probs)
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1,
            "ROC_AUC": roc_auc
        })
        
        # Generate diagnostic plots for this model
        save_confusion_matrix(
            y_test, y_pred, 
            os.path.join(reports_dir, f"{name}_confusion_matrix.png"), 
            name.replace("_", " ")
        )
        save_roc_curve(
            y_test, y_probs, 
            os.path.join(reports_dir, f"{name}_roc_curve.png"), 
            name.replace("_", " ")
        )
        
        # Save feature importance if model supports it
        if hasattr(model, "feature_importances_"):
            save_feature_importance(
                model.feature_importances_, 
                X_train.columns, 
                os.path.join(reports_dir, f"{name}_feature_importance.png"), 
                name.replace("_", " ")
            )
        elif name == "Logistic_Regression":
            # Use coefficients as importance for Logistic Regression
            import numpy as np
            coefs = np.abs(model.coef_[0])
            coefs_normalized = coefs / np.sum(coefs)
            save_feature_importance(
                coefs_normalized, 
                X_train.columns, 
                os.path.join(reports_dir, f"{name}_feature_importance.png"), 
                name.replace("_", " ")
            )
            
    # 9. Compare and Report Results
    results_df = pd.DataFrame(results).sort_values(by="ROC_AUC", ascending=False)
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON (Sorted by ROC-AUC)")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80 + "\n")
    
    # Save performance summary report
    report_csv = os.path.join(reports_dir, "model_comparison_report.csv")
    results_df.to_csv(report_csv, index=False)
    logger.info(f"Performance report saved to {report_csv}")
    
    # 10. Select & Save Best Model
    best_model_name = results_df.iloc[0]["Model"]
    best_roc_auc = results_df.iloc[0]["ROC_AUC"]
    best_model = trained_models[best_model_name]
    
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(best_model, best_model_path)
    logger.info(f"Selected Best Model: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")
    logger.info(f"Saved best model to {best_model_path}")
    
if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "healthcare_dataset.csv")
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
    
    run_training_pipeline(DATA_PATH, MODELS_DIR, REPORTS_DIR)
