import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)

class HealthcarePreprocessor:
    """
    Handles training/inference preprocessing pipelines including imputation, encoding, and scaling.
    """
    
    BINARY_COLS = ["Gender", "Ever_Married", "Residence_Type"]
    NUM_COLS = ["Age", "Avg_Glucose_Level", "BMI"]
    MULTICLASS_COLS = ["Work_Type", "Smoking_Status"]
    
    def __init__(self):
        self.median_bmi = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.columns_after_dummies = []
        self.is_fit = False
        
    def fit(self, X: pd.DataFrame) -> "HealthcarePreprocessor":
        """
        Fits the preprocessor on the training dataset.
        """
        logger.info("Fitting HealthcarePreprocessor...")
        
        # 1. Compute median BMI for imputation
        self.median_bmi = X["BMI"].median()
        if pd.isna(self.median_bmi):
            self.median_bmi = 28.0 # default fallback
        logger.info(f"Median BMI calculated for imputation: {self.median_bmi}")
        
        # Temporary copy for fitting encoders
        X_tmp = X.copy()
        X_tmp["BMI"] = X_tmp["BMI"].fillna(self.median_bmi)
        
        # 2. Fit label encoders for binary categories
        for col in self.BINARY_COLS:
            le = LabelEncoder()
            X_tmp[col] = le.fit_transform(X_tmp[col].astype(str))
            self.label_encoders[col] = le
            logger.info(f"Fit LabelEncoder for {col}: classes = {le.classes_}")
            
        # 3. Fit scaler on numerical features
        self.scaler.fit(X_tmp[self.NUM_COLS])
        logger.info(f"Fit StandardScaler for numerical features: {self.NUM_COLS}")
        X_tmp[self.NUM_COLS] = self.scaler.transform(X_tmp[self.NUM_COLS])
        
        # 4. Perform One-Hot encoding to retrieve column structure
        X_dummy = pd.get_dummies(X_tmp, columns=self.MULTICLASS_COLS, drop_first=True, dtype=int)
        self.columns_after_dummies = list(X_dummy.columns)
        
        self.is_fit = True
        logger.info("Preprocessor fit successfully.")
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms a dataset using the fitted preprocessor.
        """
        if not self.is_fit:
            raise ValueError("Preprocessor has not been fitted yet. Call fit() first.")
            
        X_out = X.copy()
        
        # 1. Impute missing BMI
        X_out["BMI"] = X_out["BMI"].fillna(self.median_bmi)
        
        # 2. Apply label encoders
        for col in self.BINARY_COLS:
            le = self.label_encoders[col]
            # Handle unseen categories gracefully
            X_out[col] = X_out[col].map(lambda s: s if s in le.classes_ else le.classes_[0])
            X_out[col] = le.transform(X_out[col].astype(str))
            
        # 3. Apply standard scaler
        X_out[self.NUM_COLS] = self.scaler.transform(X_out[self.NUM_COLS])
        
        # 4. Apply One-Hot Encoding
        X_out = pd.get_dummies(X_out, columns=self.MULTICLASS_COLS, drop_first=True, dtype=int)
        
        # 5. Realign columns to match the fitted columns (for missing categories during inference)
        for col in self.columns_after_dummies:
            if col not in X_out.columns:
                X_out[col] = 0 # fill missing one-hot columns with 0
                
        # Reorder columns to match fit shape exactly
        X_out = X_out[self.columns_after_dummies]
        
        return X_out
        
    def save(self, file_path: str) -> None:
        """
        Saves the fitted preprocessor to a file.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self, file_path)
        logger.info(f"Preprocessor saved to {file_path}")
        
    @staticmethod
    def load(file_path: str) -> "HealthcarePreprocessor":
        """
        Loads a preprocessor from a file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Preprocessor file not found at {file_path}")
        preprocessor = joblib.load(file_path)
        logger.info(f"Preprocessor loaded from {file_path}")
        return preprocessor
