import os
import pandas as pd
import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HealthcareDataLoader:
    """
    Handles loading and validation of the Healthcare Analytics dataset.
    """
    
    REQUIRED_COLUMNS = [
        "Patient_ID", "Age", "Gender", "Hypertension", "Heart_Disease", 
        "Ever_Married", "Work_Type", "Residence_Type", "Avg_Glucose_Level", 
        "BMI", "Smoking_Status", "Admission_Date", "Admission_Type", 
        "Medical_Condition", "Insurance_Provider", "Length_of_Stay", 
        "Billing_Amount", "Risk_Category", "Stroke"
    ]
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    def load_data(self) -> pd.DataFrame:
        """
        Loads the dataset from CSV and runs schema validation.
        
        Returns:
            pd.DataFrame: Loaded and validated dataframe.
        """
        if not os.path.exists(self.file_path):
            error_msg = f"Data file not found at path: {os.path.abspath(self.file_path)}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            logger.info(f"Loading healthcare data from: {self.file_path}")
            df = pd.read_csv(self.file_path)
            self.validate_schema(df)
            logger.info(f"Successfully loaded data. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading healthcare data: {str(e)}")
            raise e
            
    def validate_schema(self, df: pd.DataFrame) -> None:
        """
        Validates that all required columns are present in the dataframe.
        
        Args:
            df (pd.DataFrame): Dataframe to validate.
        """
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            error_msg = f"Data integrity check failed. Missing columns: {missing_cols}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.info("Schema validation successful. All required columns are present.")
