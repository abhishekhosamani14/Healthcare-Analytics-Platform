import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_healthcare_data(num_samples=10000):
    print(f"Generating {num_samples} patient records...")
    
    # 1. Base Patient Demographics
    patient_ids = [f"PT{10000 + i}" for i in range(num_samples)]
    
    # Age: Normal-ish distribution skewed older since it's a healthcare dataset
    ages = np.clip(np.random.normal(loc=54, scale=18, size=num_samples).astype(int), 1, 95)
    
    genders = np.random.choice(["Male", "Female"], size=num_samples, p=[0.48, 0.52])
    
    # Ever Married: Highly correlated with age
    ever_married = []
    for age in ages:
        if age < 22:
            ever_married.append("No")
        elif age < 30:
            ever_married.append(random.choice(["Yes", "No"]))
        else:
            ever_married.append(np.random.choice(["Yes", "No"], p=[0.85, 0.15]))
            
    # Work Type: Correlated with age
    work_types = []
    for age in ages:
        if age < 16:
            work_types.append("Children")
        elif age > 65:
            work_types.append(np.random.choice(["Self-employed", "Private", "Govt_job"], p=[0.4, 0.4, 0.2]))
        else:
            work_types.append(np.random.choice(["Private", "Self-employed", "Govt_job", "Never_worked"], p=[0.65, 0.20, 0.14, 0.01]))
            
    residence_types = np.random.choice(["Urban", "Rural"], size=num_samples, p=[0.51, 0.49])
    
    # 2. Clinical Indicators
    # Hypertension: Risk increases with age
    hypertension = []
    for age in ages:
        prob = 0.05 + 0.007 * age # 5% baseline, up to ~70% at age 95
        prob = min(max(prob, 0.02), 0.75)
        hypertension.append(np.random.choice([1, 0], p=[prob, 1 - prob]))
    hypertension = np.array(hypertension)
    
    # Heart Disease: Risk increases with age and hypertension
    heart_disease = []
    for i in range(num_samples):
        age = ages[i]
        ht = hypertension[i]
        prob = 0.02 + 0.005 * age + (0.15 if ht == 1 else 0.0)
        prob = min(max(prob, 0.01), 0.60)
        heart_disease.append(np.random.choice([1, 0], p=[prob, 1 - prob]))
    heart_disease = np.array(heart_disease)
    
    # Avg Glucose Level: Log-normal distribution, higher risk of diabetes with age/hypertension
    avg_glucose = []
    for i in range(num_samples):
        age = ages[i]
        ht = hypertension[i]
        # Base glucose around 90-100, standard deviation
        base = 90 + (0.2 * age) + (15 if ht == 1 else 0)
        # 15% diabetic outlier probability
        is_diabetic = np.random.choice([True, False], p=[0.12, 0.88])
        if is_diabetic:
            glucose = np.random.uniform(140, 275)
        else:
            glucose = np.random.normal(base, 15)
        avg_glucose.append(np.clip(glucose, 55, 280))
    avg_glucose = np.array(avg_glucose)
    
    # BMI: Normal distribution centered around overweight/obese (common in clinical datasets)
    bmis = []
    for i in range(num_samples):
        age = ages[i]
        # Children have lower BMI, older adults slightly higher
        if age < 16:
            mean_bmi = 18 + (age * 0.4)
        else:
            mean_bmi = 27 + (0.05 * age)
        bmi = np.random.normal(mean_bmi, 6)
        # Add random missing values to BMI (approx 4%) to practice data imputation
        if random.random() < 0.04:
            bmis.append(np.nan)
        else:
            bmis.append(np.clip(bmi, 14.0, 58.0))
    bmis = np.array(bmis)
    
    # Smoking Status
    smoking_status = np.random.choice(
        ["never smoked", "formerly smoked", "smokes", "Unknown"], 
        size=num_samples, 
        p=[0.38, 0.17, 0.15, 0.30]
    )
    
    # 3. Target Variable: Stroke
    # Risk calculation using a logistic function based on real stroke risk factors
    stroke = []
    for i in range(num_samples):
        age = ages[i]
        ht = hypertension[i]
        hd = heart_disease[i]
        gluc = avg_glucose[i]
        bmi = bmis[i] if not np.isnan(bmis[i]) else 28.0
        smoke = smoking_status[i]
        
        # Logistic formula coefficients (manual weights)
        logit = -6.5 + (0.055 * age) + (0.85 * ht) + (1.1 * hd) + (0.008 * gluc) + (0.02 * bmi)
        if smoke == "smokes":
            logit += 0.5
        elif smoke == "formerly smoked":
            logit += 0.25
        elif smoke == "Unknown":
            logit -= 0.1
            
        prob = 1 / (1 + np.exp(-logit))
        # Add slight background noise
        prob = np.clip(prob, 0.005, 0.95)
        stroke.append(np.random.choice([1, 0], p=[prob, 1 - prob]))
    stroke = np.array(stroke)
    
    # 4. Administrative Data (Power BI dashboard enrichment)
    # Admission Date: uniform dates over last 18 months
    start_date = datetime(2025, 1, 1)
    admission_dates = [start_date + timedelta(days=random.randint(0, 500)) for _ in range(num_samples)]
    admission_dates_str = [d.strftime("%Y-%m-%d") for d in admission_dates]
    
    # Admission Type: Stroke cases are highly likely to be Emergency
    admission_types = []
    for i in range(num_samples):
        st = stroke[i]
        if st == 1:
            admission_types.append(np.random.choice(["Emergency", "Urgent"], p=[0.90, 0.10]))
        else:
            admission_types.append(np.random.choice(["Emergency", "Urgent", "Elective"], p=[0.30, 0.30, 0.40]))
            
    # Medical Condition
    medical_conditions = []
    for i in range(num_samples):
        st = stroke[i]
        hd = heart_disease[i]
        if st == 1:
            medical_conditions.append(np.random.choice(["Neurology", "Cardiology"], p=[0.85, 0.15]))
        elif hd == 1:
            medical_conditions.append(np.random.choice(["Cardiology", "General"], p=[0.80, 0.20]))
        else:
            medical_conditions.append(np.random.choice(
                ["Cardiology", "Oncology", "Neurology", "Gastroenterology", "General"],
                p=[0.15, 0.15, 0.10, 0.20, 0.40]
            ))
            
    # Insurance Provider
    insurance_providers = np.random.choice(["Medicare", "Medicaid", "Private", "None"], size=num_samples, p=[0.35, 0.20, 0.38, 0.07])
    
    # Length of Stay: Stroke and Emergency admissions are longer
    length_of_stay = []
    for i in range(num_samples):
        st = stroke[i]
        adm = admission_types[i]
        cond = medical_conditions[i]
        
        base_days = 2
        if adm == "Emergency":
            base_days += 3
        elif adm == "Urgent":
            base_days += 1
            
        if st == 1:
            base_days += np.random.poisson(lam=7)
        elif cond == "Oncology":
            base_days += np.random.poisson(lam=6)
        elif cond == "Cardiology":
            base_days += np.random.poisson(lam=4)
        else:
            base_days += np.random.poisson(lam=2)
            
        length_of_stay.append(int(np.clip(base_days, 1, 30)))
        
    # Billing Amount: Length of Stay * Cost Per Day + random complexity charge
    billing_amounts = []
    for i in range(num_samples):
        los = length_of_stay[i]
        cond = medical_conditions[i]
        adm = admission_types[i]
        
        cost_per_day = 1200
        if cond == "Oncology":
            cost_per_day = 1800
        elif cond == "Cardiology" or cond == "Neurology":
            cost_per_day = 1500
            
        if adm == "Emergency":
            complexity_charge = np.random.uniform(2000, 8000)
        else:
            complexity_charge = np.random.uniform(500, 3000)
            
        total_billing = (los * cost_per_day) + complexity_charge
        billing_amounts.append(round(total_billing, 2))
        
    # Risk Category (Computed Business Rule)
    risk_categories = []
    for i in range(num_samples):
        score = 0
        if ages[i] > 60: score += 2
        elif ages[i] > 40: score += 1
        if hypertension[i] == 1: score += 2
        if heart_disease[i] == 1: score += 2
        if avg_glucose[i] > 140: score += 2
        if bmis[i] > 30: score += 1
        if smoking_status[i] in ["smokes", "formerly smoked"]: score += 1
        
        if score >= 6:
            risk_categories.append("High")
        elif score >= 3:
            risk_categories.append("Medium")
        else:
            risk_categories.append("Low")
            
    # Combine into DataFrame
    df = pd.DataFrame({
        "Patient_ID": patient_ids,
        "Age": ages,
        "Gender": genders,
        "Hypertension": hypertension,
        "Heart_Disease": heart_disease,
        "Ever_Married": ever_married,
        "Work_Type": work_types,
        "Residence_Type": residence_types,
        "Avg_Glucose_Level": avg_glucose,
        "BMI": bmis,
        "Smoking_Status": smoking_status,
        "Admission_Date": admission_dates_str,
        "Admission_Type": admission_types,
        "Medical_Condition": medical_conditions,
        "Insurance_Provider": insurance_providers,
        "Length_of_Stay": length_of_stay,
        "Billing_Amount": billing_amounts,
        "Risk_Category": risk_categories,
        "Stroke": stroke
    })
    
    print(f"Data generation complete. Dataset shape: {df.shape}")
    print(f"Stroke cases: {df['Stroke'].sum()} ({df['Stroke'].mean()*100:.2f}%)")
    return df

if __name__ == "__main__":
    # Ensure data/ directory exists
    os.makedirs("data", exist_ok=True)
    
    df = generate_healthcare_data()
    output_path = os.path.join("data", "healthcare_dataset.csv")
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to: {os.path.abspath(output_path)}")
