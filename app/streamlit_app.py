import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Set page config
st.set_page_config(
    page_title="Healthcare Analytics: Patient Stroke Risk Platform",
    page_icon="🏥",
    layout="wide"
)

# Custom Styling (using Streamlit markdown for CSS injects)
st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }
    .stButton>button {
        background-color: #1a73e8;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
        box-shadow: 0 4px 6px rgba(50, 50, 93, 0.11), 0 1px 3px rgba(0, 0, 0, 0.08);
        transition: all 0.15s ease;
    }
    .stButton>button:hover {
        background-color: #1557b0;
        transform: translateY(-1px);
        box-shadow: 0 7px 14px rgba(50, 50, 93, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08);
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #1a73e8;
        margin-bottom: 1rem;
    }
    .risk-high {
        background-color: #fce8e6;
        border-left: 5px solid #d93025;
        padding: 1.5rem;
        border-radius: 12px;
        color: #c5221f;
        margin-bottom: 1rem;
    }
    .risk-medium {
        background-color: #fef7e0;
        border-left: 5px solid #f9ab00;
        padding: 1.5rem;
        border-radius: 12px;
        color: #b06000;
        margin-bottom: 1rem;
    }
    .risk-low {
        background-color: #e6f4ea;
        border-left: 5px solid #137333;
        padding: 1.5rem;
        border-radius: 12px;
        color: #137333;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model assets
@st.cache_resource
def load_assets():
    # Find paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "models", "best_model.pkl")
    preprocessor_path = os.path.join(project_root, "models", "preprocessor.pkl")
    dataset_path = os.path.join(project_root, "data", "healthcare_dataset.csv")
    
    model = joblib.load(model_path)
    # Load custom preprocessor module (needed since python needs to import it)
    import sys
    sys.path.append(os.path.join(project_root, "src"))
    from preprocessing import HealthcarePreprocessor
    preprocessor = HealthcarePreprocessor.load(preprocessor_path)
    
    df = pd.read_csv(dataset_path)
    return model, preprocessor, df

try:
    model, preprocessor, df = load_assets()
    assets_loaded = True
except Exception as e:
    st.error(f"Error loading model assets: {str(e)}. Please run `src/train_model.py` first.")
    assets_loaded = False

# Layout Structure
st.title("🏥 Patient Stroke Risk & Healthcare Analytics Platform")
st.markdown("Predict stroke probability and compare patient clinical metrics against the clinical population database.")

if assets_loaded:
    # --- SIDEBAR: Patient Demographics & Background ---
    st.sidebar.header("Demographics & Administrative")
    
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    age = st.sidebar.slider("Age (Years)", 1, 100, 45)
    ever_married = st.sidebar.selectbox("Ever Married", ["Yes", "No"])
    residence = st.sidebar.selectbox("Residence Type", ["Urban", "Rural"])
    
    work_type = st.sidebar.selectbox(
        "Employment Type",
        ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"]
    )
    
    insurance = st.sidebar.selectbox(
        "Insurance Provider",
        ["Medicare", "Medicaid", "Private", "None"]
    )
    
    admission_type = st.sidebar.selectbox(
        "Admitting Case Type",
        ["Emergency", "Urgent", "Elective"]
    )
    
    # --- MAIN CONTENT: Patient Clinical Inputs ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Clinical Biomarkers & History")
        
        with st.form("patient_clinical_form"):
            form_col1, form_col2 = st.columns(2)
            
            with form_col1:
                hypertension = st.checkbox("Hypertension Diagnosed", value=False)
                heart_disease = st.checkbox("Heart Disease Diagnosed", value=False)
                smoking_status = st.selectbox(
                    "Smoking Profile",
                    ["never smoked", "formerly smoked", "smokes", "Unknown"]
                )
                
            with form_col2:
                glucose = st.number_input(
                    "Avg Glucose Level (mg/dL)", 
                    min_value=50.0, 
                    max_value=300.0, 
                    value=105.0, 
                    step=1.0,
                    help="Normal fasting glucose: < 100 mg/dL. Diabetic levels: > 125 mg/dL."
                )
                
                bmi = st.number_input(
                    "Body Mass Index (BMI)", 
                    min_value=10.0, 
                    max_value=60.0, 
                    value=26.5, 
                    step=0.1,
                    help="Healthy range: 18.5 - 24.9. Obese: 30 or higher."
                )
                
            submit_button = st.form_submit_button("Assess Stroke Risk")
            
        if submit_button:
            # Prepare data row
            patient_data = pd.DataFrame({
                "Age": [age],
                "Gender": [gender],
                "Hypertension": [1 if hypertension else 0],
                "Heart_Disease": [1 if heart_disease else 0],
                "Ever_Married": [ever_married],
                "Work_Type": [work_type],
                "Residence_Type": [residence],
                "Avg_Glucose_Level": [glucose],
                "BMI": [bmi],
                "Smoking_Status": [smoking_status]
            })
            
            # Preprocess using loaded pipeline
            patient_processed = preprocessor.transform(patient_data)
            
            # Predict
            prob = model.predict_proba(patient_processed)[0, 1]
            prob_percent = prob * 100
            
            # Display Prediction Card
            st.subheader("Risk Assessment Report")
            
            if prob_percent >= 50:
                card_class = "risk-high"
                risk_level = "HIGH RISK"
                advice = "Patient displays critical indicators for stroke. Clinical observation and further diagnostic imaging (CT/MRI) is strongly advised. Immediate review of cardiovascular profile is indicated."
            elif prob_percent >= 20:
                card_class = "risk-medium"
                risk_level = "MODERATE RISK"
                advice = "Patient displays elevated indicators. Recommend regular monitoring of blood glucose, diet adjustments, blood pressure checks, and management of secondary risk factors."
            else:
                card_class = "risk-low"
                risk_level = "LOW RISK"
                advice = "Patient clinical markers fall within normal/low risk distributions. Maintain routine health screenings and healthy lifestyle parameters."
                
            st.markdown(f"""
            <div class="{card_class}">
                <h3><strong>{risk_level}</strong></h3>
                <h1><strong>{prob_percent:.2f}%</strong> Probability of Stroke</h1>
                <p style="margin-top: 10px; font-size: 1.1rem;"><strong>Clinical Recommendation:</strong> {advice}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Population Comparison Chart
            st.subheader("Clinical Benchmark Comparisons")
            st.markdown("Compare the patient's metrics (orange line) against the distribution of existing patients in our healthcare database.")
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # Age distribution
            sns.kdeplot(data=df, x="Age", ax=axes[0], fill=True, color="#1a73e8")
            axes[0].axvline(age, color="#d93025", lw=3, label=f"Patient: {age} yrs")
            axes[0].set_title("Age Distribution")
            axes[0].legend()
            
            # Glucose distribution
            sns.kdeplot(data=df, x="Avg_Glucose_Level", ax=axes[1], fill=True, color="#137333")
            axes[1].axvline(glucose, color="#d93025", lw=3, label=f"Patient: {glucose} mg/dL")
            axes[1].set_title("Glucose level Distribution")
            axes[1].legend()
            
            # BMI distribution
            sns.kdeplot(data=df, x="BMI", ax=axes[2], fill=True, color="#f9ab00")
            axes[2].axvline(bmi, color="#d93025", lw=3, label=f"Patient: {bmi}")
            axes[2].set_title("BMI Distribution")
            axes[2].legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
    with col2:
        st.subheader("Platform Summary & Data Insights")
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Database Population</h4>
            <h2>{len(df):,} Patients</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid #137333;">
            <h4>Average Age</h4>
            <h2>{df['Age'].mean():.1f} Years</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid #f9ab00;">
            <h4>Average Length of Stay</h4>
            <h2>{df['Length_of_Stay'].mean():.1f} Days</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid #d93025;">
            <h4>Stroke Incidence Rate</h4>
            <h2>{df['Stroke'].mean() * 100:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Display short metadata
        st.info(
            "This model is a Logistic Regression classifier calibrated on a clinical health dataset. "
            "Model parameters are checked against missing indicators and cross-validated."
        )
