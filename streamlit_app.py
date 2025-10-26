import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import io
import base64
from datetime import datetime
import json
import os
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Gallstone Risk Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
        border-bottom: 3px solid #3498db;
        padding-bottom: 1rem;
    }
    
    .section-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #3498db;
        padding-left: 1rem;
        font-weight: 500;
    }
    
    .metric-card {
        color: #2c3e50;
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e1e8ed;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    .metric-card h4 {
        color: #2c3e50;
        margin-bottom: 1rem;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    .prediction-result {
        padding: 2rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1rem 0;
        border: 2px solid;
    }
    
    .critical-risk {
        background-color: #fff5f5;
        color: #c53030;
        border-color: #fc8181;
    }
    
    .high-risk {
        background-color: #fffaf0;
        color: #dd6b20;
        border-color: #f6ad55;
    }
    
    .moderate-risk {
        background-color: #fffff0;
        color: #d69e2e;
        border-color: #f6e05e;
    }
    
    .low-risk {
        background-color: #f0fff4;
        color: #38a169;
        border-color: #68d391;
    }
    
    .feature-group {
        color: #2c3e50;    
        background-color: #fafbfc;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e1e8ed;
        margin-bottom: 1rem;
    }
    
    .feature-group h4 {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
        font-size: 1rem;
        font-weight: 600;
    }
    
    .alert-box {
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .alert-warning {
        background-color: #fffbf0;
        border-color: #f6ad55;
        color: #744210;
    }
    
    .alert-error {
        background-color: #fff5f5;
        border-color: #fc8181;
        color: #742a2a;
    }
    
    .alert-success {
        background-color: #f0fff4;
        border-color: #68d391;
        color: #276749;
    }
    
    .alert-info {
        background-color: #ebf8ff;
        border-color: #63b3ed;
        color: #2a4365;
    }
    
    .sidebar-header {
        background-color: #3498db;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .recommendation-box {
        color: #2a4365;    
        background-color: #ebf8ff;
        border: 1px solid #63b3ed;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .recommendation-box h5 {
        color: #2a4365;
        margin-bottom: 0.5rem;
    }
    
    .stButton > button {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #2980b9;
    }
</style>
""", unsafe_allow_html=True)

class GallstonePredictor:
    def __init__(self):
        self.model = None
        self.explainer = None
        self.feature_names = None
        self.model_loaded = False
        self.error_message = None
        
    def load_model_file(self, file_path):
        """Load model with multiple methods"""
        try:
            # Try pickle first
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except:
            try:
                # Try joblib
                model = joblib.load(file_path)
                return model
            except Exception as e:
                raise Exception(f"Failed to load model: {str(e)}")
        
    def load_models(self):
        """Load model with comprehensive error handling"""
        model_paths = [
            'gallstone_xgb.pkl', 
            'gallstone_model.pkl',
            'model.pkl'
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    self.model = self.load_model_file(model_path)
                    
                    # Verify model has required methods
                    if not (hasattr(self.model, 'predict') and hasattr(self.model, 'predict_proba')):
                        continue
                        
                    self.model_loaded = True
                    st.success(f"Model loaded successfully from {model_path}")
                    
                    # Initialize SHAP explainer
                    self.initialize_shap()
                    return True
                    
                except Exception as e:
                    st.error(f"Error loading {model_path}: {str(e)}")
                    continue
        
        self.error_message = "No valid model file found. Please ensure you have a trained model (.pkl file) in the directory."
        return False
    
    def initialize_shap(self):
        """Initialize SHAP explainer safely"""
        try:
            self.explainer = shap.Explainer(self.model)
            st.success("SHAP explainer initialized")
        except:
            try:
                self.explainer = shap.TreeExplainer(self.model)
                st.success("SHAP TreeExplainer initialized")
            except:
                st.warning("SHAP explainer could not be initialized - explanations will be limited")
                self.explainer = None

def calculate_bmi(height, weight):
    """Calculate BMI with error handling"""
    try:
        if height <= 0 or weight <= 0:
            return 0.0
        height_m = height / 100
        bmi = weight / (height_m ** 2)
        return round(bmi, 2)
    except:
        return 0.0

def get_bmi_category(bmi):
    """Get BMI category"""
    if bmi == 0:
        return "Invalid BMI"
    elif bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal Weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    elif 30 <= bmi < 35:
        return "Obese Class I"
    elif 35 <= bmi < 40:
        return "Obese Class II"
    else:
        return "Obese Class III"

def get_risk_level_info(probability):
    """Get risk level information"""
    if probability >= 0.8:
        return {
            "level": "CRITICAL",
            "class": "critical-risk",
            "recommendation": "Immediate medical consultation recommended",
            "description": "Very high probability of gallstone presence"
        }
    elif probability >= 0.6:
        return {
            "level": "HIGH", 
            "class": "high-risk",
            "recommendation": "Schedule medical evaluation soon",
            "description": "Significant risk factors present"
        }
    elif probability >= 0.4:
        return {
            "level": "MODERATE",
            "class": "moderate-risk", 
            "recommendation": "Consider preventive measures and monitoring",
            "description": "Some risk factors identified"
        }
    else:
        return {
            "level": "LOW",
            "class": "low-risk",
            "recommendation": "Continue healthy lifestyle",
            "description": "Low risk based on current factors"
        }

def create_risk_gauge(probability):
    """Create risk probability gauge"""
    risk_info = get_risk_level_info(probability)
    
    # Determine gauge colors based on risk level
    if probability >= 0.8:
        gauge_color = "#c53030"
    elif probability >= 0.6:
        gauge_color = "#dd6b20"  
    elif probability >= 0.4:
        gauge_color = "#d69e2e"
    else:
        gauge_color = "#38a169"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Gallstone Risk Assessment (%)", 'font': {'color': "white",'size': 18}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': gauge_color, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e1e8ed",
            'steps': [
                {'range': [0, 40], 'color': '#f0fff4'},
                {'range': [40, 60], 'color': '#fffff0'},
                {'range': [60, 80], 'color': '#fffaf0'}, 
                {'range': [80, 100], 'color': '#fff5f5'}
            ],
            'threshold': {
                'line': {'color': "#c53030", 'width': 3},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(height=300, font={'color': "#2c3e50", 'family': "Arial"})
    return fig

def create_shap_plots(shap_values, feature_values, feature_names):
    """Create SHAP visualization plots"""
    try:
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_vals = shap_values
            
        # Create feature impact dataframe
        impact_df = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_vals,
            'feature_value': feature_values,
            'abs_impact': np.abs(shap_vals)
        }).sort_values('abs_impact', ascending=False).head(15)
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Feature Impact on Prediction', 'Feature Importance'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Impact plot (positive/negative)
        colors = ['#c53030' if x > 0 else '#38a169' for x in impact_df['shap_value']]
        fig.add_trace(
            go.Bar(
                x=impact_df['shap_value'],
                y=impact_df['feature'],
                orientation='h',
                marker_color=colors,
                text=[f"{val:+.3f}" for val in impact_df['shap_value']],
                textposition='outside',
                name="SHAP Impact"
            ),
            row=1, col=1
        )
        
        # Importance plot
        fig.add_trace(
            go.Bar(
                x=impact_df['abs_impact'],
                y=impact_df['feature'],
                orientation='h',
                marker_color='#3498db',
                text=[f"{val:.3f}" for val in impact_df['abs_impact']],
                textposition='outside',
                name="Absolute Impact"
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="SHAP Analysis - Feature Contributions",
            title_x=0.5
        )
        
        return fig, impact_df
        
    except Exception as e:
        st.error(f"Error creating SHAP plots: {str(e)}")
        return None, None

def validate_inputs(inputs):
    """Input validation"""
    errors = []
    warnings = []
    
    # Basic validation
    if inputs['Age'] < 18 or inputs['Age'] > 120:
        errors.append("Age must be between 18 and 120 years")
    
    if inputs['Height'] < 100 or inputs['Height'] > 250:
        errors.append("Height must be between 100 and 250 cm")
        
    if inputs['Weight'] < 30 or inputs['Weight'] > 300:
        errors.append("Weight must be between 30 and 300 kg")
    
    # BMI validation
    bmi = calculate_bmi(inputs['Height'], inputs['Weight'])
    if bmi < 10 or bmi > 80:
        errors.append("Calculated BMI seems unrealistic, please check height and weight")
    
    return errors, warnings

def collect_patient_inputs():
    """Collect all patient input data"""
    with st.sidebar:
        st.markdown('<div class="sidebar-header"><h3>Patient Information</h3></div>', 
                   unsafe_allow_html=True)
        
        # Demographics
        st.markdown('<div class="feature-group"><h4>Demographics</h4>', unsafe_allow_html=True)
        age = st.number_input("Age (years)", min_value=18, max_value=120, value=45, step=1)
        gender = st.selectbox("Gender", options=["Female", "Male"])
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.5)
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=70.0, step=0.1)
        
        bmi = calculate_bmi(height, weight)
        bmi_category = get_bmi_category(bmi)
        st.info(f"Calculated BMI: {bmi} kg/m² ({bmi_category})")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Medical History
        st.markdown('<div class="feature-group"><h4>Medical History</h4>', unsafe_allow_html=True)
        comorbidity = st.selectbox("Any Comorbidities", ["No", "Yes"])
        cad = st.selectbox("Coronary Artery Disease", ["No", "Yes"])
        hypothyroidism = st.selectbox("Hypothyroidism", ["No", "Yes"])
        hyperlipidemia = st.selectbox("Hyperlipidemia", ["No", "Yes"])
        dm = st.selectbox("Diabetes Mellitus", ["No", "Yes"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Body Composition
        st.markdown('<div class="feature-group"><h4>Body Composition</h4>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            tbw = st.number_input("Total Body Water (L)", min_value=20.0, max_value=80.0, value=40.0, step=0.5)
            ecw = st.number_input("Extracellular Water (L)", min_value=5.0, max_value=30.0, value=15.0, step=0.5)
            icw = st.number_input("Intracellular Water (L)", min_value=15.0, max_value=50.0, value=25.0, step=0.5)
        with col2:
            ecf_tbw = st.number_input("ECF/TBW Ratio", min_value=0.3, max_value=0.7, value=0.4, step=0.01)
            tbfr = st.number_input("Total Body Fat Ratio (%)", min_value=5.0, max_value=60.0, value=20.0, step=0.5)
            lean_mass = st.number_input("Lean Mass (%)", min_value=30.0, max_value=95.0, value=70.0, step=0.5)
        
        protein_content = st.number_input("Body Protein Content (%)", min_value=8.0, max_value=25.0, value=15.0, step=0.5)
        vfr = st.number_input("Visceral Fat Rating", min_value=1, max_value=30, value=10, step=1)
        bone_mass = st.number_input("Bone Mass (kg)", min_value=1.0, max_value=5.0, value=2.5, step=0.1)
        muscle_mass = st.number_input("Muscle Mass (kg)", min_value=15.0, max_value=80.0, value=35.0, step=0.5)
        obesity = st.number_input("Obesity (%)", min_value=0.0, max_value=100.0, value=25.0, step=0.5)
        tfc = st.number_input("Total Fat Content (%)", min_value=5.0, max_value=60.0, value=20.0, step=0.5)
        vfa = st.number_input("Visceral Fat Area (cm²)", min_value=30.0, max_value=300.0, value=100.0, step=1.0)
        vma = st.number_input("Visceral Muscle Area (kg)", min_value=15.0, max_value=60.0, value=30.0, step=0.5)
        hfa = st.number_input("Hepatic Fat Accumulation", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Laboratory Values
        st.markdown('<div class="feature-group"><h4>Laboratory Results</h4>', unsafe_allow_html=True)
        
        # Glucose & Lipids
        st.write("**Glucose & Lipid Profile**")
        col1, col2 = st.columns(2)
        with col1:
            glucose = st.number_input("Glucose (mg/dL)", min_value=50.0, max_value=600.0, value=100.0, step=1.0)
            total_cholesterol = st.number_input("Total Cholesterol (mg/dL)", min_value=100.0, max_value=500.0, value=200.0, step=1.0)
            ldl = st.number_input("LDL Cholesterol (mg/dL)", min_value=20.0, max_value=400.0, value=100.0, step=1.0)
        with col2:
            hdl = st.number_input("HDL Cholesterol (mg/dL)", min_value=20.0, max_value=150.0, value=50.0, step=1.0)
            triglyceride = st.number_input("Triglycerides (mg/dL)", min_value=30.0, max_value=1000.0, value=150.0, step=1.0)
        
        # Liver Function
        st.write("**Liver Function Tests**")
        col1, col2 = st.columns(2)
        with col1:
            ast = st.number_input("AST (U/L)", min_value=10.0, max_value=300.0, value=25.0, step=1.0)
            alt = st.number_input("ALT (U/L)", min_value=10.0, max_value=300.0, value=25.0, step=1.0)
        with col2:
            alp = st.number_input("Alkaline Phosphatase (U/L)", min_value=30.0, max_value=500.0, value=100.0, step=1.0)
        
        # Other Tests
        st.write("**Additional Tests**")
        col1, col2 = st.columns(2)
        with col1:
            creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.3, max_value=10.0, value=1.0, step=0.1)
            gfr = st.number_input("GFR (mL/min/1.73m²)", min_value=10.0, max_value=150.0, value=90.0, step=1.0)
            crp = st.number_input("C-Reactive Protein (mg/dL)", min_value=0.0, max_value=20.0, value=0.3, step=0.1)
        with col2:
            hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=25.0, value=14.0, step=0.1)
            vitamin_d = st.number_input("Vitamin D (ng/mL)", min_value=5.0, max_value=100.0, value=30.0, step=1.0)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Return all collected data
    return {
        'Age': age, 'Gender': gender, 'Comorbidity': comorbidity,
        'Coronary Artery Disease (CAD)': cad, 'Hypothyroidism': hypothyroidism,
        'Hyperlipidemia': hyperlipidemia, 'Diabetes Mellitus (DM)': dm,
        'Height': height, 'Weight': weight, 'Body Mass Index (BMI)': bmi,
        'Total Body Water (TBW)': tbw, 'Extracellular Water (ECW)': ecw,
        'Intracellular Water (ICW)': icw,
        'Extracellular Fluid/Total Body Water (ECF/TBW)': ecf_tbw,
        'Total Body Fat Ratio (TBFR) (%)': tbfr, 'Lean Mass (LM) (%)': lean_mass,
        'Body Protein Content (Protein) (%)': protein_content,
        'Visceral Fat Rating (VFR)': vfr, 'Bone Mass (BM)': bone_mass,
        'Muscle Mass (MM)': muscle_mass, 'Obesity (%)': obesity,
        'Total Fat Content (TFC)': tfc, 'Visceral Fat Area (VFA)': vfa,
        'Visceral Muscle Area (VMA) (Kg)': vma,
        'Hepatic Fat Accumulation (HFA)': hfa, 'Glucose': glucose,
        'Total Cholesterol (TC)': total_cholesterol,
        'Low Density Lipoprotein (LDL)': ldl,
        'High Density Lipoprotein (HDL)': hdl, 'Triglyceride': triglyceride,
        'Aspartat Aminotransferaz (AST)': ast,
        'Alanin Aminotransferaz (ALT)': alt,
        'Alkaline Phosphatase (ALP)': alp, 'Creatinine': creatinine,
        'Glomerular Filtration Rate (GFR)': gfr,
        'C-Reactive Protein (CRP)': crp, 'Hemoglobin (HGB)': hemoglobin,
        'Vitamin D': vitamin_d
    }

def main():
    # Initialize predictor
    predictor = GallstonePredictor()
    
    # Header
    st.markdown('<h1 class="main-header">Gallstone Risk Assessment System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="alert-box alert-info">
        <strong>Clinical Decision Support Tool</strong><br>
        This application uses machine learning to assess gallstone risk based on patient data.
        Results should be used in conjunction with clinical judgment.
    </div>
    """, unsafe_allow_html=True)
    
    # Model loading
    with st.spinner('Loading model...'):
        model_loaded = predictor.load_models()
    
    if not model_loaded:
        st.markdown(f"""
        <div class="alert-box alert-error">
            <strong>Model Loading Error</strong><br>
            {predictor.error_message}
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_model = st.file_uploader("Upload Model File", type=['pkl'])
        if uploaded_model:
            try:
                predictor.model = pickle.load(uploaded_model)
                if hasattr(predictor.model, 'predict') and hasattr(predictor.model, 'predict_proba'):
                    predictor.model_loaded = True
                    predictor.initialize_shap()
                    st.success("Model uploaded successfully!")
                    st.experimental_rerun()
                else:
                    st.error("Uploaded file is not a valid machine learning model")
            except Exception as e:
                st.error(f"Error loading uploaded model: {str(e)}")
        st.stop()
    
    # Collect patient data
    patient_data = collect_patient_inputs()
    
    # Input validation
    errors, warnings = validate_inputs(patient_data)
    
    if errors:
        st.markdown('<div class="alert-box alert-error">', unsafe_allow_html=True)
        st.error("Please correct the following errors:")
        for error in errors:
            st.write(f"• {error}")
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()
    
    if warnings:
        st.markdown('<div class="alert-box alert-warning">', unsafe_allow_html=True)
        st.warning("Please review:")
        for warning in warnings:
            st.write(f"• {warning}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="section-header">Patient Summary</h2>', unsafe_allow_html=True)
        
        # Demographics card
        st.markdown(f"""
        <div class="metric-card">
            <h4>Demographics</h4>
            <strong>Age:</strong> {patient_data['Age']} years<br>
            <strong>Gender:</strong> {patient_data['Gender']}<br>
            <strong>BMI:</strong> {patient_data['Body Mass Index (BMI)']:.1f} kg/m² ({get_bmi_category(patient_data['Body Mass Index (BMI)'])})<br>
            <strong>Height:</strong> {patient_data['Height']} cm<br>
            <strong>Weight:</strong> {patient_data['Weight']} kg
        </div>
        """, unsafe_allow_html=True)
        
        # Medical history
        conditions = []
        medical_fields = ['Comorbidity', 'Coronary Artery Disease (CAD)', 'Hypothyroidism', 
                         'Hyperlipidemia', 'Diabetes Mellitus (DM)']
        for field in medical_fields:
            if patient_data[field] == "Yes":
                conditions.append(field.replace(' (DM)', '').replace(' (CAD)', ''))
        
        medical_history = ", ".join(conditions) if conditions else "No significant medical history"
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Medical History</h4>
            <strong>Conditions:</strong> {medical_history}
        </div>
        """, unsafe_allow_html=True)
        
        # Key lab values
        st.markdown(f"""
        <div class="metric-card">
            <h4>Key Laboratory Values</h4>
            <strong>Glucose:</strong> {patient_data['Glucose']} mg/dL<br>
            <strong>Total Cholesterol:</strong> {patient_data['Total Cholesterol (TC)']} mg/dL<br>
            <strong>HDL:</strong> {patient_data['High Density Lipoprotein (HDL)']} mg/dL<br>
            <strong>LDL:</strong> {patient_data['Low Density Lipoprotein (LDL)']} mg/dL<br>
            <strong>Triglycerides:</strong> {patient_data['Triglyceride']} mg/dL
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h2 class="section-header">Risk Assessment</h2>', unsafe_allow_html=True)
        
        try:
            with st.spinner('Analyzing patient data...'):
                # Convert categorical variables
                processed_data = patient_data.copy()
                processed_data['Gender'] = 1 if patient_data['Gender'] == "Male" else 0
                for field in ['Comorbidity', 'Coronary Artery Disease (CAD)', 
                             'Hypothyroidism', 'Hyperlipidemia', 'Diabetes Mellitus (DM)']:
                    processed_data[field] = 1 if patient_data[field] == "Yes" else 0
                
                # Create DataFrame
                df = pd.DataFrame([processed_data])
                
                # Make prediction
                prediction_proba = predictor.model.predict_proba(df)[0]
                prediction = predictor.model.predict(df)[0]
                
                risk_probability = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
                risk_info = get_risk_level_info(risk_probability)
                
                # Display prediction result
                st.markdown(f"""
                <div class="prediction-result {risk_info['class']}">
                    {risk_info['level']} RISK<br>
                    Probability: {risk_probability*100:.1f}%<br>
                    <small>{risk_info['description']}</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk gauge
                fig_gauge = create_risk_gauge(risk_probability)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Recommendation
                st.markdown(f"""
                <div class="recommendation-box">
                    <h5>Clinical Recommendation</h5>
                    <p><strong>{risk_info['recommendation']}</strong></p>
                    <p>{risk_info['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.markdown(f"""
            <div class="alert-box alert-error">
                <strong>Prediction Error</strong><br>
                Error: {str(e)}<br>
                Please verify your model file and input data.
            </div>
            """, unsafe_allow_html=True)
            st.stop()
    
    # SHAP Analysis Section
    st.markdown('<h2 class="section-header">AI Explanation Analysis</h2>', unsafe_allow_html=True)
    
    if predictor.explainer is not None:
        try:
            with st.spinner('Computing SHAP explanations...'):
                # Compute SHAP values
                shap_values = predictor.explainer(df)
                
                # Handle SHAP output format
                if hasattr(shap_values, 'values'):
                    if len(shap_values.values.shape) > 2:
                        shap_vals = shap_values.values[0, :, 1]  # Binary classification positive class
                    else:
                        shap_vals = shap_values.values[0]
                else:
                    shap_vals = shap_values[0]
                    if isinstance(shap_vals, list):
                        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
                
                feature_names = df.columns.tolist()
                feature_values = df.iloc[0].values
                
                # Create SHAP plots
                fig_shap, impact_df = create_shap_plots(shap_vals, feature_values, feature_names)
                
                if fig_shap is not None:
                    st.plotly_chart(fig_shap, use_container_width=True)
                    
                    # Feature impact summary
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("#### Risk-Increasing Factors")
                        risk_increasing = impact_df[impact_df['shap_value'] > 0].head(5)
                        if not risk_increasing.empty:
                            for _, row in risk_increasing.iterrows():
                                st.markdown(f"**{row['feature']}:** {row['feature_value']:.2f} (Impact: +{row['shap_value']:.3f})")
                        else:
                            st.markdown("*No significant risk-increasing factors*")
                    
                    with col2:
                        st.markdown("#### Risk-Decreasing Factors")
                        risk_decreasing = impact_df[impact_df['shap_value'] < 0].head(5)
                        if not risk_decreasing.empty:
                            for _, row in risk_decreasing.iterrows():
                                st.markdown(f"**{row['feature']}:** {row['feature_value']:.2f} (Impact: {row['shap_value']:.3f})")
                        else:
                            st.markdown("*No significant risk-decreasing factors*")
                    
                    # Complete analysis table
                    st.markdown("#### Complete Feature Analysis")
                    display_df = impact_df[['feature', 'feature_value', 'shap_value']].copy()
                    display_df.columns = ['Feature', 'Patient Value', 'SHAP Impact']
                    display_df['Patient Value'] = display_df['Patient Value'].round(2)
                    display_df['SHAP Impact'] = display_df['SHAP Impact'].round(4)
                    display_df['Effect'] = display_df['SHAP Impact'].apply(lambda x: 'Increases Risk' if x > 0 else 'Decreases Risk')
                    
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                
        except Exception as e:
            st.error(f"Error generating SHAP explanations: {str(e)}")
            st.info("SHAP explanations require compatible model and data formats")
    else:
        st.warning("SHAP explainer not available - showing basic model information")
        
        # Fallback: show feature importance if available
        if hasattr(predictor.model, 'feature_importances_'):
            st.markdown("#### Model Feature Importance")
            feature_names = df.columns.tolist()
            importances = predictor.model.feature_importances_
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances,
                'Patient Value': df.iloc[0].values
            }).sort_values('Importance', ascending=False).head(15)
            
            fig_importance = px.bar(
                importance_df, 
                x='Importance', 
                y='Feature',
                orientation='h',
                title='Top 15 Most Important Features',
                color_discrete_sequence=['#3498db']
            )
            fig_importance.update_layout(height=500)
            st.plotly_chart(fig_importance, use_container_width=True)
            
            st.dataframe(importance_df, use_container_width=True, hide_index=True)
    
    # Clinical Guidelines
    st.markdown('<h2 class="section-header">Clinical Guidelines</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("""
        #### Risk Factors
        - **Age**: Risk increases after 40
        - **Gender**: Women at higher risk
        - **Obesity**: BMI >30 increases risk
        - **Diabetes**: Impairs gallbladder function
        - **High Cholesterol**: Promotes stone formation
        - **Family History**: Genetic predisposition
        """)
    
    with col2:
        st.markdown("""
        #### Prevention
        - **Healthy Weight**: Maintain normal BMI
        - **Regular Exercise**: 30+ minutes daily
        - **Balanced Diet**: High fiber, low fat
        - **Avoid Rapid Weight Loss**: Gradual changes
        - **Stay Hydrated**: Adequate water intake
        - **Regular Meals**: Don't skip meals
        """)
    
    with col3:
        st.markdown("""
        #### Warning Signs
        - **Severe abdominal pain** (RUQ)
        - **Nausea and vomiting**
        - **Fever with abdominal pain**
        - **Jaundice** (yellowing of skin/eyes)
        - **Clay-colored stools**
        - **Dark urine**
        """)
    
    # Risk-specific recommendations
    if 'risk_probability' in locals():
        st.markdown("#### Personalized Recommendations")
        
        if risk_probability >= 0.8:
            st.markdown("""
            <div class="alert-box alert-error">
                <strong>CRITICAL RISK - Immediate Action Required</strong>
                <ul>
                    <li>Schedule <strong>immediate consultation</strong> with gastroenterologist</li>
                    <li>Consider <strong>abdominal ultrasound</strong> for screening</li>
                    <li>Monitor symptoms closely - seek emergency care for severe pain</li>
                    <li>Implement immediate lifestyle modifications</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        elif risk_probability >= 0.6:
            st.markdown("""
            <div class="alert-box alert-warning">
                <strong>HIGH RISK - Proactive Management</strong>
                <ul>
                    <li>Schedule consultation with physician within 2-4 weeks</li>
                    <li>Consider screening ultrasound based on symptoms</li>
                    <li>Implement lifestyle changes - diet and exercise</li>
                    <li>Monitor and manage risk factors</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        elif risk_probability >= 0.4:
            st.markdown("""
            <div class="alert-box alert-warning">
                <strong>MODERATE RISK - Preventive Measures</strong>
                <ul>
                    <li>Discuss with primary care physician at next visit</li>
                    <li>Focus on prevention through healthy lifestyle</li>
                    <li>Monitor for symptoms</li>
                    <li>Address modifiable risk factors</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.markdown("""
            <div class="alert-box alert-success">
                <strong>LOW RISK - Continue Healthy Habits</strong>
                <ul>
                    <li>Maintain current healthy lifestyle</li>
                    <li>Continue regular exercise and balanced diet</li>
                    <li>Monitor for any new symptoms</li>
                    <li>Regular health screenings as appropriate</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Export functionality
    st.markdown('<h2 class="section-header">Export Results</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("Download Report", type="primary"):
            # Create report data
            if 'risk_probability' in locals():
                report_data = {
                    'timestamp': datetime.now().isoformat(),
                    'patient_data': patient_data,
                    'risk_assessment': {
                        'probability': float(risk_probability),
                        'prediction': int(prediction),
                        'risk_level': risk_info['level']
                    }
                }
                
                if 'impact_df' in locals():
                    report_data['feature_analysis'] = impact_df.to_dict('records')
                
                json_str = json.dumps(report_data, indent=2)
                
                st.download_button(
                    label="Download JSON Report",
                    data=json_str,
                    file_name=f"gallstone_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    with col2:
        if 'impact_df' in locals() and st.button("Export Analysis", type="secondary"):
            csv_data = impact_df.to_csv(index=False)
            st.download_button(
                label="Download CSV Analysis",
                data=csv_data,
                file_name=f"feature_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("Reset Form", type="secondary"):
            st.experimental_rerun()
    
    # Medical Disclaimer
    st.markdown("---")
    st.markdown("""
    <div class="alert-box alert-warning">
        <strong>Medical Disclaimer</strong><br>
        This AI tool is for educational and research purposes only. Results should not replace 
        professional medical advice, diagnosis, or treatment. Always consult qualified healthcare 
        providers for medical decisions. For medical emergencies, contact emergency services immediately.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()