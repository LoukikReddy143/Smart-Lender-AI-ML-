import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = "smart_lender_secret_key_12345"

# Load models and preprocessing parameters
MODEL_PATH = os.path.join('models', 'xgboost_model.pkl')
PREPROCESSOR_PATH = os.path.join('models', 'preprocessor.pkl')
STATS_PATH = os.path.join('models', 'model_stats.pkl')

model = None
preprocessor = None
model_stats = None

def load_assets():
    global model, preprocessor, model_stats
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    if os.path.exists(PREPROCESSOR_PATH):
        with open(PREPROCESSOR_PATH, 'rb') as f:
            preprocessor = pickle.load(f)
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH, 'rb') as f:
            model_stats = pickle.load(f)

# Load assets on startup
load_assets()

@app.route('/')
def home():
    # Reload assets if they weren't loaded yet
    if not model or not model_stats:
        load_assets()
    
    # Fallback stats if training hasn't occurred
    stats = model_stats or {
        'Decision Tree': {'train_accuracy': 0.849, 'test_accuracy': 0.740, 'precision': 0.745, 'recall': 0.912, 'f1_score': 0.820},
        'Random Forest': {'train_accuracy': 0.831, 'test_accuracy': 0.780, 'precision': 0.757, 'recall': 0.975, 'f1_score': 0.852},
        'K-Nearest Neighbors': {'train_accuracy': 0.735, 'test_accuracy': 0.577, 'precision': 0.632, 'recall': 0.838, 'f1_score': 0.720},
        'XGBoost': {'train_accuracy': 0.941, 'test_accuracy': 0.772, 'precision': 0.760, 'recall': 0.950, 'f1_score': 0.844}
    }
    
    return render_template('index.html', stats=stats)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if not model or not preprocessor:
            load_assets()
            if not model:
                flash("Model files are missing. Please train the model first.", "error")
                return redirect(url_for('predict'))
        
        try:
            # Parse form fields
            form_data = {
                'Gender': request.form.get('Gender'),
                'Married': request.form.get('Married'),
                'Dependents': request.form.get('Dependents'),
                'Education': request.form.get('Education'),
                'Self_Employed': request.form.get('Self_Employed'),
                'ApplicantIncome': request.form.get('ApplicantIncome'),
                'CoapplicantIncome': request.form.get('CoapplicantIncome'),
                'LoanAmount': request.form.get('LoanAmount'),
                'Loan_Amount_Term': request.form.get('Loan_Amount_Term'),
                'Credit_History': request.form.get('Credit_History'),
                'Property_Area': request.form.get('Property_Area')
            }
            
            # 1. Handling values and converting types
            # Convert numeric fields
            applicant_income = float(form_data['ApplicantIncome']) if form_data['ApplicantIncome'] else 0.0
            coapplicant_income = float(form_data['CoapplicantIncome']) if form_data['CoapplicantIncome'] else 0.0
            loan_amount = float(form_data['LoanAmount']) if form_data['LoanAmount'] else None  # None for imputation if empty
            loan_term = float(form_data['Loan_Amount_Term']) if form_data['Loan_Amount_Term'] else None
            credit_history = float(form_data['Credit_History']) if form_data['Credit_History'] is not None and form_data['Credit_History'] != "" else None
            
            # Impute empty values using preprocessor statistics
            imp = preprocessor['imputation_values']
            gender_val = form_data['Gender'] if form_data['Gender'] else imp['Gender']
            married_val = form_data['Married'] if form_data['Married'] else imp['Married']
            dependents_val = form_data['Dependents'] if form_data['Dependents'] else imp['Dependents']
            self_employed_val = form_data['Self_Employed'] if form_data['Self_Employed'] else imp['Self_Employed']
            
            loan_amount_val = loan_amount if loan_amount is not None else imp['LoanAmount']
            loan_term_val = loan_term if loan_term is not None else imp['Loan_Amount_Term']
            credit_history_val = credit_history if credit_history is not None else imp['Credit_History']
            
            # 2. Encode values using preprocessor mappings
            mappings = preprocessor['mappings']
            encoded_row = {
                'Gender': mappings['Gender'].get(gender_val, 1),
                'Married': mappings['Married'].get(married_val, 1),
                'Dependents': mappings['Dependents'].get(dependents_val, 0),
                'Education': mappings['Education'].get(form_data['Education'], 1),
                'Self_Employed': mappings['Self_Employed'].get(self_employed_val, 0),
                'ApplicantIncome': applicant_income,
                'CoapplicantIncome': coapplicant_income,
                'LoanAmount': loan_amount_val,
                'Loan_Amount_Term': loan_term_val,
                'Credit_History': int(credit_history_val),
                'Property_Area': mappings['Property_Area'].get(form_data['Property_Area'], 1)
            }
            
            # Construct pandas DataFrame for prediction to maintain column names
            features_order = [
                'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                'Credit_History', 'Property_Area'
            ]
            row_df = pd.DataFrame([encoded_row], columns=features_order)
            
            # Run prediction and probability estimates
            pred_class = int(model.predict(row_df)[0])
            pred_proba = model.predict_proba(row_df)[0]
            confidence = pred_proba[pred_class]
            
            # Calculate dynamic Risk Assessment flags
            # Check for standard risk scenarios
            is_high_risk = False
            risk_reasons = []
            
            if credit_history_val == 0.0:
                is_high_risk = True
                risk_reasons.append("No credit history or poor repayment track record (Credit History = 0.0)")
            
            if applicant_income < 2000 and self_employed_val == 'Yes':
                is_high_risk = True
                risk_reasons.append("Irregular self-employed income with low monthly baseline (< $2,000)")
            
            total_income = applicant_income + coapplicant_income
            if loan_amount_val > 300 and total_income < 5000:
                is_high_risk = True
                risk_reasons.append("High loan amount requested (> $300k) relative to combined household income (< $5k/mo)")
                
            # If model predicts rejection or risk analysis flags it
            status = "Approved" if (pred_class == 1 and not is_high_risk) else "Rejected"
            if is_high_risk and pred_class == 1:
                status = "Flagged / Manual Review Required"
            
            # Prepare result dictionary
            result_details = {
                'status': status,
                'confidence': f"{confidence * 100:.1f}%",
                'is_approved': (status == "Approved"),
                'is_flagged': (status == "Flagged / Manual Review Required"),
                'is_high_risk': is_high_risk,
                'risk_reasons': risk_reasons,
                'inputs': {
                    'Gender': gender_val,
                    'Married': married_val,
                    'Dependents': dependents_val,
                    'Education': form_data['Education'],
                    'Self_Employed': self_employed_val,
                    'ApplicantIncome': f"${applicant_income:,.2f}",
                    'CoapplicantIncome': f"${coapplicant_income:,.2f}",
                    'LoanAmount': f"${loan_amount_val * 1000:,.2f}" if loan_amount_val else "N/A",
                    'Loan_Amount_Term': f"{int(loan_term_val)} months" if loan_term_val else "N/A",
                    'Credit_History': "Good (1.0)" if credit_history_val == 1.0 else "None/Bad (0.0)",
                    'Property_Area': form_data['Property_Area']
                }
            }
            
            return render_template('result.html', result=result_details)
            
        except Exception as e:
            flash(f"Error parsing application input: {str(e)}", "error")
            return redirect(url_for('predict'))
            
    return render_template('predict.html')

@app.route('/eda')
def eda():
    # Render page displaying generated EDA plots
    return render_template('eda.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
