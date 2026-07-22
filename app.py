import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='static')
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load model artifacts
model_path = os.path.join(base_dir, 'credit_scoring_model.pkl')
scaler_path = os.path.join(base_dir, 'scaler.pkl')

if not os.path.exists(model_path):
    from train_model import train_and_evaluate
    train_and_evaluate()

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

CLASS_MAP = {0: 'Poor Credit Risk', 1: 'Standard Credit', 2: 'Good Credit (Low Risk)'}
COLOR_MAP = {0: '#ef4444', 1: '#f59e0b', 2: '#10b981'}

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        age = float(data.get('age', 30))
        annual_income = float(data.get('annual_income', 55000))
        num_bank_accounts = float(data.get('num_bank_accounts', 3))
        num_credit_cards = float(data.get('num_credit_cards', 4))
        interest_rate = float(data.get('interest_rate', 14))
        num_loans = float(data.get('num_loans', 2))
        delay_from_due_date_days = float(data.get('delay_from_due_date_days', 2))
        num_delayed_payments = float(data.get('num_delayed_payments', 1))
        credit_card_utilization = float(data.get('credit_card_utilization', 0.35))
        credit_history_age_months = float(data.get('credit_history_age_months', 120))
        outstanding_debt = float(data.get('outstanding_debt', 5000))
        
        debt_to_income_ratio = (outstanding_debt / max(annual_income, 1.0)) * 100.0
        
        input_data = pd.DataFrame([{
            'age': age,
            'annual_income': annual_income,
            'num_bank_accounts': num_bank_accounts,
            'num_credit_cards': num_credit_cards,
            'interest_rate': interest_rate,
            'num_loans': num_loans,
            'delay_from_due_date_days': delay_from_due_date_days,
            'num_delayed_payments': num_delayed_payments,
            'credit_card_utilization': credit_card_utilization,
            'credit_history_age_months': credit_history_age_months,
            'outstanding_debt': outstanding_debt,
            'debt_to_income_ratio': debt_to_income_ratio
        }])
        
        if hasattr(model, 'feature_importances_'):
            pred_class = int(model.predict(input_data)[0])
            probs = model.predict_proba(input_data)[0].tolist()
        else:
            scaled_data = scaler.transform(input_data)
            pred_class = int(model.predict(scaled_data)[0])
            probs = model.predict_proba(scaled_data)[0].tolist()
            
        score_estimate = int(300 + (probs[2] * 400 + probs[1] * 200 + probs[0] * 50))
        
        return jsonify({
            'status': 'success',
            'prediction_class': pred_class,
            'rating': CLASS_MAP[pred_class],
            'badge_color': COLOR_MAP[pred_class],
            'estimated_score': score_estimate,
            'probabilities': {
                'Poor': round(probs[0] * 100, 2),
                'Standard': round(probs[1] * 100, 2),
                'Good': round(probs[2] * 100, 2)
            },
            'debt_to_income_ratio': round(debt_to_income_ratio, 2)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
