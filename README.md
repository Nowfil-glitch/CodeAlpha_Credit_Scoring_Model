# Task 1: Credit Scoring Model — CodeAlpha Machine Learning Internship

## 📌 Project Overview
This repository contains a complete machine learning solution for **Credit Scoring & Creditworthiness Prediction**, developed as part of the **CodeAlpha Machine Learning Internship**.

The model classifies individuals into creditworthiness categories (`Poor`, `Standard`, `Good`) based on financial history parameters, helping financial institutions assess credit default risk automatically.

---

## 🚀 Key Features
- **Feature Engineering**: Calculates critical financial metrics like *Debt-to-Income Ratio (DTI)*, credit age index, payment delay penalties, and utilization rates.
- **Multi-Model Benchmark**: Evaluates **Logistic Regression**, **Decision Trees**, and **Random Forest Classifiers**.
- **Model Evaluation**: Assesses Performance using **Precision**, **Recall**, **F1-Score**, and **ROC-AUC** metrics.
- **Interactive Web Interface**: Complete Flask visual app with real-time score index calculator, glassmorphic UI, and probability breakdowns.

---

## 📊 Dataset Parameters
- `age`: Borrower age
- `annual_income`: Total yearly income
- `num_bank_accounts`: Active bank accounts count
- `num_credit_cards`: Total active credit cards
- `interest_rate`: Average interest rate across loans
- `delay_from_due_date_days`: Days overdue past payment deadline
- `num_delayed_payments`: Total historical late payments
- `credit_card_utilization`: Ratio of credit used vs total limit
- `credit_history_age_months`: Age of credit account history in months
- `outstanding_debt`: Total current debt
- `debt_to_income_ratio`: Engineered feature `(outstanding_debt / annual_income) * 100`

---

## 🛠️ Installation & Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Dataset & Train Model**:
   ```bash
   python train_model.py
   ```

3. **Run Interactive Web Application**:
   ```bash
   python app.py
   ```
   Open your browser at `http://localhost:5001`.

---

## 📈 Model Performance & Evaluation Metrics
- **Best Model**: Random Forest Classifier
- **F1-Score**: ~94%+
- **ROC-AUC Score**: ~0.98+

---

Developed for **CodeAlpha Machine Learning Internship**.
