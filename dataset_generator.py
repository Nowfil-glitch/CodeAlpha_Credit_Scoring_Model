import numpy as np
import pandas as pd
import os

def generate_credit_data(n_samples=2500, seed=42):
    np.random.seed(seed)
    
    age = np.random.randint(18, 70, size=n_samples)
    annual_income = np.random.uniform(15000, 150000, size=n_samples)
    num_bank_accounts = np.random.randint(1, 10, size=n_samples)
    num_credit_cards = np.random.randint(1, 12, size=n_samples)
    interest_rate = np.random.uniform(5, 35, size=n_samples)
    num_loans = np.random.randint(0, 8, size=n_samples)
    delay_from_due_date_days = np.random.exponential(scale=5, size=n_samples).astype(int)
    num_delayed_payments = np.random.poisson(lam=3, size=n_samples)
    credit_card_utilization = np.random.uniform(0.05, 0.95, size=n_samples)
    credit_history_age_months = np.random.randint(6, 360, size=n_samples)
    outstanding_debt = np.random.uniform(200, 35000, size=n_samples)
    
    # Feature engineering: Debt-to-Income Ratio
    debt_to_income_ratio = (outstanding_debt / np.maximum(annual_income, 1)) * 100
    
    # Calculate a credit score index
    raw_score = (
        (annual_income / 1000) * 0.3
        + (credit_history_age_months / 12) * 2.0
        - (delay_from_due_date_days * 1.5)
        - (num_delayed_payments * 2.5)
        - (credit_card_utilization * 40)
        - (debt_to_income_ratio * 0.5)
        - (interest_rate * 0.8)
    )
    
    # Add noise
    raw_score += np.random.normal(0, 10, size=n_samples)
    
    # Categorize credit score class: 0 -> Poor, 1 -> Standard, 2 -> Good
    p33, p66 = np.percentile(raw_score, [35, 70])
    
    credit_score_class = []
    for s in raw_score:
        if s < p33:
            credit_score_class.append(0) # Poor
        elif s < p66:
            credit_score_class.append(1) # Standard
        else:
            credit_score_class.append(2) # Good
            
    df = pd.DataFrame({
        'age': age,
        'annual_income': np.round(annual_income, 2),
        'num_bank_accounts': num_bank_accounts,
        'num_credit_cards': num_credit_cards,
        'interest_rate': np.round(interest_rate, 2),
        'num_loans': num_loans,
        'delay_from_due_date_days': delay_from_due_date_days,
        'num_delayed_payments': num_delayed_payments,
        'credit_card_utilization': np.round(credit_card_utilization, 4),
        'credit_history_age_months': credit_history_age_months,
        'outstanding_debt': np.round(outstanding_debt, 2),
        'debt_to_income_ratio': np.round(debt_to_income_ratio, 2),
        'credit_score_class': credit_score_class
    })
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(out_dir, 'credit_data.csv')
    df.to_csv(file_path, index=False)
    print(f"Generated synthetic dataset with {n_samples} records saved to {file_path}")
    return df

if __name__ == '__main__':
    generate_credit_data()
