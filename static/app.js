document.getElementById('credit-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const payload = {
        age: parseFloat(document.getElementById('age').value),
        annual_income: parseFloat(document.getElementById('annual_income').value),
        outstanding_debt: parseFloat(document.getElementById('outstanding_debt').value),
        credit_card_utilization: parseFloat(document.getElementById('credit_card_utilization').value),
        credit_history_age_months: parseFloat(document.getElementById('credit_history_age_months').value),
        delay_from_due_date_days: parseFloat(document.getElementById('delay_from_due_date_days').value),
        num_delayed_payments: parseFloat(document.getElementById('num_delayed_payments').value),
        interest_rate: parseFloat(document.getElementById('interest_rate').value),
        num_bank_accounts: parseFloat(document.getElementById('num_bank_accounts').value),
        num_credit_cards: parseFloat(document.getElementById('num_credit_cards').value),
        num_loans: 2
    };

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        if (data.status === 'success') {
            document.getElementById('credit-score').innerText = data.estimated_score;
            document.getElementById('dti-val').innerText = `${data.debt_to_income_ratio}%`;
            
            const badge = document.getElementById('rating-badge');
            badge.innerText = data.rating;
            badge.style.backgroundColor = `${data.badge_color}22`;
            badge.style.color = data.badge_color;
            badge.style.border = `1px solid ${data.badge_color}`;
            
            document.getElementById('prob-good').innerText = `${data.probabilities.Good}%`;
            document.getElementById('bar-good').style.width = `${data.probabilities.Good}%`;
            
            document.getElementById('prob-standard').innerText = `${data.probabilities.Standard}%`;
            document.getElementById('bar-standard').style.width = `${data.probabilities.Standard}%`;
            
            document.getElementById('prob-poor').innerText = `${data.probabilities.Poor}%`;
            document.getElementById('bar-poor').style.width = `${data.probabilities.Poor}%`;
        }
    } catch (err) {
        console.error("Error predicting credit score:", err);
    }
});
