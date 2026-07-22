import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support

from dataset_generator import generate_credit_data

def train_and_evaluate():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'credit_data.csv')
    
    if not os.path.exists(csv_path):
        print("Dataset not found. Generating new dataset...")
        df = generate_credit_data()
    else:
        df = pd.read_csv(csv_path)
        
    print(f"Dataset shape: {df.shape}")
    
    X = df.drop(columns=['credit_score_class'])
    y = df['credit_score_class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    }
    
    results = {}
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
            
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        
        results[name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'model': model
        }
        
        print(f"\n--- {name} ---")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=['Poor', 'Standard', 'Good']))
        
        if f1 > best_score:
            best_score = f1
            best_model = model
            best_name = name
            
    print(f"\nBest Model selected: {best_name} with F1-Score: {best_score:.4f}")
    
    # Save artifacts
    joblib.dump(best_model, os.path.join(base_dir, 'credit_scoring_model.pkl'))
    joblib.dump(scaler, os.path.join(base_dir, 'scaler.pkl'))
    joblib.dump(list(X.columns), os.path.join(base_dir, 'feature_names.pkl'))
    
    # Visualizations
    plt.figure(figsize=(8, 6))
    if best_name == 'Logistic Regression':
        cm = confusion_matrix(y_test, best_model.predict(X_test_scaled))
    else:
        cm = confusion_matrix(y_test, best_model.predict(X_test))
        
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Poor', 'Standard', 'Good'], yticklabels=['Poor', 'Standard', 'Good'])
    plt.title(f'Confusion Matrix - {best_name}')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Feature Importance Plot
    if hasattr(best_model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.bar(range(X.shape[1]), importances[indices], align='center', color='#4f46e5')
        plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=45, ha='right')
        plt.title('Feature Importances for Credit Scoring')
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, 'feature_importance.png'), dpi=300)
        plt.close()
        
    print(f"Artifacts successfully saved to {base_dir}")

if __name__ == '__main__':
    train_and_evaluate()
