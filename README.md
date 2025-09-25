# Customer-Retention-Analytics
This project demonstrates end-to-end data science value: from raw data to business recommendations in a single, efficient pipeline.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score

# Generate sample data
np.random.seed(42)
data = {
    'tenure': np.random.randint(1, 72, 1000),
    'monthly_charges': np.random.normal(70, 30, 1000),
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], 1000, p=[0.5, 0.3, 0.2]),
    'churn': np.random.binomial(1, 0.2, 1000)
}
df = pd.DataFrame(data)

# Preprocess
le = LabelEncoder()
df['contract_encoded'] = le.fit_transform(df['contract_type'])
X = df[['tenure', 'monthly_charges', 'contract_encoded']]
y = df['churn']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Results
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"AUC: {roc_auc_score(y_test, y_proba):.3f}")

# Feature importance
importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
print("\nFeature Importance:\n", importance.sort_values('importance', ascending=False))

# Business impact
high_risk = y_proba > 0.7
potential_churn = y_test[high_risk].sum()
revenue_saved = potential_churn * 70 * 12 * 0.4  # 40% retention success
print(f"\nBusiness Impact: ${revenue_saved:,.0f} annual revenue saved")

# Visualization
plt.figure(figsize=(10, 3))
plt.subplot(1, 2, 1)
plt.bar(importance['feature'], importance['importance'])
plt.title('Feature Importance')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
risk_groups = ['Low', 'Medium', 'High']
risk_counts = [(y_proba <= 0.3).sum(), ((y_proba > 0.3) & (y_proba <= 0.7)).sum(), (y_proba > 0.7).sum()]
plt.bar(risk_groups, risk_counts, color=['green', 'orange', 'red'])
plt.title('Customers by Risk Level')
plt.tight_layout()
plt.show()
