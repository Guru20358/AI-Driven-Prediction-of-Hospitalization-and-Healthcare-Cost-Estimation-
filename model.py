from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

# Load synthetic dataset
data = pd.read_csv('synthetic_healthcare_data.csv')

# Prepare features and target for disease prediction
X_disease = data.drop(['Disease_Flu', 'Disease_COVID-19', 'Disease_Asthma', 'Disease_Hypertension', 
                       'Severity_Mild', 'Severity_Moderate', 'Severity_Severe', 'Hospital_Stay', 'Medical_Cost'], axis=1)
y_disease = data[['Disease_Flu', 'Disease_COVID-19', 'Disease_Asthma', 'Disease_Hypertension']].idxmax(axis=1)

# Train-test split for disease prediction
X_train_disease, X_test_disease, y_train_disease, y_test_disease = train_test_split(X_disease, y_disease, test_size=0.2, random_state=42)

# Train Random Forest Classifier for disease prediction
disease_model = RandomForestClassifier()
disease_model.fit(X_train_disease, y_train_disease)

# Save the disease model
joblib.dump(disease_model, 'disease_model.pkl')
print("Disease model saved as 'disease_model.pkl'")

# Prepare features and target for severity prediction
X_severity = data.drop(['Severity_Mild', 'Severity_Moderate', 'Severity_Severe', 'Hospital_Stay', 'Medical_Cost'], axis=1)
y_severity = data[['Severity_Mild', 'Severity_Moderate', 'Severity_Severe']].idxmax(axis=1)

# Train-test split for severity prediction
X_train_severity, X_test_severity, y_train_severity, y_test_severity = train_test_split(X_severity, y_severity, test_size=0.2, random_state=42)

# Train Random Forest Classifier for severity prediction
severity_model = RandomForestClassifier()
severity_model.fit(X_train_severity, y_train_severity)

# Save the severity model
joblib.dump(severity_model, 'severity_model.pkl')
print("Severity model saved as 'severity_model.pkl'")

# Prepare features and target for hospital stay prediction
X_stay = data.drop(['Hospital_Stay', 'Medical_Cost'], axis=1)
y_stay = data['Hospital_Stay']

# Train-test split for hospital stay prediction
X_train_stay, X_test_stay, y_train_stay, y_test_stay = train_test_split(X_stay, y_stay, test_size=0.2, random_state=42)

# Train Random Forest Regressor for hospital stay prediction
stay_model = RandomForestRegressor()
stay_model.fit(X_train_stay, y_train_stay)

# Save the stay model
joblib.dump(stay_model, 'stay_model.pkl')
print("Stay model saved as 'stay_model.pkl'")

# Prepare features and target for medical cost prediction
X_cost = data.drop(['Medical_Cost'], axis=1)
y_cost = data['Medical_Cost']

# Train-test split for medical cost prediction
X_train_cost, X_test_cost, y_train_cost, y_test_cost = train_test_split(X_cost, y_cost, test_size=0.2, random_state=42)

# Train Random Forest Regressor for medical cost prediction
cost_model = RandomForestRegressor()
cost_model.fit(X_train_cost, y_train_cost)

# Save the cost model
joblib.dump(cost_model, 'cost_model.pkl')
print("Cost model saved as 'cost_model.pkl'")
