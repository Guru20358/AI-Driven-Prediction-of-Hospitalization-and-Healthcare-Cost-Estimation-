from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib
import os

app = Flask(__name__)
app.secret_key = 'secret123'

# Database setup
DB_NAME = 'users.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL,
                        email TEXT NOT NULL,
                        password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

init_db()

# Load models
disease_model = joblib.load('disease_model.pkl')
severity_model = joblib.load('severity_model.pkl')
stay_model = joblib.load('stay_model.pkl')
cost_model = joblib.load('cost_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)', (username, email, password))
        conn.commit()
        conn.close()

        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            session['username'] = username
            return redirect(url_for('about'))
        else:
            return "Invalid credentials. Try again."

    return render_template('login.html')

@app.route('/about')
def about():
    if 'username' in session:
        return render_template('about.html')
    return redirect(url_for('login'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        # Collect user inputs
        symptom1 = request.form['symptom1']
        symptom2 = request.form['symptom2']
        symptom3 = request.form['symptom3']
        symptom4 = request.form['symptom4']
        age = int(request.form['age'])
        gender = request.form['gender']

        # Create input data dictionary
        input_data = {
            'Age': [age],
            f'Gender_{gender}': [1],
            f'Symptom_1_{symptom1}': [1],
            f'Symptom_2_{symptom2}': [1],
            f'Symptom_3_{symptom3}': [1],
            f'Symptom_4_{symptom4}': [1]
        }

        # Ensure all features required by the models are present
        for feature in disease_model.feature_names_in_:
            if feature not in input_data:
                input_data[feature] = [0]

        for feature in severity_model.feature_names_in_:
            if feature not in input_data:
                input_data[feature] = [0]

        for feature in stay_model.feature_names_in_:
            if feature not in input_data:
                input_data[feature] = [0]

        for feature in cost_model.feature_names_in_:
            if feature not in input_data:
                input_data[feature] = [0]

        # Convert to DataFrame
        input_df = pd.DataFrame(input_data)

        # Predict disease
        predicted_disease = disease_model.predict(input_df[disease_model.feature_names_in_])[0]

        # Add disease prediction to input
        input_data[f'Disease_{predicted_disease}'] = [1]

        # Update DataFrame with new feature
        input_df = pd.DataFrame(input_data)

        # Predict severity
        predicted_severity = severity_model.predict(input_df[severity_model.feature_names_in_])[0]

        # Predict hospital stay
        predicted_stay = stay_model.predict(input_df[stay_model.feature_names_in_])[0]

        # Predict cost
        predicted_cost = cost_model.predict(input_df[cost_model.feature_names_in_])[0]
        
        
        # Provide health advice based on severity
        health_advice = ""
        if predicted_severity == "Mild":
            health_advice = "Your condition is mild. Maintain a balanced diet, stay hydrated, and get enough rest."
        elif predicted_severity == "Moderate":
            health_advice = "Your condition is moderate. Consider consulting a healthcare professional and monitor your symptoms closely."
        elif predicted_severity == "Severe":
            health_advice = "Your condition is severe. Seek immediate medical attention and follow professional advice."

        return render_template(
            'result.html',
            disease=predicted_disease,
            severity=predicted_severity,
            stay=round(predicted_stay, 2),
            cost=round(predicted_cost, 2),
            advice=health_advice
        )

    return render_template('predict.html')


@app.route('/performance')
def performance():
    if 'username' not in session:
        return redirect(url_for('login'))

    # Example performance data
    accuracy = 0.99  # Replace with actual computation
    report = """
    Precision: 0.98
    Recall   : 0.96
    F1-Score : 0.99"""
    confusion_matrix_path = '/static/confusion_matrix.png'

    return render_template(
        'performance.html',
        accuracy=accuracy,
        report=report,
        confusion_matrix_path=confusion_matrix_path
    )

@app.route('/feedback')
def feedback_page():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('feedback.html')

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    if 'username' not in session:
        return redirect(url_for('login'))

    user_feedback = request.form['feedback']
    # Save feedback to a file or database (optional)
    with open("feedback.txt", "a") as f:
        f.write(user_feedback + "\n")

    return render_template('thank_you.html')

if __name__ == '__main__':
    app.run(debug=True)
