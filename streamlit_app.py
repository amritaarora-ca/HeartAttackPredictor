# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the dataset
file_path = 'Heart Attack Data Set.csv'  # Replace with your file path if needed
data = pd.read_csv(file_path)

# Split data into features (X) and target (y)
X = data.drop('target', axis=1)
y = data['target']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model using joblib
from joblib import dump
dump(model, 'heart_attack_model.joblib')

# Streamlit App
st.title("Heart Attack Risk Predictor")

# Sidebar for user input
st.sidebar.header("User Input Parameters")

# Function to collect user input
def user_input_features():
    age = st.sidebar.slider('Age', int(X['age'].min()), int(X['age'].max()), 50)
    sex = st.sidebar.selectbox('Sex (0 = Female, 1 = Male)', [0, 1])
    cp = st.sidebar.slider('Chest Pain Type (0-3)', 0, 3, 1)
    trestbps = st.sidebar.slider('Resting Blood Pressure', int(X['trestbps'].min()), int(X['trestbps'].max()), 120)
    chol = st.sidebar.slider('Cholesterol Level', int(X['chol'].min()), int(X['chol'].max()), 200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (0 = No, 1 = Yes)', [0, 1])
    restecg = st.sidebar.slider('Resting ECG Results (0-2)', 0, 2, 1)
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved', int(X['thalach'].min()), int(X['thalach'].max()), 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina (0 = No, 1 = Yes)', [0, 1])
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise', float(X['oldpeak'].min()), float(X['oldpeak'].max()), 1.0)
    slope = st.sidebar.slider('Slope of Peak Exercise ST Segment (0-2)', 0, 2, 1)
    ca = st.sidebar.slider('Number of Major Vessels Colored by Fluoroscopy (0-4)', 0, 4, 0)
    thal = st.sidebar.slider('Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)', 0, 2, 1)

    # Combine inputs into a DataFrame
    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Display user inputs
user_input = user_input_features()
st.subheader("User Input Parameters")
st.write(user_input)

# Load the trained model
from joblib import load
model = load('heart_attack_model.joblib')

# Make prediction
prediction = model.predict(user_input)
prediction_proba = model.predict_proba(user_input)

# Display the prediction
st.subheader("Prediction")
result = "Prone to Heart Attack" if prediction[0] == 1 else "Not Prone to Heart Attack"
st.write(result)

# Display prediction probabilities
st.subheader("Prediction Probability")
st.write(prediction_proba)
