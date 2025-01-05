import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

# Load the model, scaler and feature names
with open('titanic_model.pkl', 'rb') as f:
  model = pickle.load(f)
with open('titanic_scaler.pkl', 'rb') as f:
  scaler = pickle.load(f)
with open('feature_names.pkl', 'rb') as f:
  feature_names = pickle.load(f)

# App title
st.title("Titanic Survival Prediction")

# Input features with appropriate defaults
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Number of Siblings/Spouses", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=30.0) 
embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# Create input DataFrame
input_data = pd.DataFrame({
       'Pclass': [pclass],
       'Sex': [sex],
       'Age': [age],
       'SibSp': [sibsp],
       'Parch': [parch],
       'Fare': [fare],
       'Embarked': [embarked]
   })

# Preprocess input data
input_data = pd.get_dummies(input_data, columns=['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked'])
# Handle missing columns to match training data
missing_cols = set(model.coef_[0]) - set(input_data.columns) # Get columns from model instead
for c in missing_cols:
  input_data[c] = 0
input_data = input_data[feature_names]

input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled)[0]

# Display prediction
if prediction == 1:
  st.write("This passenger is predicted to have survived.")
else:
  st.write("This passenger is predicted to have not survived.")
