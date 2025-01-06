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

# Create an empty DataFrame with the features expected by the model
input_data = pd.DataFrame(columns=feature_names, index=[0]).fillna(0) 

input_data['Age'] = age
input_data['Fare'] = fare
input_data[f'Pclass_{pclass}'] = 1  # One-hot encode Pclass
input_data[f'Sex_{sex}'] = 1  # One-hot encode Sex
input_data[f'SibSp_{sibsp}'] = 1  # One-hot encode SibSp
input_data[f'Parch_{parch}'] = 1  # One-hot encode Parch
input_data[f'Embarked_{embarked}'] = 1 # One-hot encode Embarked

input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled)[0]

# Display prediction
if prediction == 1:
  st.write("This passenger is predicted to have survived.")
else:
  st.write("This passenger is predicted to have not survived.")
