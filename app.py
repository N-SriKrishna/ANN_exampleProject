
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle
import json


## Load the trained model
model = tf.keras.models.load_model("model.h5")

with open("onehot_encoder_geo.pkl","rb") as file:
    OneHotEncoder_geo=pickle.load(file)
with open("label_encoder_gender.pkl","rb") as file:
    label_encoder_gender=pickle.load(file)
with open("scaler.pkl","rb") as file:
    scaler=pickle.load(file)

## streamlit app
st.title("Customer Churn Prediction")

#input
geography = st.selectbox("Geography",OneHotEncoder_geo.categories_[0])
gender = st.selectbox("Gender",label_encoder_gender.classes_)
age = st.slider("Age",18,92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tensure",0,10)
num_of_products = st.slider("Number of Products",1,4)
has_cr_card = st.selectbox("Has credit card",[0,1])
is_active_member = st.selectbox("Is Active Member",[0,1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-Hot encoded Geography
geo_encoded = OneHotEncoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=OneHotEncoder_geo.get_feature_names_out(["Geography"]))

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

input_data_scaled = scaler.transform(input_data)

#prediction
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

if prediction_proba > 0.5:
    st.write("The customer is likely to churn")
else:
    st.write("The customer is not likely to churn")

st.write(f"Current Churn probability is {prediction_proba}")

with open("training_history.json", "r") as file:
    training_history = json.load(file)

st.write(f"Training accuracy is {training_history['accuracy'][-1]}")
