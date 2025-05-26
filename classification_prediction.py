

from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import pickle
import pandas as pd
import numpy as np
import json


## load trained model ,scaler pickle,onehot
model=load_model("model.h5")
with open("onehot_encoder_geo.pkl","rb") as file:
    OneHotEncoder_geo=pickle.load(file)
with open("label_encoder_gender.pkl","rb") as file:
    label_encoder_gender=pickle.load(file)
with open("scaler.pkl","rb") as file:
    scaler=pickle.load(file)

# Example input data
input_data = {
    'CreditScore': 650,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000
}

geo_encoded = OneHotEncoder_geo.transform([[input_data["Geography"]]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=OneHotEncoder_geo.get_feature_names_out(["Geography"]))
#print(geo_encoded_df)
input_df=pd.DataFrame([input_data])
#print(input_df)
input_df["Gender"]=label_encoder_gender.transform(input_df["Gender"])  #encoding gender

input_df=pd.concat([input_df.drop("Geography",axis=1),geo_encoded_df],axis=1)
#print(input_df)
input_scaled=scaler.transform(input_df)
#print(input_scaled)

###Prediction

prediction=model.predict(input_scaled)
prediction_proba=prediction[0][0]

#print(prediction)
#print(prediction_proba)

if prediction_proba > 0.5:
    print("the customer is likely to churn")
else:
    print("the customer is not likely to churn")

