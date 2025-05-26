## solving a binary classification problem using a neural network

## steps followed :
# 1. take a dataset
# 2. Basic feature engineering ( variables into numericals , standardization )
# 3. Create a neural network model ( using keras and tensorflow )
# 4. Compile the model ( loss function , optimizer , metrics )
# 5. Using streamlit to create a web app

from altair import Fit
from numpy import histogram
import pandas as pd # type: ignore
from sklearn import metrics
from sklearn.model_selection import train_test_split # type:ignore
from sklearn.preprocessing import StandardScaler,LabelEncoder # type: ignore
import pickle
import json
import tensorboard

data=pd.read_csv("Churn_Modelling.csv")
#print(data)

## preprocess the data 

# 1)drop irrelevent info

data = data.drop(["RowNumber","CustomerId","Surname"],axis=1)
#print(data)

# 2)encode categorical variables
label_encoder_gender=LabelEncoder()
data["Gender"]=label_encoder_gender.fit_transform(data["Gender"])
#print(data)

# using onehot encoding for geography
from sklearn.preprocessing import OneHotEncoder # type:ignore
onehot_encoder_geo=OneHotEncoder() 
geo_encoder=onehot_encoder_geo.fit_transform(data[["Geography"]])
#print(geo_encoder)
#print(onehot_encoder_geo.get_feature_names_out(["Geography"]))
#print(geo_encoder.toarray())
geo_encoded_df=pd.DataFrame(geo_encoder.toarray(),columns=onehot_encoder_geo.get_feature_names_out(["Geography"]))
#print(geo_encoded_df)

#combine one hot encoded data with original data
data=pd.concat([data.drop("Geography",axis=1),geo_encoded_df],axis=1)
#print(data)

##save encoders and scalers
with open("label_encoder_gender.pkl","wb") as file:
    pickle.dump(label_encoder_gender,file)

with open("onehot_encoder_geo.pkl","wb") as file:
    pickle.dump(onehot_encoder_geo,file)

##Divide the dataset into independent and dependent features
X=data.drop("Exited",axis=1)
y=data["Exited"]

##Split the data in training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

##scale these features
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)
#print(X_train.shape)  #(8000,12)

with open("scaler.pkl","wb") as file:
    pickle.dump(scaler,file)

## 3)create neural network model
#steps followed
  # 1)sequenttial Neural network
  # 2)dense 
  # 3)Activation funtion = sigmoid , softmax , relu 
  # 4)Optimizer -> for updating weights
  # 5)Loss function minimization
  # 6)Metrics = accuracy or error
  # 7)Training = save logs in folder for tensorboard visualization

import tensorflow
import tensorflow as tf 
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard # type: ignore
import datetime

model = Sequential([
    Dense(64,activation="relu",input_shape=(X_train.shape[1],)),  #hidden layer 1
    Dense(32,activation="relu"), #hiddel layer 2 ,no need to give shape as we used sequential
    Dense(1,activation="sigmoid") #HL3 = output layer
    ])

#print(model.summary()) #trainable paramters = 2945

## 4)compile the model

opt=tensorflow.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt,loss="binary_crossentropy",metrics=["accuracy"])


# setup tensorboard

log_dir="logs/fit" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorflow_callback=TensorBoard(log_dir=log_dir,histogram_freq=1)

# setup Early stopping

early_stopping_callback=EarlyStopping(monitor="val_loss",patience=40,restore_best_weights=True)

## Training the model

history =model.fit(
    X_train,y_train,validation_data=(X_test,y_test),epochs=200,
    callbacks=[tensorflow_callback,early_stopping_callback]
)

model.save("model.h5")    ##Latest accuracy = 0.899

## load Tensorboard Extension

#tensorboard --logdir=logs/fit20250526-122835      #in terminal                     

with open("training_history.json","w") as file:
    json.dump(model.history.history,file)

