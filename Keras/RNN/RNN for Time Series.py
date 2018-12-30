# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 21:42:25 2018

@author: aanishsingla
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Pre-procesing
import os
print(os.getcwd())
os.chdir('E:\Python Dir\Deep_Learning_A_Z\Deep_Learning_A_Z\Volume 1 - Supervised Deep Learning\Part 3 - Recurrent Neural Networks (RNN)\Section 12 - Building a RNN') 

dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")

training_set = dataset_train.iloc[:,1:2]

plt.plot(training_set)

from sklearn.preprocessing import MinMaxScaler
sc  = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

#Set up data for training with 60 time steps
X_train = []
y_train =[]

for i in range(60,len(training_set_scaled)):
    y_train.append(training_set_scaled[i,0])
    X_train.append(training_set_scaled[i-60:i,0])

X_train, y_train = np.array(X_train) , np.array(y_train) #Convert to Array

#Reshape as needed by Keras
X_train =  np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

#Build a RNN

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

model = Sequential()

#LSTM layer 1
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(Dropout(rate=0.2))

#LSTM layer 2
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(rate=0.2))

#LSTM layer 3
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(rate=0.2))

#LSTM layer 4
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(rate=0.2))

#Output Layer
model.add(Dense(units=1))

model.compile(optimizer="adam", loss="mean_squared_error")

model.fit(X_train, y_train, epochs=100,batch_size=32)


#Test Data - Jan
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_prices = dataset_test.iloc[:,1:2]

total_ds = pd.concat((training_set,real_prices),axis=0)
total_ds = total_ds.iloc[len(training_set)-60:,:].values

inputs = sc.transform(total_ds)

X_test = []

for i in range(60,len(inputs)):
    X_test.append(inputs[i-60:i,0])

X_test = np.array(X_test)

#Reshape as needed by Keras
X_test =  np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

pred = model.predict(x=X_test)

#Compare the predictions
plt.plot(sc.inverse_transform(pred),color="red", label= "Predicted Values")
plt.plot(real_prices, color="blue", label= "Actual Values")
plt.legend()
plt.show()

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_prices, sc.inverse_transform(pred)))
print(rmse)
