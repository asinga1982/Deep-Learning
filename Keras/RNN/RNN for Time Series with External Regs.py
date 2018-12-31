# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 21:42:25 2018

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

training_set = dataset_train[["Open","Volume"]]
#Remove Commas form Volume numbers
training_set['Volume'] = training_set['Volume'].str.replace(',','')   
training_set['Volume'] = training_set['Volume'].astype(int)

plt.plot(training_set['Volume'])


from sklearn.preprocessing import MinMaxScaler
sc  = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)
#                   (len(training_set),1))

#sc2  = MinMaxScaler(feature_range=(0,1))
#training_set_scaled_2 = np.reshape(sc2.fit_transform(training_set[['Volume']]),
#                   (len(training_set),1))

#training_set_scaled = pd.Series({'Open':training_set_scaled_1, 
#                                    'Volume':training_set_scaled_2})

#training_set_scaled= training_set_scaled.to_frame()

#Set up data for training with 60 time steps
X_train = []
y_train =[]

for i in range(60,len(training_set_scaled)):
    y_train.append(training_set_scaled[i,0])
    X_train.append(training_set_scaled[i-60:i,0:2])

X_train, y_train = np.array(X_train) , np.array(y_train) #Convert to Array

#Reshape as needed by Keras
X_train =  np.reshape(X_train, (X_train.shape[0],X_train.shape[1],2))

#Build a RNN

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

model = Sequential()

#LSTM layer 1
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],2)))
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

model.fit(X_train, y_train, epochs=100,batch_size=32, validation_split=0.2)

#Test Data - Jan
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_prices = dataset_test[["Open", "Volume"]]
#Remove Commas form Volume numbers
real_prices['Volume'] = real_prices['Volume'].str.replace(',','')   
real_prices['Volume'] = real_prices['Volume'].astype(int)

total_ds = pd.concat((training_set,real_prices),axis=0)
total_ds = total_ds.iloc[len(training_set)-60:,:].values

inputs = sc.transform(total_ds)

X_test = []

for i in range(60,len(inputs)):
    X_test.append(inputs[i-60:i,0:2])

X_test = np.array(X_test)

#Reshape as needed by Keras
X_test1 =  np.reshape(X_test, (X_test.shape[0],X_test.shape[1],2))

pred = model.predict(x=X_test1)

#generate Dummy values for volume
import random
my_randoms = [random.randrange(0, 1, 1) for _ in range(len(pred))]

#Prepare data in the format that scaler expects
data1 = np.asmatrix(total_ds[1:21,:])
data1[:,0] =  pred
data1[:,1] = np.array(my_randoms).reshape(20,1)

yy=sc.inverse_transform(data1)

#Compare the predictions
plt.plot(yy[:,0],color="red", label= "Predicted Values")
plt.plot(real_prices["Open"], color="blue", label= "Actual Values")
plt.legend()
plt.show()

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_prices["Open"], yy[:,0]))
print(rmse)
