# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 06:26:55 2018

@author: aanishsingla
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#dummies = pd.get_dummies(X[:,1],prefix='Geo_', drop_first=True) 
#X = pd.concat([X, dummies.iloc[:,:].values], axis=1)
#X = np.delete(X, 1, 1) 

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ANN

# Import Keras 
#import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialize ANN
classifier = Sequential()

#Add Input and first hidden layer
classifier.add(Dense(units=6,activation="relu", kernel_initializer="uniform",input_dim=11))

#2nd Hidden layer
classifier.add(Dense(units=6,activation="relu", kernel_initializer="uniform"))

#Output Layer
classifier.add(Dense(units=1,activation="sigmoid", kernel_initializer="uniform"))

#ANN Complilation
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

# Fitting the classifier
classifier.fit(X_train, y_train, batch_size=1, epochs=10)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
#####################################################################
