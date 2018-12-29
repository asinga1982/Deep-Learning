# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 14:45:03 2018

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

#Training with CV and Dropout
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

def classifier_fn():
    classifier = Sequential()
    classifier.add(Dense(units=6,activation="relu", kernel_initializer="uniform",input_dim=11))
#    classifier.add(Dropout(rate = 0.1))
    
    classifier.add(Dense(units=6,activation="relu", kernel_initializer="uniform"))
    classifier.add(Dropout(rate = 0.1))
    
    classifier.add(Dense(units=1,activation="sigmoid", kernel_initializer="uniform"))
    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
    return classifier
    
model =  KerasClassifier(build_fn=classifier_fn, epochs=10, batch_size=10)
acc = cross_val_score(estimator=model, X=X_train, y=y_train, cv=10, verbose=0)
print(acc.mean())
print(acc.std())

    