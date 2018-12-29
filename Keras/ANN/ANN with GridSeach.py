# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 15:31:59 2018

@author: aanishsingla
"""

# -*- coding: utf-8 -*-

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Training with CV and Dropout
from keras.models import Sequential
from keras.layers import Dense, Dropout, PReLU
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
  
def classifier_fn(optimizer='SGD', dp1=0.12):
    classifier = Sequential()
    classifier.add(Dense(units=6,activation="linear", kernel_initializer="random_uniform",input_dim=11))
    classifier.add(PReLU(input_shape =(6,))) #Parametric RELU
    classifier.add(Dropout(rate = dp1))
    
    classifier.add(Dense(units=6,activation="linear", kernel_initializer="random_uniform"))
    classifier.add(PReLU(input_shape =(6,))) #Parametric RELU
    classifier.add(Dropout(rate = 0.1))
    
    classifier.add(Dense(units=1,activation="sigmoid", kernel_initializer="random_uniform"))
    classifier.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['accuracy'])
    return classifier

# Grid Search for finding the best hyper params    
model =  KerasClassifier(build_fn=classifier_fn, epochs=10, batch_size=25)
params = {'epochs':[10,25,100],
          'batch_size':[10,100],
          'optimizer':['adam', 'rmsprop', 'SGD'],
          'dp1':[0.12, 0.25]
        }
gridSearch = GridSearchCV(estimator=model, param_grid=params,scoring='accuracy', cv=3)
gs = gridSearch.fit(X_train, y_train)           
print(gs.best_score_)
print(gs.best_params_)
#####################################
# Set callback functions to early stop training and save the best model so far
callback_list = [EarlyStopping(monitor='loss', patience=5, min_delta=0.0001,verbose=1, mode='auto'),
             ModelCheckpoint(filepath='best_model.h5', monitor='loss', save_best_only=True)]

model =  KerasClassifier(build_fn=classifier_fn, epochs=100, batch_size=10)
model.fit(x=X_train, y=y_train,callbacks = callback_list,verbose=1)

#CV using the identified no. of epochs
acc = cross_val_score(estimator=model, X=X_train, y=y_train, cv=5, verbose=0)
print(acc.mean())
print(acc.std())
#85.4% Accuracy

# Train on full data
model =  KerasClassifier(build_fn=classifier_fn, epochs=100, batch_size=10)
model.fit(x=X_train, y=y_train,verbose=1)

# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
#86.9% Accuracy


    