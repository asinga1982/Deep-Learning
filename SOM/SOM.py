# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 12:32:13 2019

@author: aanishsingla
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Pre-procesing
import os
print(os.getcwd())
os.chdir('E:\Python Dir\Deep_Learning_A_Z\Deep_Learning_A_Z\Volume 2 - Unsupervised Deep Learning\Part 4 - Self Organizing Maps (SOM)\Section 16 - Building a SOM') 

dataset = pd.read_csv("Credit_Card_Applications.csv")
#Split the dataset
X= dataset.iloc[:,:-1]
y=dataset.iloc[:,-1].values

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_scaled = sc.fit_transform(X)

#Build Self Organizing Map
from minisom import MiniSom 

som = MiniSom(x=10,y=10,input_len=15)
som.random_weights_init(X_scaled)
som.train_random(X_scaled, num_iteration=100)

#Visualize the result
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()

markers = ["o","s"]
colors = ["r","g"]

for i,x in enumerate(X_scaled):
    w = som.winner(x)
    plot(w[0]+0.5,
         w[1]+0.5,
         markers[y[i]],
         markeredgecolor= colors[y[i]],
         markerfacecolor="None",
         markersize=10,
         markeredgewidth=2)
show()

mapping = som.win_map(X_scaled)    

x=np.concatenate((mapping[(6,7)], mapping[(6,6)]),axis=0)

frauds = pd.DataFrame(sc.inverse_transform(x))

#Approved Frauds
dataset[dataset["CustomerID"].isin(frauds[0]) & dataset['Class']==1 ]

