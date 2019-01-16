# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 16:38:23 2019

@author: aanishsingla
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim


#Pre-procesing
import os
print(os.getcwd())
os.chdir('E:\Python Dir\Deep_Learning_A_Z\Deep_Learning_A_Z\Volume 2 - Unsupervised Deep Learning\Part 5 - Boltzmann Machines (BM)\Section 19 - Building a Boltzmann Machine\Boltzmann_Machines\Boltzmann_Machines')

movies = pd.read_csv("ml-1m\movies.dat", sep='::', header=None, engine='python',encoding="latin-1")
users = pd.read_csv(r'''ml-1m\users.dat''', sep='::', header=None, engine='python',encoding="latin-1")
ratings = pd.read_csv(r'''ml-1m\ratings.dat''', sep='::', header=None, engine='python',encoding='latin-1')

#Prepare training and test data
training_set = pd.read_csv(r'''ml-100k\u1.base''',delimiter="\t", header=None).values
test_set = pd.read_csv(r'''ml-100k\u1.test''',delimiter="\t", header=None).values

#Format training and test set
def user_movie(training_set, test_set):
    xx=np.concatenate((training_set, test_set), axis=0)
    nb_user = int(max(xx[:,0]))
    nb_movie = int(max(xx[:,1]))
    return nb_user, nb_movie

nb_user, nb_movie = user_movie(training_set, test_set)

#Convert the passed dataset as a list of users i.e. One List per user.
#The inner list has the movies       
def user_movie_list(data):
    new_data = []
    for user in range(1, nb_user+1):
        movie_list = data[:,1][data[:,0]==user]
        rating_list = data[:,2][data[:,0]==user]
        ratings = np.zeros(nb_movie)
        ratings[movie_list-1] = rating_list
        new_data.append(list(ratings))
    return new_data

test_set=user_movie_list(test_set)  
training_set=user_movie_list(training_set)

#Convert the date to Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

#Class for Stacked Auto-Encoder, inherits Pytorch nn Module
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movie, 20) # with  20 hidden layers
        self.fc2 = nn.Linear(20, 10) #2nd hidden layer
        self.fc3 = nn.Linear(10,20)  #Decoding layer       
        self.fc4 = nn.Linear(20,nb_movie) #Final Layer
        self.activation = nn.Sigmoid()
        
    def forward(self,x):  #x is the input vector and is also used a input for subsequesnt layers 
        x=self.activation(self.fc1(x))
        x=self.activation(self.fc2(x))
        x=self.activation(self.fc3(x))
        x=self.fc4(x)  # No activation fc used
        return x     # Retun the prediction

sae = SAE()
criterion = nn.MSELoss() 
optimizer = optim.RMSprop(sae.parameters(),lr=0.01, weight_decay=0.5)

#Training
nb_epoch = 50

for epoch in range(1,nb_epoch+1):
    train_loss = 0.
    s=0.
    for id_user in range(nb_user):
        input = Variable(training_set[id_user]).unsqueeze(0) #Convert to a batch
        target = input.clone()
        if torch.sum(target.data > 0) > 0: 
            pred = sae(input)
            target.require_grad = False
            pred[target==0] = 0
            loss = criterion(pred, target)
            mean_correction = nb_movie/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss +=  np.sqrt(loss.item()*mean_correction)
            optimizer.step()
            s += 1.
    print("Epoch:"+str(epoch)+" Loss:"+str(train_loss/s))

#Testing    
test_loss =0.
s=0.    
for id_user in range(nb_user):
    input = Variable(training_set[id_user]).unsqueeze(0) #Convert to a batch
    target = Variable(test_set[id_user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0: 
        pred = sae(input)
        target.require_grad = False
        pred[target==0] = 0
        loss = criterion(pred, target)
        mean_correction = nb_movie/float(torch.sum(target.data > 0) + 1e-10)
#        loss.backward()
        test_loss +=  np.sqrt(loss.item()*mean_correction)
#        optimizer.step()
        s += 1.
print("Test Set Loss: "+str(test_loss/s))    