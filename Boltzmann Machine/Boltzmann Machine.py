# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 07:53:25 2019

@author: aanishsingla
Implements Restricted Boltzmann Manchine for Binary Recommendation System
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.autograd as Variable
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

#Convert to Binary - 1 (Liked) 0 (Not Liked). Replace Not rated ones to -1
training_set[training_set==0] = -1
training_set[training_set==1] = 0
training_set[training_set==2] = 0
training_set[training_set==3] = 1
training_set[training_set==4] = 1
training_set[training_set==5] = 1

test_set[test_set==0] = -1
test_set[test_set==1] = 0
test_set[test_set==2] = 0
test_set[test_set>=3] = 1
"""
We have to create a class with three functions for the Restricted Boltzmann
Machine which it will obey.
        
1. Initialise tensors of all weights and biases of the  visible nodes and
   hidden nodes. Add weight parameter of the probabilities of the visible 
   nodes according to the hidden nodes.
2. Sample hidden nodes
   For every each hidden node activate them for a given probablity given v.
   In which the activation is a linear function of the neurons where the 
   coefficients are the functions. So, the activation is probability that the
   hidden node will be activated according to the value of the visible node. 
   The activation is returned as a sigmoid function. But we're making a 
   Bernoulli RBM. p[h|v] is vector of nh elements, each element corresponds to 
   each hidden node. We use this probabilities to sample activation of each 
   hidden node, depending on p[h|v] for v. If randn < 0.7 = activate neuron, 
   and if randn > 0.7 = not activate neuron. Obtain vector with a binary outcome 
   to list which hidden nodes activated or not activated.
3. Sample visible nodes.
   If randn < 0.25 = activate neuron, 
   and if randn > 0.25 = not activate neuron. Obtain vector with a binary outcome 
   to list which hidden nodes activated or not activated.
"""
class RBM():
    #Initializes the weights and Biases according to Normal Dist
    def __init__(self, nv, nh): 
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1,nh) #Bias from v to h 
        self.b = torch.randn(1,nv)  # Bias fom h to v
    # Finds the activation of hidden layer gven visible layer     
    def sample_h(self,x): 
        wx = torch.mm(x,self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)       
    # Finds the activation of visible layer gven hidden layer     
    def sample_v(self,y): 
        wy = torch.mm(y,self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)  
    #Updates weight after k iterations of contrastive divergence
    def update_weights(self, v0, vk, ph0, phk):
#        print("\nInside weights: W="+str(self.W.shape))
#        print("Inside weights: vo="+str(v0.shape))
#        print("Inside weights: ph0="+str(ph0.shape))
#        self.W += torch.mm(ph0,v0) - torch.mm(phk,vk)
        self.b += torch.sum((v0-vk),0)
        self.a += torch.sum((ph0-phk),0)
        #return self.W, self.a, self.b

nv = len(training_set[0])
nh = 100 # No. of features
rbm = RBM(nv, nh)

batch_size = 100 
nb_epoch = 10

for epoch in range(1,nb_epoch+1):
    loss = 0.
    s=0.
    for id_user in range(0,nb_user-batch_size , batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        
        for k in range(10):
            _,hk = rbm.sample_h(vk) 
            _,vk = rbm.sample_v(hk)
            vk[v0<0]=v0[v0<0]  #Dont update the not rated values
        phk,_ = rbm.sample_h(vk)
#        print("len v0="+str(len(v0[0])))
#        print("len vk="+str(len(vk)))
#        print("len ph0="+str(len(ph0[0])))
#        print("len phk="+str(len(ph0)))
        rbm.update_weights(v0,vk,ph0,phk)    
               
        loss += torch.mean(torch.abs(v0[v0>=0]-vk[v0>=0]))
        s += 1.
    print("\nEpoch:"+str(epoch)+ " Loss:"+str(loss/s))

#Testing
test_loss = 0.
s=0.
for id_user in range(nb_user):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)        
        s+=1.       
        test_loss += torch.mean(torch.abs(v[vt>=0]-vt[vt>=0]))
print("\nTesting Loss:"+str(test_loss/s))   
