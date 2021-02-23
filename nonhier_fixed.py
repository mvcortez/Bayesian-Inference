#!/usr/bin/env python
# coding: utf-8

# In[1]:


#This performs Hierarchical modeling on a partially-observed birth-death process 0->X->0 with birth parameter A and death parameter B.
import pandas as pd
import numpy as np
import math
#import matplotlib.pyplot as plt
#from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import gamma
from scipy.stats import norm
from scipy.stats import poisson
from scipy.stats import skellam
import scipy.special as special


# In[2]:
#import saved data
pd_subsampled_x=pd.read_csv("subsampled.csv")
x= pd_subsampled_x.values
size=np.shape(x)

pd_B=pd.read_csv("B.csv")
B= pd_B.values

# In[3]:

L=10000      #length of the chain
#I=size[0]    #[0,I-1] is the observation window
#M=size[1]    #number of individuals
#initializations
I=41
M=40
div=1 #time-scaling factor div is the divisor of 1 minute, for e.g. for subsampled every 1/10 min, div=10, for every 2 min div=1/2
A=np.zeros((L,M))               #birth parameter
B*=(1./div)
delay=np.zeros((L,M))
r=np.zeros((2,I-1,M))     #reaction numbers
sum_x=np.zeros(M)   

var_delay=1
var_a_A=1                      #birth shape hyperparameter proposal variance
var_a_delay=1                  #delay shape hyperparameter proposal variance
tune=300                    #reaction proposal tuning parameter

#initial conditions
A[0,:]=1
delay[0,:]=1
a_A=0.001
b_A=0.001 
a_delay=0.001
b_delay=0.001

# In[4]:

def init_reaction(x,s):         #initialize reaction numbers           
    q=np.zeros([2,s-1])
    for i in range(s-1):
        if x[i+1]-x[i]>0:
            q[0,i]=np.floor((4/3)*(x[i+1]-x[i]))
            q[1,i]=np.floor((1/3)*(x[i+1]-x[i]))
        else:
            q[0,i]=-np.floor((1/3)*(x[i+1]-x[i]))
            q[1,i]=-np.floor((4/3)*(x[i+1]-x[i]))
    r=q.astype(int)        
    return r  

def accept_rate_react(x,r,prop,i,A,B,jump,b,delay):        #acceptance rate for reaction numbers
    lambda_prop=1+(prop[0]**2)/b
    lambda_cur=1+(r[0]**2)/b
    if i+1<=delay:
        delay_factor=0
    else:
        delay_factor=min(1,i+1-delay)
    prop_like=np.log(poisson.pmf(prop[0],A*delay_factor)+1e-300)+np.log(poisson.pmf(prop[1],0.5*B*(x[0]+x[1]))+1e-300)+np.log(skellam.pmf(abs(jump),lambda_prop,lambda_prop)+1e-300)
    current_like=np.log(poisson.pmf(r[0],A*delay_factor)+1e-300)+np.log(poisson.pmf(r[1],0.5*B*(x[0]+x[1]))+1e-300)+np.log(skellam.pmf(abs(jump),lambda_cur,lambda_cur)+1e-300)
    rate=np.minimum(1,np.exp(prop_like-current_like))
    return rate

def delay_lik(delay,A,a,b,I,r,prev):      #delay loglikelihood 
    sum_kappa=0
    sum_log_kappa=0
    for jj in range(I-1):
        if jj+1<=delay:
            kappa=0
        else:
            kappa=min(1,jj+1-delay)
        log_kappa=r[jj]*np.log(kappa+1e-300)
        sum_kappa+=kappa
        sum_log_kappa+=log_kappa
    q=sum_log_kappa-(A*sum_kappa)+gamma.logpdf(delay,a=a,scale=1./b)+norm.logcdf(prev/var_delay,0,1)
    return q

def MH_delay(A,delay,a_delay,b_delay,r):
    delay_prop=-1
    while delay_prop<0:
        delay_prop=delay+np.random.normal(0,var_delay)
    a=delay_lik(delay_prop,A,a_delay,b_delay,I,r,delay) 
    b=delay_lik(delay,A,a_delay,b_delay,I,r,delay_prop) 
    rate=np.minimum(1,np.exp(a-b))
    #print(rate)
    if np.random.uniform(0,1,1)<rate:    
        k=delay_prop
    else:
        k=delay
    return k

# In[5]:
#initialize number of reactions
for s in range(M):
    r[:,:,s]= init_reaction(x[:,s],I)
    
# In[6]:
#perform sampling       
for s in range(M):            
    for i in range(L-1): 
        #sample parameters
        A[i+1,s]=np.random.gamma(np.sum(r[0,:,s])+a_A,(I-1-delay[i,s]+b_A)**(-1))
        #print('pass A,B')
        
        #update parameter DELAY
        delay[i+1,s]=MH_delay(A[i+1,s],delay[i,s],a_delay,b_delay,r[0,:,s])
        
        #update reaction numbers
        for j in range(I-1):
            r_vprop=np.array([-1,-1])
            lamb=1+(r[0,j,s]**2)/tune                 #tuning parameter suggested in Boy's paper 
            while r_vprop[0]<0 or r_vprop[1]<0:
                jump=np.random.poisson(lamb)-np.random.poisson(lamb)
                r_prop=r[0,j,s]+jump
                r_vprop=[r_prop, r_prop-(x[j+1,s]-x[j,s])]    
            rate=accept_rate_react([x[j,s],x[j+1,s]],r[:,j,s],r_vprop,j,A[i+1,s],B[s],jump,tune,delay[i+1,s]) 
            if np.random.uniform(0,1,1)<rate:
                r[:,j,s]=r_vprop    
        #print('pass r')   

        if i%1000 == 0:  
            print('Individual %d' %(s))
            print(i+1,'A',A[i+1,s],'delay',delay[i+1,s]) 
            print("%%%%%%%%%%%%%%%%%next%%%%%%%%%%%%%%%%")

# In[ ]:
#save the simulated values into .csv files
columns=' '
np.savetxt("A.csv", A, delimiter=",",header=columns)
#np.savetxt("B1.csv", B, delimiter=",",header=columns)
np.savetxt("delay.csv", delay, delimiter=",",header=columns)
# In[ ]:
