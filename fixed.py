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
I=31
M=30
div=1 #time-scaling factor div is the divisor of 1 minute, for e.g. for subsampled every 1/10 min, div=10, for every 2 min div=1/2
A=np.zeros((L,M))               #birth parameter
B*=(1./div)
delay=np.zeros((L,M))
a_A=np.zeros(L+1)               #birth shape hyperparameter
b_A=np.zeros(L+1)               #death shape hyperparameter
a_delay=np.zeros(L+1)           #alpha shape hyperparameter
b_delay=np.zeros(L+1)            #beta shape hyperparameter
r=np.zeros((2,I-1,M))     #reaction numbers
sum_x=np.zeros(M)   

var_delay=1
var_a_A=5                      #birth shape hyperparameter proposal variance
var_a_delay=5                  #delay shape hyperparameter proposal variance
tune=300                    #reaction proposal tuning parameter

#initial conditions
A[0,:]=1
delay[0,:]=1
a_A[0]=500
b_A[0]=1 
a_delay[0]=500
b_delay[0]=1

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

def a_lik(A,a,b,prev,var_param):           #a_A and a_delay likelihood function 
    q=-M*math.lgamma(a)+M*a*np.log(b)+(a-1)*np.sum(np.log(A))+norm.logcdf(prev/var_param,0,1)
    return q

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

def MH_a_A(A,a_A,b_A):
    a_Aprop=-1
    while a_Aprop<0:
        a_Aprop=a_A+np.random.normal(0,var_a_A)
    a=a_lik(A,a_Aprop,b_A,a_A,var_a_A) 
    b=a_lik(A,a_A,b_A,a_Aprop,var_a_A)
    rate=np.minimum(1,np.exp(a-b))
    if np.random.uniform(0,1,1)<rate:    
        k=a_Aprop
    else:
        k=a_A
    return k

def MH_a_delay(delay,a_delay,b_delay):
    a_delayprop=-1
    while a_delayprop<0:
        a_delayprop=a_delay+np.random.normal(0,var_a_delay)
    a=a_lik(delay,a_delayprop,b_delay,a_delay,var_a_delay) 
    b=a_lik(delay,a_delay,b_delay,a_delayprop,var_a_delay)
    rate=np.minimum(1,np.exp(a-b))
    if np.random.uniform(0,1,1)<rate:    
        k=a_delayprop
    else:
        k=a_delay
    return k
# In[5]:
#initialize number of reactions
for s in range(M):
    r[:,:,s]= init_reaction(x[:,s],I)


# In[6]:
#perform sampling       
for i in range(L-1): 
    for s in range(M):            
        #sample parameters
        A[i+1,s]=np.random.gamma(np.sum(r[0,:,s])+a_A[i],(I-1-delay[i,s]+b_A[i])**(-1))
        #print('pass A,B')
        
        #update parameter DELAY
        delay[i+1,s]=MH_delay(A[i+1,s],delay[i,s],a_delay[i],b_delay[i],r[0,:,s])
        
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
        
    #sample hyperparameters a_A, a_B, and a_delay
    a_A[i+1]=MH_a_A(A[i+1,:],a_A[i],b_A[i])
    a_delay[i+1]=MH_a_delay(delay[i+1,:],a_delay[i],b_delay[i])
    
    #sample hyperparameters b_A, b_B, b_alpha, b_beta
    sum_A=np.sum(A[i+1,:])
    sum_delay=np.sum(delay[i+1,:])
    b_A[i+1]=np.random.gamma(M*a_A[i+1],sum_A**(-1))
    b_delay[i+1]=np.random.gamma(M*a_delay[i+1],sum_delay**(-1))
        
    if i%1 == 0:  
        print(i+1,'birth',a_A[i+1],b_A[i+1],'delay',a_delay[i+1],b_delay[i+1]) 
        print("%%%%%%%%%%%%%%%%%next%%%%%%%%%%%%%%%%")

# In[ ]:
#save the simulated values into .csv files
columns=' '
np.savetxt("A.csv", A, delimiter=",",header=columns)
#np.savetxt("B1.csv", B, delimiter=",",header=columns)
np.savetxt("delay.csv", delay, delimiter=",",header=columns)
np.savetxt("a_A.csv", a_A, delimiter=",",header=columns)
np.savetxt("b_A.csv", b_A, delimiter=",",header=columns)
np.savetxt("a_delay.csv", a_delay, delimiter=",",header=columns)
np.savetxt("b_delay.csv", b_delay, delimiter=",",header=columns)


# In[ ]:
