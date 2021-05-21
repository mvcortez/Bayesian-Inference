#!/usr/bin/env python
# coding: utf-8

# In[1]:
#This performs hierarchical modeling on a partially-observed birth-death process 0->X->0 with birth parameter A and death parameter B.
import pandas as pd
import numpy as np
import math
#import matplotlib.pyplot as plt
#from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import gamma
from scipy.stats import norm
from scipy.stats import poisson
from scipy.stats import skellam
import scipy.integrate as integrate
#import scipy.special as special

# In[2]:
#import saved data
pd_subsampled_x=pd.read_csv("YFP2.csv")
x= pd_subsampled_x.values
size=np.shape(x)

pd_B=pd.read_csv("B2.csv")
B= pd_B.values

# In[3]:
L=10000     #length of the chain
I=size[0]    #[0,I-1] is the observation window
M=size[1]    #number of individuals
#I=21
#M=20
div=1         #time-scaling factor div is the divisor of 1 minute, for e.g. for subsampled every 1/2 min, n=2

#initializations
A=np.zeros((L,M))               #birth parameter
alpha=np.zeros((L,M))           #alpha parameter
beta=np.zeros((L,M))            #beta parameter
a_A=np.zeros(L+1)               #birth shape hyperparameter
a_alpha=np.zeros(L+1)           #alpha shape hyperparameter
a_beta=np.zeros(L+1)            #beta shape hyperparameter
b_A=np.zeros(L+1)               #birth rate hyperparameter
b_alpha=np.zeros(L+1)           #alpha rate hyperparameter
b_beta=np.zeros(L+1)            #beta rate hyperparameter
v_int_gammainc=np.zeros(I-1)    #vector of average completion propensities
r=np.zeros((2,I-1,M))           #reaction numbers

#initial conditions
A[0,:]=1
#B[0,:]=1
alpha[0,:]=10
beta[0,:]=1
a_A[0]=1
b_A[0]=1
#a_B[0]=1
#b_B[0]=1
a_alpha[0]=10
b_alpha[0]=1
a_beta[0]=1
b_beta[0]=1

#proposal distribution variances and parameters
var_alpha=1                     #alpha proposal variance
var_beta=0.1                     #alpha proposal variance
var_a_A=1                       #birth shape hyperparameter proposal variance
var_a_alpha=1                       #birth shape hyperparameter proposal variance
var_a_beta=1                       #birth shape hyperparameter proposal variance
tune=300                        #reaction proposal tuning parameter
tune_beta=600


# In[4]:
def init_reaction(x):         #initialize reaction numbers   
    '''
    Initializes the number of completed reactions between all time intervals for an individual n 
    :param x: subsampled trajectory of molecular counts for individual n
    :return: q
    '''     
    q=np.zeros([2,I-1])
    for i in range(I-1):
        if x[i+1]-x[i]>0:
            q[0,i]=x[i+1]-x[i]
            q[1,i]=0
        else:
            q[0,i]=0
            q[1,i]=-(x[i+1]-x[i])      
    return q.astype(int)  

def alpha_lik(alpha,beta,A,a,b,r,var):      #logposterior function for alpha
    '''
    Computes the log-posterior function evaluated at alpha_n for an individual n
    :param alpha: parameter alpha_n for individual n 
    :param beta:  parameter beta_n for individual n
    :param A: current sample for A_n for individual n
    :param a: hyperparameter a_alpha
    :param b: hyperparameter b_alpha
    :param r: number of completed birth reactions at all intervals for individual n 
    :return: q
    ''' 
    kappa=int_gammainc(alpha,beta)
    sum_log_gammainc = np.dot(r,np.log(kappa+1e-100))
    q=sum_log_gammainc-(A*np.sum(kappa))+gamma.logpdf(alpha,a=a,scale=1./b)+norm.logcdf(var/var_alpha,0,1)
    return q

def beta_lik(alpha,beta,A,a,b,r,var):    #logposterior function for beta
    '''
    Computes the log-posterior function evaluated at beta_n for an individual n
    :param alpha: parameter alpha_n for individual n 
    :param beta:  parameter beta_n for individual n
    :param A: current sample for A_n for individual n
    :param a: hyperparameter a_alpha
    :param b: hyperparameter b_alpha
    :param r: number of completed birth reactions at all intervals for individual n 
    :param var: value at which the proposal distribution should be evaluated at
    :return: q
    ''' 
    kappa=int_gammainc(alpha,beta)
    sum_log_gammainc = np.dot(r,np.log(kappa+1e-100))
    q=sum_log_gammainc-(A*np.sum(kappa))+gamma.logpdf(beta,a=a,scale=1./b)+gamma.logpdf(var,a=beta*tune_beta,scale=1./tune_beta)
    return q

def a_lik(param,a,b,prev,var_a_param):           #logposterior function for a_Z for Z = A,B,alpha,beta
    '''
    Computes the log-posterior function evaluated at a_A conditioned at b_A and vector A
    :param param: Vector of param_n values at the current iteration for all individuals n 
    :param a: hyperparameter a_param 
    :param b: hyperparameter b_param
    :param prev: a_param[i]    
    :return: q
    ''' 
    q=-M*math.lgamma(a)+M*a*np.log(b)+(a-1)*np.sum(np.log(param))+norm.logcdf(prev/var_a_param,0,1)
    return q


def int_gammainc(alpha,beta):
    '''
    Computes the average completion propensty on the time interval [m,m+1] for an individual n
    :param alpha: current alpha_n sample for individual n 
    :param beta: current beta_n sample for individual n
    :param m: time index    
    :return: k
    '''
    q=np.zeros(I-1)
    for m in range(I-1):
        k,s=integrate.quad(lambda x: gamma.cdf(x,a=alpha,scale=beta**(-1)),m/div,(m+1)/div)
        q[m]=k
    return q

def accept_rate_react(x,r,prop,A,B,jump,kappa):        #acceptance rate for reaction numbers
    '''
    Computes the acceptance rate for the number of completed reactions at a given time interval for a specific individual n
    :param x: vector [x_i, x_i+1], the molecular counts at the time i and time i+1
    :param r: vector [r_birth, r_death] the current number of completed reactions of each type 
    :param prop: vector [prop_r_birth, prop_r_death] the proposed number of completed reactions of each type 
    :param A: latest sample for A_n 
    :param B: latest sample for B_n
    :param jump: sampled from the Skellam distribution that serves to augment the current reaction numbers
    :param kappa: average completion propensity on the time interval [i,i+1]
    :return: rate
    ''' 
    lambda_prop=1+(prop[0]**2)/tune
    lambda_cur=1+(r[0]**2)/tune
    prop_like=poisson.logpmf(prop[0],A*kappa)+poisson.logpmf(prop[1],0.5*(1./div)*B*(x[0]+x[1]))+skellam.logpmf(abs(jump),lambda_prop,lambda_prop)
    current_like=poisson.logpmf(r[0],A*kappa)+poisson.logpmf(r[1],0.5*(1./div)*B*(x[0]+x[1]))+skellam.logpmf(abs(jump),lambda_cur,lambda_cur)
    rate=np.minimum(1,np.exp(prop_like-current_like))
    return rate

def MH_alpha(A,alpha,beta,a_alpha,b_alpha,r):
    alpha_prop=-1
    while alpha_prop<0:
        alpha_prop=alpha+np.random.normal(0,var_alpha)
    a=alpha_lik(alpha_prop,beta,A,a_alpha,b_alpha,r,alpha) 
    b=alpha_lik(alpha,beta,A,a_alpha,b_alpha,r,alpha_prop)
    rate=np.minimum(1,np.exp(a-b))
    if np.isnan(rate)==False and np.random.uniform(0,1,1)<rate:    
        k=alpha_prop
    else:
        k=alpha
    return k

def MH_beta(A,alpha,beta,a_beta,b_beta,r):
    beta_prop=np.random.gamma(beta*tune_beta,1./tune_beta)
    a=beta_lik(alpha,beta_prop,A,a_beta,b_beta,r,beta)   
    b=beta_lik(alpha,beta,A,a_beta,b_beta,r,beta_prop)
    rate=np.minimum(1,np.exp(a-b))
    if np.isnan(rate)==False and np.random.uniform(0,1,1)<rate:    
        k=beta_prop
    else:
        k=beta
    return k

def MH_r(A,B,r,kappa,x,j):
    param=1+(r[0]**2)/tune                 #tuning parameter suggested in Boy's paper 
    r_vprop=np.array([-1,-1]) 
    while r_vprop[0]<0 or r_vprop[1]<0:
        jump=np.random.poisson(param)-np.random.poisson(param)
        r_prop=r[0]+jump
        r_vprop=[r_prop, r_prop-(x[j+1]-x[j])]    
    rate=accept_rate_react([x[j],x[j+1]],r,r_vprop,A,B,jump,kappa) 
    if np.random.uniform(0,1,1)<rate:
        k=r_vprop
    else:
        k=r
    return k

def MH_a_param(param,a_param,b_param,var_a_param):
    a_prop=-1
    while a_prop<0:
        a_prop=a_param+np.random.normal(0,var_a_param)
    a=a_lik(param,a_prop,b_param,a_param,var_a_param) 
    b=a_lik(param,a_param,b_param,a_prop,var_a_param)
    rate=np.minimum(1,np.exp(a-b))
    if np.random.uniform(0,1,1)<rate:    
        k=a_prop
    else:
        k=a_param
    return k

# In[5]:
#initialize number of reactions
for s in range(M):
    r[:,:,s]= init_reaction(x[:,s])


# In[6]:
#perform sampling       
for i in range(L-1):     #i is iteration index   
    for s in range(M):   #s is individual index
        #sample A and B
        A[i+1,s]=np.random.gamma(np.sum(r[0,:,s])+a_A[i],(sum(v_int_gammainc)+b_A[i])**(-1))
             
        #update parameter alpha
        alpha[i+1,s]=MH_alpha(A[i+1,s],alpha[i,s],beta[i,s],a_alpha[i],b_alpha[i],r[0,:,s])
        beta[i+1,s]=MH_beta(A[i+1,s],alpha[i+1,s],beta[i,s],a_beta[i],b_beta[i],r[0,:,s])
        
        #update gamma incomplete function values
        v_int_gammainc = int_gammainc(alpha[i+1,s],beta[i+1,s]) 
            
        #update reaction numbers
        for j in range(I-1):
            r[:,j,s]=MH_r(A[i+1,s],B[s],r[:,j,s],v_int_gammainc[j],x[:,s],j)    
           
    #sample hyperparameters a hyperparameters
    a_A[i+1]=MH_a_param(A[i+1,:],a_A[i],b_A[i],var_a_A)
    a_alpha[i+1]=MH_a_param(alpha[i+1,:],a_alpha[i],b_alpha[i],var_a_alpha)
    a_beta[i+1]=MH_a_param(beta[i+1,:],a_beta[i],b_beta[i],var_a_beta)
   
    #sample hyperparameters b_A, b_alpha, b_beta
    sum_A=np.sum(A[i+1,:])
    sum_alpha=np.sum(alpha[i+1,:])
    sum_beta=np.sum(beta[i+1,:])
    b_A[i+1]=np.random.gamma(M*a_A[i+1],sum_A**(-1))
    b_alpha[i+1]=np.random.gamma(M*a_alpha[i+1],sum_alpha**(-1))
    b_beta[i+1]=np.random.gamma(M*a_beta[i+1],sum_beta**(-1))
    
      
    if i%1 == 0:  
        print(i+1,'birth',a_A[i+1],b_A[i+1]) 
        print(i+1,'alpha',a_alpha[i+1],b_alpha[i+1],'beta',a_beta[i+1],b_beta[i+1]) 
        print("%%%%%%%%%%%%%%%%%next%%%%%%%%%%%%%%%%")

# In[ ]:
#save the simulated values into .csv files
columns=' '
np.savetxt("A.csv", A, delimiter=",",header=columns)
np.savetxt("a_A.csv", a_A, delimiter=",",header=columns)
np.savetxt("b_A.csv", b_A, delimiter=",",header=columns)
np.savetxt("alpha.csv", alpha, delimiter=",",header=columns)
np.savetxt("beta.csv", beta, delimiter=",",header=columns)
np.savetxt("a_alpha.csv", a_alpha, delimiter=",",header=columns)
np.savetxt("a_beta.csv", a_beta, delimiter=",",header=columns)
np.savetxt("b_alpha.csv", b_alpha, delimiter=",",header=columns)
np.savetxt("b_beta.csv", b_beta, delimiter=",",header=columns)
np.save("r.npy", r)
