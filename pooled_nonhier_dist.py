#!/usr/bin/env python
# coding: utf-8

# In[1]:


#This performs Hierarchical modeling on a partially-observed birth-death process 0->X->0 with birth parameter A and death parameter B.
import pandas as pd
import numpy as np
from scipy.stats import gamma
from scipy.stats import norm
from scipy.stats import poisson
from scipy.stats import skellam
import scipy.integrate as integrate
#from numba import jit
# In[2]:
pd_subsampled_x=pd.read_csv("subsampled.csv")
x= pd_subsampled_x.values
size=np.shape(x)

#pd_B=pd.read_csv("B.csv")
#B= pd_B.values

# In[3]:
L=10000      #length of the chain
#I=size[0]    #[0,I-1] is the observation window
#M=size[1]    #number of individuals
I=21
M=20
div=1       #time-scaling factor

#initializations
A=np.zeros(L)               #birth parameter
B=np.ones(L)*9*0.0016                #death parameter
alpha=np.zeros(L)           #alpha parameter
beta=np.zeros(L)            #beta parameter
v_int_gammainc=np.zeros(I-1)      #vector of integrals
r=np.zeros((2,I-1,M))     #reaction numbers  

var_alpha=0.1                    #birth shape hyperparameter proposal variance
tune=300   
tune_beta=7000                   #reaction proposal tuning parameter

#initial conditions
alpha[0]=7
beta[0]=1
a_A=0.001
b_A=0.001
a_alpha=0.001  
b_alpha=0.001
a_beta=0.001
b_beta=0.001

# In[4]:
def init_reaction(x,s):         #initialize reaction numbers           
    '''
    Initializes the number of completed reactions between all time intervals for an individual n 
    :param x: subsampled trajectory of molecular counts for individual n
    :return: q
    '''  
    q=np.zeros([2,s-1])
    for i in range(s-1):
        if x[i+1]-x[i]>0:
            q[0,i]=(x[i+1]-x[i])
            q[1,i]=0
        else:
            q[0,i]=0
            q[1,i]=-(x[i+1]-x[i])
    r=q.astype(int)        
    return r  

def accept_rate_react(x,r,prop,A,B,jump,delay_factor):        #acceptance rate for reaction numbers
    '''
    Computes the acceptance rate for the number of completed reactions at a given time interval for a specific individual n
    :param x: vector [x_i, x_i+1], the molecular counts at the time i and time i+1
    :param r: vector [r_birth, r_death] the current number of completed reactions of each type 
    :param prop: vector [prop_r_birth, prop_r_death] the proposed number of completed reactions of each type 
    :param A: latest sample for A_n 
    :param B: latest sample for B_n
    :param jump: sampled from the Skellam distribution that serves to augment the current reaction numbers
    :param delay_factor: average completion propensity on the time interval [i,i+1]
    :return: rate
    ''' 
    lambda_prop=1+(prop[0]**2)/tune
    lambda_cur=1+(r[0]**2)/tune
    prop_like=poisson.logpmf(prop[0],A*delay_factor)+poisson.logpmf(prop[1],0.5*(1./div)*B*(x[0]+x[1]))+skellam.logpmf(abs(jump),lambda_prop,lambda_prop)
    current_like=poisson.logpmf(r[0],A*delay_factor)+poisson.logpmf(r[1],0.5*(1./div)*B*(x[0]+x[1]))+skellam.logpmf(abs(jump),lambda_cur,lambda_cur)    
    q=np.exp(prop_like-current_like)
    if np.isnan(q)==False:
        rate=np.minimum(1,q)
    else:
        rate=0
    return rate

#@jit
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
    sum_log_gammainc=0
    kappa=int_gammainc(alpha,beta)
    for s in range(M):        
        log_kappa=np.dot(r[0,:,s],np.log(kappa+1e-300))
        sum_log_gammainc+=log_kappa 
    q=sum_log_gammainc-(A*M*np.sum(kappa))+gamma.logpdf(alpha,a=a,scale=1./b)+norm.logcdf(var/var_alpha,0,1)
    return q

#@jit
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
    sum_log_gammainc=0
    kappa=int_gammainc(alpha,beta)
    for s in range(M):        
        log_kappa=np.dot(r[0,:,s],np.log(kappa+1e-300))
        sum_log_gammainc+=log_kappa
    q=sum_log_gammainc-(A*M*np.sum(kappa))+gamma.logpdf(beta,a=a,scale=1./b)+gamma.logpdf(var,a=beta*tune_beta,scale=1./tune_beta)#+np.log(tr_normal(var,0,beta,var_beta)+1e-300)
    return q

def int_gammainc(alpha,beta):
    '''
    Computes the average completion propensty on each time interval [m,m+1] for an individual n
    :param alpha: current alpha_n sample for individual n 
    :param beta: current beta_n sample for individual n 
    :return: k
    '''
    q=np.zeros(I-1)
    for m in range(I-1):
        k,s=integrate.quad(lambda x: gamma.cdf(x,a=alpha,scale=beta**(-1)),m/div,(m+1)/div)
        q[m]=k
    return q

def MH_alpha(A,alpha,beta,r):
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

def MH_beta(A,alpha,beta,r):
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
    r_vprop=np.array([-1,-1])
    lamb=1+(r[0]**2)/tune                 #tuning parameter suggested in Boy's paper 
    while r_vprop[0]<0 or r_vprop[1]<0:
        jump=np.random.poisson(lamb)-np.random.poisson(lamb)
        r_prop=r[0]+jump
        r_vprop=[r_prop, r_prop-(x[j+1]-x[j])]    
    rate=accept_rate_react([x[j],x[j+1]],r,r_vprop,A,B,jump,kappa) 
    if np.random.uniform(0,1,1)<rate:
        k=r_vprop
    else:
        k=r
    return k

# In[5]:
#initialize number of reactions
for s in range(M):
    r[:,:,s]= init_reaction(x[:,s],I)

# In[6]:
#perform sampling        
for i in range(L-1):             
    #sample parameters
    A[i+1]=np.random.gamma(np.sum(r[0,:,:])+a_A,(M*np.sum(v_int_gammainc)+b_A)**(-1))
                  
    #update parameters alpha and beta
    alpha[i+1]=MH_alpha(A[i+1],alpha[i],beta[i],r)
    beta[i+1]=MH_beta(A[i+1],alpha[i+1],beta[i],r)
    
    #gamma incomplete function values
    v_int_gammainc = int_gammainc(alpha[i+1],beta[i+1]) 
    
    #update reaction numbers
    for s in range(M):    
        for j in range(I-1):
            r[:,j,s]=MH_r(A[i+1],B[s],r[:,j,s],v_int_gammainc[j],x[:,s],j)
  
    if i%100 == 0:  
        print(i+1,'birth',A[i+1],'alpha',alpha[i+1],'beta',beta[i+1]) 
        #print(alpha_prop,beta_prop)
        print("%%%%%%%%%%%%%%%%%next%%%%%%%%%%%%%%%%")
# In[ ]:

#save the simulated values into .csv files
columns=' '
np.savetxt("A.csv", A, delimiter=",",header=columns)
np.savetxt("alpha.csv", alpha, delimiter=",",header=columns)
np.savetxt("beta.csv", beta, delimiter=",",header=columns)
# In[ ]:




