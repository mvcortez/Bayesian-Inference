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
pd_subsampled_x=pd.read_csv("subsampled.csv")
x= pd_subsampled_x.values
size=np.shape(x)

pd_B=pd.read_csv("B.csv")
B= pd_B.values

# In[3]:
L=10000     #length of the chain
#I=size[0]    #[0,I-1] is the observation window
#M=size[1]    #number of individuals
I=41
M=40
div=1         #time-scaling factor div is the divisor of 1 minute, for e.g. for subsampled every 1/2 min, n=2

#initializations
A=np.zeros((L,M))               #birth parameter
#B=np.zeros((L,M))              #death parameter
alpha=np.zeros((L,M))           #alpha parameter
beta=np.zeros((L,M))            #beta parameter
a_A=np.zeros(L+1)               #birth shape hyperparameter
#a_B=np.zeros(L+1)              #death shape hyperparameter
a_alpha=np.zeros(L+1)           #alpha shape hyperparameter
a_beta=np.zeros(L+1)            #beta shape hyperparameter
b_A=np.zeros(L+1)               #birth rate hyperparameter
#b_B=np.zeros(L+1)              #death rate hyperparameter
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
a_alpha[0]=60
b_alpha[0]=6
a_beta[0]=7
b_beta[0]=7

#proposal distribution variances and parameters
var_alpha=1                     #alpha proposal variance
var_beta=0.1                     #alpha proposal variance
var_a_A=1                       #birth shape hyperparameter proposal variance
#var_a_B=1                      #death shape hyperparameter proposal variance
var_a_alphabeta=0.5                  #a_alpha and _beta hyperparameter proposal variance
var_b_alphabeta=0.5                  #b_alpha and _beta hyperparameter proposal variance
tune=300                        #reaction proposal tuning parameter
tune_beta=600

#parameters of the folded normal hyperpriors for alpha and beta
mu1_alpha=60                    #mean of the underlying Gaussian distribution for a_alpha
sig1_alpha=3                    #standard deviation of the underlying Gaussian distribution for a_alpha
mu2_alpha=6                     #mean of the underlying Gaussian distribution for the b_alpha
sig2_alpha=3                    #standard deviation of the underlying Gaussian distribution for b_alpha
p_alpha=0                       #correlation coefficient

mu1_beta=7                     #mean of the underlying Gaussian distribution for a_beta
sig1_beta=3                     #standard deviation of the underlying Gaussian distribution for a_beta
mu2_beta=7                     #mean of the underlying Gaussian distribution for the b_beta
sig2_beta=3                     #standard deviation of the underlying Gaussian distribution for b_beta
p_beta=0                        #correlation coefficient

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
    kappa=int_gammainc(alpha,beta)
    sum_log_gammainc = np.dot(r,np.log(kappa+1e-100))
    q=sum_log_gammainc-(A*np.sum(kappa))+gamma.logpdf(beta,a=a,scale=1./b)+gamma.logpdf(var,a=beta*tune_beta,scale=1./tune_beta)
    return q

def a_lik(A,a,b,prev):           #logposterior function for a_A or a_B
    '''
    Computes the log-posterior function evaluated at a_A conditioned at b_A and vector A
    :param A: Vector of A_n values at the current iteration for all individuals n 
    :param a: hyperparameter a_A 
    :param b: hyperparameter b_A
    :param prev: a_A[i]    
    :return: q
    ''' 
    q=-M*math.lgamma(a)+M*a*np.log(b)+(a-1)*np.sum(np.log(A))+norm.logcdf(prev/var_a_A,0,1)
    return q


def a_delay_lik(var,a,b,typ,prev):    #logposterior function for a_alpha and a_beta
    '''
    Computes the log-posterior function evaluated at a conditioned at b for either alpha or beta
    :param var: Vector of alpha_n (or beta_n) values at the current iteration for all individuals n 
    :param a: hyperparameter a for either alpha or beta
    :param b: hyperparameter b for either alpha or beta
    :param typ: indicator of whether the log-posterior is for alpha or beta
    :return: q
    '''
    if typ=='alpha':
        mu1=mu1_alpha
        mu2=mu2_alpha
        sig1=sig1_alpha
        sig2=sig2_alpha
        p=p_alpha
    else:
        mu1=mu1_beta
        mu2=mu2_beta
        sig1=sig1_beta
        sig2=sig2_beta
        p=p_beta
    q=-M*np.log(math.gamma(a))+a*M*np.log(b)+(a-1)*np.sum(np.log(var))+np.log(folded_normal(mu1,sig1,mu2,sig2,p,a,b)+1e-300)+norm.logcdf(prev/var_a_alphabeta,0,1)
    return q

def b_delay_lik(var,a,b,typ,prev):    #logposterior function for b_alpha and b_beta
    '''
    Computes the log-posterior function evaluated at b conditioned at a for either alpha or beta
    :param var: Vector of alpha_n (or beta_n) values at the current iteration for all individuals n 
    :param a: hyperparameter a for either alpha or beta
    :param b: hyperparameter b for either alpha or beta
    :param typ: indicator of whether the log-posterior is for alpha or beta
    :return: q
    '''
    if typ=='alpha':
        mu1=mu1_alpha
        mu2=mu2_alpha
        sig1=sig1_alpha
        sig2=sig2_alpha
        p=p_alpha
    else:
        mu1=mu1_beta
        mu2=mu2_beta
        sig1=sig1_beta
        sig2=sig2_beta
        p=p_beta  
    q=a*M*np.log(b)-b*np.sum(var)+np.log(folded_normal(mu1,sig1,mu2,sig2,p,a,b)+1e-300)+norm.logcdf(prev/var_a_alphabeta,0,1)
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

def folded_normal(mu1,sig1,mu2,sig2,p,x,y):
    '''
    Evaluates the bivariate folded normal pdf at x and y
    :param mu1: mean of the underlying Gaussian distribution for the random variable X
    :param sig1: standard deviation of the underlying Gaussian distribution for the random variable X
    :param mu2: mean of the underlying Gaussian distribution for the random variable Y
    :param sig2: standard deviation of the underlying Gaussian distribution for the random variable Y
    :param p: correlation coefficient
    :param x: first variable of the bivariate folded normal pdf
    :param y: secord variable of the bivariate folded normal pdf
    :return: function evaluation
    '''
    k1=np.exp(-(1/(2*(1-p**2)))*(((x-mu1)**2/sig1**2)+(-2*p*(((x-mu1)*(y-mu2))/(sig1*sig2)))+((y-mu2)**2/sig2**2)))
    k2=np.exp(-(1/(2*(1-p**2)))*(((x+mu1)**2/sig1**2)+(-2*p*(((x+mu1)*(y+mu2))/(sig1*sig2)))+((y+mu2)**2/sig2**2)))
    k3=np.exp(-(1/(2*(1-p**2)))*(((x+mu1)**2/sig1**2)+(2*p*(((x+mu1)*(y-mu2))/(sig1*sig2)))+((y-mu2)**2/sig2**2)))
    k4=np.exp(-(1/(2*(1-p**2)))*(((x-mu1)**2/sig1**2)+(2*p*(((x-mu1)*(y+mu2))/(sig1*sig2)))+((y+mu2)**2/sig2**2)))    
    return (1/(2*np.pi*sig1*sig2*np.sqrt(1-p**2)))*(k1+k2+k3+k4)

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
    '''
    Performs Metropolis-Hastings sampling for the alpha delay parameter
    :param A: current A_n sample for individual n
    :param alpha: current alpha_n sample for individual n
    :param beta: current beta_n sample for individual n
    :param a_alpha: current a_alpha sample
    :param b_alpha: current b_alpha sample
    :param r: matrix containing the number of births and deaths in every subinterval for individual n
    :return: k
    '''
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
     '''
    Performs Metropolis-Hastings sampling for the beta delay parameter
    :praram A: current A_n sample for individual n
    :param alpha: current alpha_n sample for individual n
    :param beta: current beta_n sample for individual n
    :param a_alpha: current a_alpha sample
    :param b_alpha: current b_alpha sample
    :param r: matrix containing the number of births and deaths in every subinterval for individual n
    :return: k
    '''
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
    '''
    Performs Metropolis-Hastings sampling for number of birth reactions completed in every interval for invidual n
    :param A: current A_n sample for individual n
    :param B: current B_n sample for individual n
    :param r: matrix containing the number of births and deaths in every subinterval for individual n
    :param kappa: average completion propensity on the time interval [j,j+1]
    :param x: vector [x_i, x_i+1], the molecular counts at the time j and time j+1 
    :param j: time index
    :return: k
    '''
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

def MH_a_A(A,a_A,b_A):
    '''
    Performs Metropolis-Hastings sampling for the hyperparameter a_A
    :param A: collection of current A_n sample for all individuals
    :param a_alpha: current a_alpha sample
    :param b_alpha: current b_alpha sample
    :return: k
    '''
    a_Aprop=-1
    while a_Aprop<0:
        a_Aprop=a_A+np.random.normal(0,var_a_A)
    a=a_lik(A,a_Aprop,b_A,a_A) 
    b=a_lik(A,a_A,b_A,a_Aprop)
    rate=np.minimum(1,np.exp(a-b))
    if np.random.uniform(0,1,1)<rate:    
        k=a_Aprop
    else:
        k=a_A
    return k

def MH_a_delay(param,a_param,b_param,typ):
    '''
    Performs Metropolis-Hastings sampling for the hyperparameter a_alpha or a_beta
    :param param: collection of either the current alpha_n or beta_n sample for all individuals
    :param a_param: current a_alpha sample
    :param b_alpha: current b_alpha sample
    :return: k
    '''
    a_prop=-1
    while a_prop<0:
        a_prop=a_param+np.random.normal(0,var_a_alphabeta)
    a=a_delay_lik(param,a_prop,b_param,typ,a_param) 
    b=a_delay_lik(param,a_param,b_param,typ,a_prop)
    rate=np.minimum(1,np.exp(a-b))
    if np.random.uniform(0,1,1)<rate:    
        k=a_prop
    else:
        k=a_param
    return k

def MH_b_delay(param,a_param,b_param,typ):    
    '''
    Performs Metropolis-Hastings sampling for either the hyperparameter b_alpha or b_beta
    :param param: collection of either the current alpha_n or beta_n sample for all individuals
    :param a_param: current a_alpha or a_beta sample
    :param b_param: current b_alpha or b_beta sample
    :return: k
    '''
    b_prop=-1
    while b_prop<0:
        b_prop=b_param+np.random.normal(0,var_b_alphabeta)
    a=b_delay_lik(param,a_param,b_prop,typ,b_param) 
    b=b_delay_lik(param,a_param,b_param,typ,b_prop)
    rate=np.minimum(1,np.exp(a-b))
    if np.random.uniform(0,1,1)<rate:    
        k=b_prop
    else:
        k=b_param
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
    a_A[i+1]=MH_a_A(A[i+1,:],a_A[i],b_A[i])
    a_alpha[i+1]=MH_a_delay(alpha[i+1,:],a_alpha[i],b_alpha[i],'alpha')
    a_beta[i+1]=MH_a_delay(beta[i+1,:],a_beta[i],b_beta[i],'beta')
    
    #sample hyperparameters b_A
    sum_A=np.sum(A[i+1,:])
    b_A[i+1]=np.random.gamma(M*a_A[i+1],sum_A**(-1))
    
    #sample hyperparameter b_alpha and b_beta
    b_alpha[i+1]=MH_b_delay(alpha[i+1,:],a_alpha[i+1],b_alpha[i],'alpha')
    b_beta[i+1]=MH_b_delay(beta[i+1,:],a_beta[i+1],b_beta[i],'beta')
        
    if i%100 == 0:  
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
