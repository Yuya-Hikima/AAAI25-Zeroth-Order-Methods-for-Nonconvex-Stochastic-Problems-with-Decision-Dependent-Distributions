from scipy.stats import bernoulli
import numpy as np
import time
import GPyOpt
import math
import csv
from scipy import stats
import statistics
import sys
import pickle
import os
import matplotlib.pyplot as plt


def index_realization_vectorized(cumulate_list, num_rands):
    #This function returns the indices where random values fall within the cumulative distribution.
    return np.searchsorted(cumulate_list, num_rands)
    
def cost_cal(xi):
    # This function calculates the total cost based on the input demand vector xi.
    cost=np.zeros(n)
    for i in range(n):
        if xi[i]<L_v[i]:
            cost[i]=a_v[i]*xi[i]
        elif xi[i]<U_v[i]:
            cost[i]=b_v[i]*(xi[i]-L_v[i])+a_v[i]*L_v[i]
        else:
            cost[i]=c_v[i]*(xi[i]-U_v[i])+b_v[i]*(U_v[i]-L_v[i])+a_v[i]*L_v[i]
    return np.sum(cost)

def one_realized_f(x):
    #This function computes the realized profit in a single trial for a given price vector x.
    price_v=x.reshape(n,1)
    exp_value=np.exp(gamma_v*(alpha_v-price_v))
    sum_exp_value=a_0+np.sum(exp_value)
    purchase_prob=exp_value/sum_exp_value
    cumulate_list = np.cumsum(purchase_prob)
    num_rands = np.random.rand(m)
    indices = index_realization_vectorized(cumulate_list, num_rands)
    sum_sold = np.bincount(indices, minlength=n+1)
    cost=np.zeros(n)
    for i in range(n):
        if sum_sold[i]<L_v[i]:
            cost[i]=a_v[i]*sum_sold[i]
        elif sum_sold[i]<U_v[i]:
            cost[i]=b_v[i]*(sum_sold[i]-L_v[i])+a_v[i]*L_v[i]
        else:
            cost[i]=c_v[i]*(sum_sold[i]-U_v[i])+b_v[i]*(U_v[i]-L_v[i])+a_v[i]*L_v[i]
    return -np.sum(price_v*sum_sold[:-1].reshape(n,1))+np.sum(cost)

def expected_f(x):
    #This function calculates the average realized profit in ``metric_iter'' trials for a given price vector x.
    #That is,  it is the function that computes the metric in Section 6.2 of our paper.
    price_v=x.reshape(n,1)
    exp_value=np.exp(gamma_v*(alpha_v-price_v))
    sum_exp_value=a_0+np.sum(exp_value)
    purchase_prob=exp_value/sum_exp_value

    revenue_v=[]
    for k in range(metric_iter):
        cumulate_list = np.cumsum(purchase_prob)
        num_rands = np.random.rand(m)
        indices = index_realization_vectorized(cumulate_list, num_rands)
        sum_sold = np.bincount(indices, minlength=n+1)
        
        cost=np.zeros(n)
        for i in range(n):
            if sum_sold[i]<L_v[i]:
                cost[i]=a_v[i]*sum_sold[i]
            elif sum_sold[i]<U_v[i]:
                cost[i]=b_v[i]*(sum_sold[i]-L_v[i])+a_v[i]*L_v[i]
            else:
                cost[i]=c_v[i]*(sum_sold[i]-U_v[i])+b_v[i]*(U_v[i]-L_v[i])+a_v[i]*L_v[i]
            #debug
            if cost[i]<0:
                error
        revenue_v.append(np.sum(price_v*sum_sold[:-1].reshape(n,1))-np.sum(cost))
    return -np.average(revenue_v)

def expected_f_sample(x,random_num):
    #This function calculates the average realized profit in ``random_num'' trials for a given price vector x.
    price_v=x.reshape(n,1)
    exp_value=np.exp(gamma_v*(alpha_v-price_v))
    sum_exp_value=a_0+np.sum(exp_value)
    purchase_prob=exp_value/sum_exp_value

    revenue_v=[]
    sample_list=[]
    for k in range(random_num):
        cumulate_list = np.cumsum(purchase_prob)
        num_rands = np.random.rand(m)
        indices = index_realization_vectorized(cumulate_list, num_rands)
        sum_sold = np.bincount(indices, minlength=n+1)
        
        cost=np.zeros(n)
        for i in range(n):
            if sum_sold[i]<L_v[i]:
                cost[i]=a_v[i]*sum_sold[i]
            elif sum_sold[i]<U_v[i]:
                cost[i]=b_v[i]*(sum_sold[i]-L_v[i])+a_v[i]*L_v[i]
            else:
                cost[i]=c_v[i]*(sum_sold[i]-U_v[i])+b_v[i]*(U_v[i]-L_v[i])+a_v[i]*L_v[i]
        revenue_v.append(np.sum(price_v*sum_sold[:-1].reshape(n,1))-np.sum(cost))
        sample_list.append(sum_sold)
    return -np.average(revenue_v),sample_list

def f_given(x,sum_sold):
    #This function calculates the realized profit given a price vector and a demand vector.
    price_v=x.reshape(n,1)
    cost=np.zeros(n)
    for i in range(n):
        if sum_sold[i]<L_v[i]:
            cost[i]=a_v[i]*sum_sold[i]
        elif sum_sold[i]<U_v[i]:
            cost[i]=b_v[i]*(sum_sold[i]-L_v[i])+a_v[i]*L_v[i]
        else:
            cost[i]=c_v[i]*(sum_sold[i]-U_v[i])+b_v[i]*(U_v[i]-L_v[i])+a_v[i]*L_v[i]
        #debug
        if cost[i]<0:
            error
    return -np.sum(price_v*sum_sold[:-1].reshape(n,1))+np.sum(cost)

def save_plots(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    plot_data = {
        'm_k_sum_v': m_k_sum_v,
        'f_v': f_v,
        'm_k_sum_v_m1': m_k_sum_v_m1,
        'f_v_batch_1': f_v_batch_1,
        'm_k_sum_2_v': m_k_sum_2_v,
        'P_2_f_v_m': P_2_f_v_m,
        'm_k_sum_v_two': m_k_sum_v_two,
        'P_2_f_v_batch_1': P_2_f_v_batch_1,
        'CZO_m_k_sum_v': CZO_m_k_sum_v,
        'CZO_f_v': CZO_f_v,
        'CZO_m_k_sum_v_m1': CZO_m_k_sum_v_m1,
        'CZO_f_v_batch_1': CZO_f_v_batch_1
    }
    pickle_file_name = f'plot_data_{date_ID}.pkl'
    pickle_file_path = os.path.join(folder_name, pickle_file_name)
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(plot_data, f)

    fig, ax = plt.subplots()
    c1, c2, c3, c4, c5, c6 = "blue", "cyan", "green", "red", "black", "orange"  
    l1, l2, l3, l4, l5, l6 = "Proposed-1 (mini-batch)", "Proposed-1 ($m_k=1$)", "Proposed-2 (mini-batch)", "Proposed-2 ($m_k=1$)", "CZO-1 (mini-batch)", "CZO-1 ($m_k=1$)" 
    ax.set_xlabel('sample number') 
    ax.set_ylabel('obj')  
    ax.grid()  
    ax.plot(m_k_sum_v, f_v, color=c1, label=l1)
    ax.plot(m_k_sum_v_m1, f_v_batch_1, color=c2, label=l2)
    ax.plot(m_k_sum_2_v, P_2_f_v_m, color=c3, label=l3)
    ax.plot(m_k_sum_v_two, P_2_f_v_batch_1, color=c4, label=l4)
    ax.plot(CZO_m_k_sum_v, CZO_f_v, color=c5, label=l5)
    ax.plot(CZO_m_k_sum_v_m1, CZO_f_v_batch_1, color=c6, label=l6)
    ax.legend(loc=0)
    fig.tight_layout() 
    plt.savefig(os.path.join(folder_name, 'plot1.png')) 

#the setting of the random seed
np.random.seed(2024)
#This sets the seed for NumPy's random number generator to ensure reproducibility.
#By setting the seed, the sequence of random numbers generated by NumPy will be the same every time the code is run.

#the setting of the proposed method
mu_0_proposed=float(sys.argv[1])
mu_min_proposed=float(sys.argv[2])
beta_0_proposed=float(sys.argv[4])
coef_m_k_to_iteration=int(sys.argv[5])
L_xi_alpha_sq_devided_sigma_sq=float(sys.argv[6])
s_max=int(sys.argv[7])
gamma_mu=0.95
gamma_beta=0.95
m_k_initial=30 
initial_samples=20 # samples to calculate c_0

#the setting of the CZO method
mu_c_0=float(sys.argv[3])
beta_0_one_point=0.00001

#Number of samples to calculate the metric defined in Section 6.2 od our paper.
metric_iter=1000

#Maximum number of samples for each method, which is used as the termination condition.
max_m_k_sum=int(sys.argv[8])

#used data
date_ID=int(sys.argv[9])
date_ID=f'{date_ID:02}'

#Problem setting
usedata='data/2022_%s.csv' %date_ID
co_a_0=0.1
n=int(sys.argv[10])
m=int(sys.argv[11])

#the initial point for each method
initial_point=float(sys.argv[12])
x_0_proposed=initial_point*np.ones([n,1]).reshape(n,1)

#the number of simulations
num_sim=int(sys.argv[13])

#lists to store the results of all method
proposed_1_f_result=[]
proposed_1_time_result=[]

proposed_1_f_batch1_result=[]
proposed_1_time_batch1_result=[]

CZO_f_result=[]
CZO_time_result=[]

CZO_f_batch1_result=[]
CZO_time_batch1_result=[]

proposed_2_f_result=[]
proposed_2_time_result=[]

proposed_2_f_batch1_result=[]
proposed_2_time_batch1_result=[]


#For each problem instance, perform the experiment and save the results.
for problem_instance in range(num_sim):
    #Read the actual price data.
    f = open(usedata, encoding="utf-8_sig")
    areas = f.read().split()
    f.close()
    alpha_v=np.array([int(s) for s in areas])[0:n].reshape(n,1)
    
    #Normalize the price data.
    alpha_v=alpha_v/np.max(alpha_v)

    #Set the parameters of the problem.
    gamma_v=math.pi/(0.5*np.sqrt(6)*alpha_v)
    w_v=alpha_v*(0.25+0.25*np.random.rand(n,1))
    a_v=2.0*w_v
    b_v=w_v
    c_v=3.0*w_v
    a_0=co_a_0*n
    L_v=m/n*0.5*np.ones([n,1])
    U_v=m/n*1.5*np.ones([n,1])

    #Proposed-1 method (minibatch)
    #variables for measuring time
    start_time=time.time()
    overhead_time=0

    #Preparation of inputs
    ##x_0
    x_k=x_0_proposed
    ##c_0
    random_num=initial_samples
    c_value,pre_sample_list=expected_f_sample(x_k,random_num)
    ##\mu_0
    mu_k=mu_0_proposed
    ##\beta
    beta=beta_0_proposed

    #store the elapsed time, the objective value, and the total number of samples
    ##the elapsed time (Time to compute the objective value is excluded.)
    time_v=[time.time()-start_time]
    ##the objective value
    start_time2=time.time()
    f_end=expected_f(x_k)
    overhead_time+=start_time2-time.time()
    f_v=[f_end]
    ##the total number of samples
    m_k_sum_v=[random_num]

    #List stores x_k + \mu_k u_k 
    x_mu_u_k_list = [x_k]
    #List stores m_k
    m_k_list = [random_num]
    #List stores xi_k^j
    xi_list_of_list=[pre_sample_list]

    iter=0
    m_k_sum=random_num
    while 1:

        #set m_k
        m_k=m_k_initial+iter*coef_m_k_to_iteration
        m_k_sum+=m_k

        #set beta
        beta=beta*gamma_beta
        
        #Steps 2 and 3: calculate g_k
        #sample u_k
        u_k=np.random.normal(size=n).reshape(n,1)

        x_mu_u_k=x_k+mu_k*u_k
        x_mu_u_k_list.append(x_mu_u_k)
        
        #sample \xi_k^j and calculate g_k 
        f_value_sum=0
        xi_list=[]
        for k in range(m_k):
            exp_value=np.exp(gamma_v*(alpha_v-x_mu_u_k))
            sum_exp_value=a_0+np.sum(exp_value)
            px=exp_value/sum_exp_value
            px=np.append(px,a_0/sum_exp_value).reshape(n+1,1)
            cumulate_list = np.cumsum(px)
            num_rands = np.random.rand(m)
            indices = index_realization_vectorized(cumulate_list, num_rands)
            sum_sold = np.bincount(indices, minlength=n+1)
            xi_list.append(sum_sold)
            f_value_sum+=f_given(x_mu_u_k,sum_sold)
        xi_list_of_list.append(xi_list)
        f_value_mean=f_value_sum/m_k
        g_k=(f_value_mean-c_value)/mu_k*u_k

        #Step 4: Updates the iterate
        x_k=x_k-beta*g_k

        # Step 5: Compute mu_k
        mu_k=np.max([mu_k*gamma_mu,mu_min_proposed])

        iter+=1
        # Step 6: Compute s
        s = min(s_max, iter)

        # Step 7: Compute a_i
        b_i=np.zeros(s)
        for i in range(s):
            b_i[s-1-i] = L_xi_alpha_sq_devided_sigma_sq*np.linalg.norm(x_k - x_mu_u_k_list[iter-1-i])**2 + 1/m_k_list[iter-1-i]
        b_inv_sum=0
        for i in range(s):
            b_inv_sum+=1/b_i[i]
        a_i=np.zeros(s)
        for i in range(s):
            a_i[i] = 1/(b_i[i]*b_inv_sum)
        
        # Step 8: Compute c_k+1
        c_value=0
        for i in range(s):
            tmp=0
            for xi in xi_list_of_list[iter-1-i]:
                tmp += f_given(x_k,xi)
            c_value+=a_i[s-1-i]*tmp/len(xi_list_of_list[iter-1-i])

        #Store the elapsed time, the objective value, and the total number of samples
        start_time2=time.time()
        f_end=expected_f(x_k)
        overhead_time+=start_time2-time.time()
        time_end=time.time()-start_time-overhead_time
        f_v.append(f_end)
        time_v.append(time_end)
        m_k_list.append(m_k)
        m_k_sum_v.append(m_k_sum)

        #termination condition
        if m_k_sum>max_m_k_sum:
            break



    #the CZO (minibatch) method

    #variables for measuring time
    start_time=time.time()
    overhead_time=0

    #Preparation of inputs
    ##x_0
    x_k=x_0_proposed
    ##\mu_0
    mu_k=mu_c_0
    ##beta
    beta=beta_0_one_point

    #store the elapsed time, the objective value, and the total number of samples
    ##the elapsed time (Time to compute the objective value is excluded.)
    CZO_time_v=[0]
    start_time2=time.time()
    tmp=expected_f(x_k)
    overhead_time+=start_time2-time.time()
    ##objective value
    CZO_f_v=[tmp]
    ##the total number of samples
    CZO_m_k_sum_v=[0]

    iter=0
    m_k_sum=0
    while 1:
        #Set m_k
        m_k=m_k_initial+iter*coef_m_k_to_iteration
        m_k_sum+=m_k

        #Set beta       
        beta=beta*gamma_beta
        
        #Calculate a gradient
        u_k=np.random.normal(size=n).reshape(n,1)
        x_mu_u_k=x_k+mu_k*u_k
        f_value_sum=0
        for k in range(m_k):
            exp_value=np.exp(gamma_v*(alpha_v-x_mu_u_k))
            sum_exp_value=a_0+np.sum(exp_value)
            px=exp_value/sum_exp_value
            px=np.append(px,a_0/sum_exp_value).reshape(n+1,1)
            cumulate_list = np.cumsum(px)
            num_rands = np.random.rand(m)
            indices = index_realization_vectorized(cumulate_list, num_rands)
            sum_sold = np.bincount(indices, minlength=n+1)
            f_value_sum+=f_given(x_mu_u_k,sum_sold)
        f_value_mean=f_value_sum/m_k
        g_k=f_value_mean/mu_k*u_k

        #Updates the iterate
        x_k=x_k-beta*g_k

        #Store the elapsed time, the objective value, and the total number of samples
        #the elapsed time (Time to compute the objective value is excluded.)
        start_time2=time.time()
        f_end=expected_f(x_k)
        overhead_time+=start_time2-time.time()
        time_end=time.time()-start_time-overhead_time
        CZO_f_v.append(f_end)
        CZO_time_v.append(time_end)
        CZO_m_k_sum_v.append(m_k_sum)

        #termination condition
        if m_k_sum>max_m_k_sum:
            break

        iter+=1



    #the CZO (batch size 1) method

    #variables for measuring time
    start_time=time.time()
    overhead_time=0

    #Preparation of inputs
    ##x_0
    x_k=x_0_proposed
    ##\mu_0
    mu_k=mu_c_0
    ##beta
    beta=beta_0_one_point
    ##m_k
    m_k=1

    #store the elapsed time, the objective value, and the total number of samples
    ##the elapsed time (Time to compute the objective value is excluded.)
    CZO_time_v_batch_1=[0]
    start_time2=time.time()
    tmp=expected_f(x_k)
    overhead_time+=start_time2-time.time()
    ##objective value
    CZO_f_v_batch_1=[tmp]
    ##the total number of samples
    CZO_m_k_sum_v_m1=[0]

    iter=0
    m_k_sum=0
    while 1:
        #Calculate m_k_sum
        m_k_sum+=m_k
        
        #Set beta
        beta=beta*gamma_beta
        
        #Calculate a gradient
        u_k=np.random.normal(size=n).reshape(n,1)
        x_mu_u_k=x_k+mu_k*u_k
        x_mu_u_k_list.append(x_mu_u_k)
        f_value_sum=0
        for k in range(m_k):
            exp_value=np.exp(gamma_v*(alpha_v-x_mu_u_k))
            sum_exp_value=a_0+np.sum(exp_value)
            px=exp_value/sum_exp_value
            px=np.append(px,a_0/sum_exp_value).reshape(n+1,1)
            cumulate_list = np.cumsum(px)
            num_rands = np.random.rand(m)
            indices = index_realization_vectorized(cumulate_list, num_rands)
            sum_sold = np.bincount(indices, minlength=n+1)
            f_value_sum+=f_given(x_mu_u_k,sum_sold)        
        f_value_mean=f_value_sum/m_k
        g_k=f_value_mean/mu_k*u_k

        #Updates the iterate
        x_k=x_k-beta*g_k

        #Store the elapsed time, the objective value, and the total number of samples
        #the elapsed time (Time to compute the objective value is excluded.)
        start_time2=time.time()
        f_end=expected_f(x_k)
        overhead_time+=start_time2-time.time()
        time_end=time.time()-start_time-overhead_time
        CZO_f_v_batch_1.append(f_end)
        CZO_time_v_batch_1.append(time_end)
        CZO_m_k_sum_v_m1.append(m_k_sum)
                
        #termination condition
        if m_k_sum>max_m_k_sum:
            break
        
        iter+=1



    #Proposed-1 method (batch size 1)
    #variables for measuring time
    start_time=time.time()
    overhead_time=0

    #Preparation of inputs for the proposed method
    ##x_0
    x_k=x_0_proposed
    ##c_0
    random_num=initial_samples
    c_value,pre_sample_list=expected_f_sample(x_k,random_num)
    ##\mu_0
    mu_k=1
    ##\beta
    beta=beta_0_proposed

    #store the elapsed time, the objective value, and the total number of samples
    ##the elapsed time (Time to compute the objective value is excluded.)
    time_v_batch_1=[time.time()-start_time]
    ##the objective value
    start_time2=time.time()
    f_end=expected_f(x_k)
    overhead_time+=start_time2-time.time()
    f_v_batch_1=[f_end]
    ##the total number of samples
    m_k_sum_v_m1=[random_num]

    #List stores x_k + \mu_k u_k 
    x_mu_u_k_list = [x_k]
    #List stores m_k
    m_k_list = [random_num]
    #List stores xi_k^j
    xi_list_of_list=[pre_sample_list]

    iter=0
    m_k_sum=random_num
    while 1:
        #calculate m_k_sum
        m_k_sum+=m_k

        #set beta
        beta=beta*gamma_beta

        #Steps 2 and 3: calculate g_k
        #sample u_k
        u_k=np.random.normal(size=n).reshape(n,1)
        x_mu_u_k=x_k+mu_k*u_k
        x_mu_u_k_list.append(x_mu_u_k)

        #sample \xi_k and calculate g_k 
        f_value_sum=0
        xi_list=[]
        for k in range(m_k):
            exp_value=np.exp(gamma_v*(alpha_v-x_mu_u_k))
            sum_exp_value=a_0+np.sum(exp_value)
            px=exp_value/sum_exp_value
            px=np.append(px,a_0/sum_exp_value).reshape(n+1,1)
            cumulate_list = np.cumsum(px)
            num_rands = np.random.rand(m)
            indices = index_realization_vectorized(cumulate_list, num_rands)
            sum_sold = np.bincount(indices, minlength=n+1)
            xi_list.append(sum_sold)
            f_value_sum+=f_given(x_mu_u_k,sum_sold)
        xi_list_of_list.append(xi_list)
        f_value_mean=f_value_sum/m_k
        g_k=(f_value_mean-c_value)/mu_k*u_k

        #Step 4: Updates the iterate
        x_k=x_k-beta*g_k

        # Step 5: Compute mu_k
        mu_k=np.max([mu_k*gamma_mu,mu_min_proposed])

        iter+=1
        # Step 6: Compute s
        s = min(s_max, iter)

        # Step 7: Compute a_i
        b_i=np.zeros(s)
        for i in range(s):
            b_i[s-1-i] = L_xi_alpha_sq_devided_sigma_sq*np.linalg.norm(x_k - x_mu_u_k_list[iter-1-i])**2 + 1/m_k_list[iter-1-i]

        b_inv_sum=0
        for i in range(s):
            b_inv_sum+=1/b_i[i]

        a_i=np.zeros(s)
        for i in range(s):
            a_i[i] = 1/(b_i[i]*b_inv_sum)
        
        # Step 8: Compute c_k+1
        c_value=0
        for i in range(s):
            tmp=0
            for xi in xi_list_of_list[iter-1-i]:
                tmp += f_given(x_k,xi)
            c_value+=a_i[s-1-i]*tmp/len(xi_list_of_list[iter-1-i])
        
        #Store the elapsed time, the objective value, and the total number of samples
        #the elapsed time (Time to compute the objective value is excluded.)
        start_time2=time.time()
        f_end=expected_f(x_k)
        overhead_time+=start_time2-time.time()
        time_end=time.time()-start_time-overhead_time
        f_v_batch_1.append(f_end)
        time_v_batch_1.append(time_end)
        m_k_list.append(m_k)
        m_k_sum_v_m1.append(m_k_sum)

        #termination condition
        if m_k_sum>max_m_k_sum:
            break



    #Proposed-2 method (minibatch)
    start_time=time.time()
    overhead_time=0

    #Preparation of inputs
    ##x_0
    x_k=x_0_proposed
    ##\mu_0
    mu_k=mu_0_proposed
    ##\beta
    beta=beta_0_proposed

    #store the elapsed time, the objective value, and the total number of samples
    ##the elapsed time (Time to compute the objective value is excluded.)
    time_2_v_two_m=[time.time()-start_time]
    ##the objective value
    start_time2=time.time()
    f_end=expected_f(x_k)
    overhead_time+=start_time2-time.time()
    P_2_f_v_m=[f_end]
    ##the total number of samples
    m_k_sum_2_v=[0]

    iter=0
    m_k_sum=0
    while 1:
        
        #set m_k
        m_k=m_k_initial+iter*coef_m_k_to_iteration
        m_k_sum+=m_k

        #set beta      
        beta=beta*gamma_beta

        #Steps 2 and 3: calculate g_k
        #sample u_k
        u_k=np.random.normal(size=n).reshape(n,1)
        #sample \xi_k^{1,j}, \xi_k^{2,j}, and calculate g_k 
        x_mu_u_k=x_k+mu_k*u_k
        f_value_sum_1=0
        for k in range(m_k):
            exp_value=np.exp(gamma_v*(alpha_v-x_mu_u_k))
            sum_exp_value=a_0+np.sum(exp_value)
            px=exp_value/sum_exp_value
            px=np.append(px,a_0/sum_exp_value).reshape(n+1,1)
            cumulate_list = np.cumsum(px)
            num_rands = np.random.rand(m)
            indices = index_realization_vectorized(cumulate_list, num_rands)
            sum_sold = np.bincount(indices, minlength=n+1)
            f_value_sum_1+=f_given(x_mu_u_k,sum_sold)

        x_mu_m_u_k=x_k-mu_k*u_k
        f_value_sum_2=0
        for k in range(m_k):
            exp_value=np.exp(gamma_v*(alpha_v-x_mu_m_u_k))
            sum_exp_value=a_0+np.sum(exp_value)
            px=exp_value/sum_exp_value
            px=np.append(px,a_0/sum_exp_value).reshape(n+1,1)
            cumulate_list = np.cumsum(px)
            num_rands = np.random.rand(m)
            indices = index_realization_vectorized(cumulate_list, num_rands)
            sum_sold = np.bincount(indices, minlength=n+1)
            f_value_sum_2+=f_given(x_mu_m_u_k,sum_sold)
        
        f_value_1_mean=f_value_sum_1/m_k
        f_value_2_mean=f_value_sum_2/m_k
        g_k=(f_value_1_mean-f_value_2_mean)/(2*mu_k)*u_k

        #Step 4: Updates the iterate
        x_k=x_k-beta*g_k

        # Step 5: Compute mu_k
        mu_k=np.max([mu_k*gamma_mu,mu_min_proposed])

        iter+=1

        #Store the elapsed time, the objective value, and the total number of samples
        start_time2=time.time()
        f_end=expected_f(x_k)
        overhead_time+=start_time2-time.time()
        time_end=time.time()-start_time-overhead_time
        P_2_f_v_m.append(f_end)
        time_2_v_two_m.append(time_end)
        m_k_sum_2_v.append(m_k_sum)

        #termination condition
        if m_k_sum>max_m_k_sum:
            break



    #Proposed-2 method (batch size 1)
    #variables for measuring time
    start_time=time.time()
    overhead_time=0

    #Preparation of inputs for the proposed method
    ##x_0
    x_k=x_0_proposed
    ##\mu_0
    mu_k=mu_0_proposed
    ##\beta
    beta=beta_0_proposed

    #store the elapsed time, the objective value, and the total number of samples
    ##the elapsed time (Time to compute the objective value is excluded.)
    P_2_time_v_batch_1=[time.time()-start_time]
    ##the objective value
    start_time2=time.time()
    f_end=expected_f(x_k)
    overhead_time+=start_time2-time.time()
    P_2_f_v_batch_1=[f_end]
    ##the total number of samples
    m_k_sum_v_two=[0]

    iter=0
    m_k_sum=0
    while 1:
        
        #calculate m_k_sum
        m_k_sum+=m_k

        #set beta          
        beta=beta*gamma_beta
        
        #Steps 2 and 3: calculate g_k
        #sample u_k
        u_k=np.random.normal(size=n).reshape(n,1)

        #sample \xi_k^1, \xi_k^2, and calculate g_k 
        x_mu_u_k=x_k+mu_k*u_k
        f_value_sum_1=0
        for k in range(m_k):
            exp_value=np.exp(gamma_v*(alpha_v-x_mu_u_k))
            sum_exp_value=a_0+np.sum(exp_value)
            px=exp_value/sum_exp_value
            px=np.append(px,a_0/sum_exp_value).reshape(n+1,1)
            cumulate_list = np.cumsum(px)
            num_rands = np.random.rand(m)
            indices = index_realization_vectorized(cumulate_list, num_rands)
            sum_sold = np.bincount(indices, minlength=n+1)
            f_value_sum_1+=f_given(x_mu_u_k,sum_sold)

        x_mu_m_u_k=x_k-mu_k*u_k
        f_value_sum_2=0
        for k in range(m_k):
            exp_value=np.exp(gamma_v*(alpha_v-x_mu_m_u_k))
            sum_exp_value=a_0+np.sum(exp_value)
            px=exp_value/sum_exp_value
            px=np.append(px,a_0/sum_exp_value).reshape(n+1,1)
            cumulate_list = np.cumsum(px)
            num_rands = np.random.rand(m)
            indices = index_realization_vectorized(cumulate_list, num_rands)
            sum_sold = np.bincount(indices, minlength=n+1)
            f_value_sum_2+=f_given(x_mu_m_u_k,sum_sold)
        
        f_value_1_mean=f_value_sum_1/m_k
        f_value_2_mean=f_value_sum_2/m_k
        g_k=(f_value_1_mean-f_value_2_mean)/(2*mu_k)*u_k

        #Step 4: Updates the iterate
        x_k=x_k-beta*g_k

        # Step 5: Compute mu_k
        mu_k=np.max([mu_k*gamma_mu,mu_min_proposed])

        iter+=1

        #Store the elapsed time, the objective value, and the total number of samples
        start_time2=time.time()
        f_end=expected_f(x_k)
        overhead_time+=start_time2-time.time()
        time_end=time.time()-start_time-overhead_time
        P_2_f_v_batch_1.append(f_end)
        P_2_time_v_batch_1.append(time_end)
        m_k_sum_v_two.append(m_k_sum)

        if m_k_sum>max_m_k_sum:
            break

    print('1 iteration ends')

    #Save the evaluation value

    proposed_1_f_result.append(f_v[-1])
    proposed_1_time_result.append(time_v[-1])

    proposed_1_f_batch1_result.append(f_v_batch_1[-1])
    proposed_1_time_batch1_result.append(time_v_batch_1[-1])

    CZO_f_result.append(CZO_f_v[-1])
    CZO_time_result.append(CZO_time_v[-1])

    CZO_f_batch1_result.append(CZO_f_v_batch_1[-1])
    CZO_time_batch1_result.append(CZO_time_v_batch_1[-1])

    proposed_2_f_result.append(P_2_f_v_m[-1])
    proposed_2_time_result.append(time_2_v_two_m[-1])
    
    proposed_2_f_batch1_result.append(P_2_f_v_batch_1[-1])
    proposed_2_time_batch1_result.append(P_2_time_v_batch_1[-1])


#Calculate p-value of the proposed methods for the CZO (minibatch / batch size 1) method
f_p_proposed_1_CZO=stats.ttest_rel(proposed_1_f_result, CZO_f_result)[1]
f_p_proposed_1_CZO_batch_1=stats.ttest_rel(proposed_1_f_result, CZO_f_batch1_result)[1]

f_p_proposed_1_batch_1_CZO=stats.ttest_rel(proposed_1_f_batch1_result, CZO_f_result)[1]
f_p_proposed_1_batch_1_CZO_batch_1=stats.ttest_rel(proposed_1_f_batch1_result, CZO_f_batch1_result)[1]

f_p_proposed_2_CZO=stats.ttest_rel(proposed_2_f_result, CZO_f_result)[1]
f_p_proposed_2_CZO_batch_1=stats.ttest_rel(proposed_2_f_result, CZO_f_batch1_result)[1]

f_p_proposed_2_batch_1_CZO=stats.ttest_rel(proposed_2_f_batch1_result, CZO_f_result)[1]
f_p_proposed_2_batch_1_CZO_batch_1=stats.ttest_rel(proposed_2_f_batch1_result, CZO_f_batch1_result)[1]


#Outputs results
folder_name = "_".join(sys.argv[1:])

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

file_path = os.path.join(folder_name, 'results.csv')
with open(file_path, mode='w') as file_tmp:
    writer = csv.writer(file_tmp)
    writer.writerow(['Method','objective value','std_value','computation time (seconds)','std_time','p-value for CZO (minibatch)','p-value for CZO (batch size 1)'])
    writer.writerow(['Proposed-1 (minibatch)',statistics.mean(proposed_1_f_result),statistics.stdev(proposed_1_f_result),statistics.mean(proposed_1_time_result),statistics.stdev(proposed_1_time_result),f_p_proposed_1_CZO,f_p_proposed_1_CZO_batch_1])
    writer.writerow(['Proposed-1 (batch size 1)',statistics.mean(proposed_1_f_batch1_result),statistics.stdev(proposed_1_f_batch1_result),statistics.mean(proposed_1_time_batch1_result),statistics.stdev(proposed_1_time_batch1_result),f_p_proposed_1_batch_1_CZO,f_p_proposed_1_batch_1_CZO_batch_1])
    writer.writerow(['Proposed-2 (minibatch)',statistics.mean(proposed_2_f_result),statistics.stdev(proposed_2_f_result),statistics.mean(proposed_2_time_result),statistics.stdev(proposed_2_time_result),f_p_proposed_2_CZO,f_p_proposed_2_CZO_batch_1])
    writer.writerow(['Proposed-2 (batch size 1)',statistics.mean(proposed_2_f_batch1_result),statistics.stdev(proposed_2_f_batch1_result),statistics.mean(proposed_2_time_batch1_result),statistics.stdev(proposed_2_time_batch1_result),f_p_proposed_2_batch_1_CZO,f_p_proposed_2_batch_1_CZO_batch_1])
    writer.writerow(['CZO (minibatch)',statistics.mean(CZO_f_result),statistics.stdev(CZO_f_result),statistics.mean(CZO_time_result),statistics.stdev(CZO_time_result)])
    writer.writerow(['CZO (batch size 1)',statistics.mean(CZO_f_batch1_result),statistics.stdev(CZO_f_batch1_result),statistics.mean(CZO_time_batch1_result),statistics.stdev(CZO_time_batch1_result)])
    
save_plots(folder_name)
