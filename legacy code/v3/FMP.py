import random
import time

import numpy as np
import pandas as pd

import pulp as pl
'''
The FMP model is based on the following publication:

M. Verhoeff et al. (2015) Maximizing Operational Readiness in Military Aviation
by Optimizing Flight and Maintenance Planning DOI:10.1016/j.trpro.2015.09.048
'''
def fmp(svc_init, rft_init, rmt_init, AC, A_AC, cf, tau, *args, **kwargs):
    
    T = range(tau)
    print(T)
    ACR_t = len(AC)*cf.ACR
    print(ACR_t)
    FHR_tot = cf.FHR * (tau/365)
    print(FHR_tot)

    K = 10000

    FHR_t = (FHR_tot/sum(A_AC))*A_AC
    print(FHR_t)

    RFT = {(n,t):#residual flight time of AC n at time t
    pl.LpVariable(cat=pl.LpContinuous,lowBound = 0, 
                name='RFT_{0}_{1}'.format(n,t)) 
    for n in AC for t in T}  

    RMT = {(n,t):#residual maintenance time of AC n at time t
    pl.LpVariable(cat=pl.LpContinuous,lowBound = 0, 
                name='RMT_{0}_{1}'.format(n,t)) 
    for n in AC for t in T}  

    SVC = {(n,t):#servicability of AC n at time t
    pl.LpVariable(cat=pl.LpBinary, 
                name='SVC_{0}_{1}'.format(n,t)) 
    for n in AC for t in T}  

    OPR = {(n,t):#operational AC n at time t
    pl.LpVariable(cat=pl.LpBinary, 
                name='OPR_{0}_{1}'.format(n,t)) 
    for n in AC for t in T}  

    FT = {(n,t):#assigned flight time of AC n at time t
    pl.LpVariable(cat=pl.LpContinuous,lowBound = 0, 
                name='FT_{0}_{1}'.format(n,t)) 
    for n in AC for t in T}  

    MT = {(n,t):#assigned maintenance time of AC n at time t
    pl.LpVariable(cat=pl.LpContinuous,lowBound = 0, 
                name='MT_{0}_{1}'.format(n,t)) 
    for n in AC for t in T}  

    MS = {(n,t):#maintenance start time t for AC n
    pl.LpVariable(cat=pl.LpBinary, 
                name='MS_{0}_{1}'.format(n,t)) 
    for n in AC for t in T}  

    MR = {(n,t):#maintenance end time t for AC n
    pl.LpVariable(cat=pl.LpBinary, 
                name='MR_{0}_{1}'.format(n,t)) 
    for n in AC for t in T}  

    P = {(n,t):#aux var 1
    pl.LpVariable(cat=pl.LpBinary, 
                name='P_{0}_{1}'.format(n,t)) 
    for n in AC for t in T}  

    R = {(n,t):#aux var 2
    pl.LpVariable(cat=pl.LpBinary, 
                name='R_{0}_{1}'.format(n,t)) 
    for n in AC for t in T}  

    Sust_min = pl.LpVariable('objective', lowBound = 0, cat =pl.LpContinuous) 

    #--------------------MODEL----------------------------
    model = pl.LpProblem('FMP', pl.LpMaximize)

    #--------------------CONSTRAINTS-------------
    #--------------------Objective---------------
    model += Sust_min

    for t in T:
        model += Sust_min <= pl.lpSum(RFT[n,t] for n in AC)                             #EQ 5
    for n in AC:
        for t in T:
            model += RFT[(n,t)] + K * P[(n,t)] <= K                                     #EQ 6
            model += RMT[(n,t)] + K * R[(n,t)] <= K                                     #EQ 8
            
        for t in range(0, tau-1):  
            model += SVC[(n,t+1)] <= (RFT[(n,t)]-FT[(n,t)]) * K + K * P[(n,t)]          #EQ 7
            model += 1- SVC[(n,t+1)] <= (RMT[(n,t)]-MT[(n,t)]) * K + K * R[(n,t)]       #EQ 9
            
    for n in AC:
        for t in range(0, tau-1):  
            model += RFT[(n,t+1)] == RFT[(n,t)] - FT[(n,t)] + MR[(n,t+1)] * cf.RFT_max  #EQ 10
            model += MR[(n,t+1)] >= SVC[(n,t+1)] - SVC[(n,t)]                           #EQ 11
            model += 0.1 <= SVC[(n,t+1)] - SVC[(n,t)] + 1.1 * (1-MR[(n,t+1)])           #EQ 12
            
    for n in AC:
        for t in range(0, tau-1):  
            model += RMT[(n,t+1)] == RMT[(n,t)] - MT[(n,t)] + MS[(n,t+1)] * cf.RMT_max  #EQ 13
            model += MS[(n,t+1)] >= SVC[(n,t)] - SVC[(n,t+1)]                           #EQ 14
            model += 0.1 <= SVC[(n,t)] - SVC[(n,t+1)] + 1.1 * (1-MS[(n,t+1)])           #EQ 15
            
    for n in AC:
        for t in T:
            model += RFT[(n,t)] <= SVC[(n,t)] * cf.RFT_max                              #EQ 16
            model += FT[(n,t)] <= RFT[(n,t)]                                            #EQ 17
            model += RMT[(n,t)] <= (1-SVC[(n,t)]) * cf.RMT_max                          #EQ 18
            model += MT[(n,t)] <= RMT[(n,t)]                                            #EQ 19
            model += 1 - SVC[(n,t)] <= MT[(n,t)]                                        #EQ 20
            
    model += pl.lpSum(FT[(n,t)] for n in AC for t in T) >= FHR_tot                      #EQ 21

    for t in T:
        model += (1-cf.tol_FHR) * FHR_t[t] <= pl.lpSum(FT[(n,t)] for n in AC) <= \
            (1+cf.tol_FHR) * FHR_t[t]                                                   #EQ 22
        for n in AC:
            model += 0.1 <= FT[(n,t)] + K * (1-OPR[(n,t)]) <= K                         #EQ 23
        model += pl.lpSum(OPR[(n,t)] for n in AC) >= ACR_t                              #EQ 24
        for n in AC:
            model += FT[(n,t)] <= OPR[(n,t)] * FHR_t[t] / ACR_t                         #EQ 25
        model += pl.lpSum(1-SVC[(n,t)] for n in AC) <= cf.M_max                         #EQ 26
        for n in AC:
            model += MT[(n,t)] <= cf.MT_max                                             #EQ 27
            model += RFT[(n,t)] >= SVC[(n,t)] * cf.RFT_min                              #EQ 28

    #from parameters Verhoeff(2015, pp 946.)       
    for n in AC:
        model += SVC[(n,0)] == svc_init[n]
        model += RFT[(n,0)] == rft_init[n]
        model += RMT[(n,0)] == rmt_init[n]

    t1 = time.perf_counter()
    status = model.solve(pl.GUROBI_CMD(options=[('MIPGap',0.01)]) )
    t2 = time.perf_counter() - t1
    print('\nruntime: %f sec' %t2)
    print(pl.LpStatus[status])  

    writer = pd.ExcelWriter('FMP_5.xlsx')

    ft_solve = np.array([[FT[n, t].varValue for t in T] for n in AC])
    ft  = pd.DataFrame(ft_solve, index=AC)
    ft.to_excel(writer, 'FT', index=True)

    rft_solve = np.array([[RFT[n, t].varValue for t in T] for n in AC])
    rft  = pd.DataFrame(rft_solve, index=AC)
    rft.to_excel(writer, 'RFT', index=True)

    rmt_solve = np.array([[RMT[n, t].varValue for t in T] for n in AC])
    rmt  = pd.DataFrame(rmt_solve, index=AC)
    rmt.to_excel(writer, 'RMT', index=True)

    svc_solve = np.array([[SVC[n, t].varValue for t in T] for n in AC])
    svc  = pd.DataFrame(svc_solve, index=AC)
    svc.to_excel(writer, 'SVC', index=True)


    writer.save()

    return svc, rft, rmt, ft