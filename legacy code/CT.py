from contextlib import AbstractAsyncContextManager
import time
import math
import copy

import numpy as np
import pandas as pd
import pulp as pl

def mip(p_live, ftc_live, F_count, p_sim, ftc_sim, Fs_count, \
        A, A_SC, A_n, cf, M, M2, M4, M_night, P, period, Pfl2, Pfl4, QRA, mc_AC, sorties, *args, **kwargs):

    if len(args) != 0:
        x_start = args[0] 
        s_start = args[1]

    hist_p_live = copy.deepcopy(p_live)
    hist_p_sim = copy.deepcopy(p_sim)

    gradient = np.linspace(1, 0, num=(min(30, cf.tau - period)))

    if cf.tau - period <= 30:
        cg = gradient
    else:
        cg = np.full(cf.tau-period, 0.0)
        for i in range(len(gradient)):
            cg[i] = gradient[i]

    ftc_count = dict.fromkeys(P)
    ftcs_count = dict.fromkeys(P)

    for p in P:
        ftc_count[p] = min(F_count[p], cf.FTC_max[P[(p)][1]-1]) 
        ftcs_count[p] = min(Fs_count[p], cf.FTC_max[P[(p)][3]-1])
        for m in M:
            hist_p_live[p][m] = min(hist_p_live[p][m], M[m][P[p][1]]) 
            hist_p_sim[p][m] = min(hist_p_sim[p][m], M[m][P[p][3]])

    #--------------------PARAMETERS------------------------
    T = range(cf.tau - period)
    K = 1e3  #sufficiently large number

    #--------------------VARIABLES------------------------
    X = {(p,t,m):# pilot p assigned at time t to live mission m 
    pl.LpVariable(cat=pl.LpBinary, 
                name='assigned_{0}_{1}_{2}'.format(p,t,m)) 
    for p in P for t in T for m in M}  

    S = {(p,t,m):# pilot p assigned at time t to simulator mission m 
    pl.LpVariable(cat=pl.LpBinary, 
                name='Sassigned_{0}_{1}_{2}'.format(p,t,m)) 
    for p in P for t in T for m in M}  

    FTC = {(p,t,m):# pilot p assigned at time t to FTC mission m
    pl.LpVariable(cat=pl.LpBinary, 
                name='FTC_{0}_{1}_{2}'.format(p,t,m)) 
    for p in P for t in T for m in M}

    FTC_s = {(p,t,m):# pilot p assigned at time t to FTC simulator mission m
    pl.LpVariable(cat=pl.LpBinary, 
                name='FTCs_{0}_{1}_{2}'.format(p,t,m)) 
    for p in P for t in T for m in M}

    F = {(t,m):# Auxiliary var: at time t mission m 
    pl.LpVariable(cat=pl.LpInteger, 
                lowBound = 0,
                name='FlymissionL_{0}_{1}'.format(t,m)) 
    for t in T for m in M}  

    F_s = {(t,m):# Auxiliary var: at time t mission m 
    pl.LpVariable(cat=pl.LpInteger, 
                lowBound = 0,
                name='FlymissionS_{0}_{1}'.format(t,m)) 
    for t in T for m in M}  

    C = pl.LpVariable('objective', lowBound = 0, cat =pl.LpContinuous) 
    '''
    W = {(p):# Auxiliary var 
    pl.LpVariable(cat=pl.LpContinuous,
                lowBound = -1, 
                name='Out{0}'.format(p)) 
    for p in P}  
    
    V = {(p):# Auxiliary var 
    pl.LpVariable(cat=pl.LpInteger,
                lowBound = -2, 
                name='Out_sim{0}'.format(p)) 
    for p in P} ''' 



    #--------------------MODEL----------------------------
    model = pl.LpProblem('CT_opt', pl.LpMaximize)
  
    #--------------------GAMMA CONSTRAINTS-------------
    #--------------------Objective---------------
    model += C 
    
    model += C == pl.lpSum((1+cg[t]) * X[(p,t,m)] for p in P for t in T for m in M) + \
                pl.lpSum((1+cg[t]) * S[(p,t,m)] for p in P for t in T for m in M)+ \
                pl.lpSum(FTC[(p,t,m)] for p in P for t in T for m in M) + \
                pl.lpSum(FTC_s[(p,t,m)] for p in P for t in T for m in M)# + \
                #.9 * pl.lpSum(W[(p)] for p in P)# + .9 * pl.lpSum(V[(p)] for p in P)
 

    # Aim for max number of in YTP flights, penalty is incurred for out of YTP flights 
    #--------------------ALPHA CONSTRAINTS-------------
    # a pilot is either flying or in a sim not both
    for p in P:
        for t in T:
            model += pl.lpSum(X[(p,t,m)] for m in M) + \
                     pl.lpSum(S[(p,t,m)] for m in M) + \
                     pl.lpSum(FTC[(p,t,m)] for m in M) + \
                     pl.lpSum(FTC_s[(p,t,m)] for m in M)\
                     <= max(A[(p)][t+period] - QRA[p][t+period], 0)
                
    # A mission has certified flight leads      
    for t in T:
        for m in M2:
            model += F[(t,m)] <= pl.lpSum(X[(p,t,m)] for p in Pfl2) + \
                        pl.lpSum(FTC[(p,t,m)] for p in Pfl2)
            model += F_s[(t,m)] <= pl.lpSum(S[(p,t,m)] for p in Pfl2) + \
                        pl.lpSum(FTC_s[(p,t,m)] for p in Pfl2)
                
    for t in T:
        for m in M4:
            model += F[(t,m)] <= pl.lpSum(X[(p,t,m)] for p in Pfl4) + \
                        pl.lpSum(FTC[(p,t,m)] for p in Pfl4)  
            model += F_s[(t,m)] <= pl.lpSum(X[(p,t,m)] for p in Pfl4) + \
                        pl.lpSum(FTC_s[(p,t,m)] for p in Pfl4)  
            model += 2 * F[(t,m)] <= pl.lpSum(X[(p,t,m)] for p in Pfl2) + \
                        pl.lpSum(FTC[(p,t,m)] for p in Pfl2)  
            model += 2 * F_s[(t,m)] <= pl.lpSum(X[(p,t,m)] for p in Pfl2) + \
                        pl.lpSum(FTC_s[(p,t,m)] for p in Pfl2)    
                
    #--------------------BETA CONSTRAINTS-------------
    #for p in P:
        #model += W[(p)] <= 0 
        #model += V[(p)] <= 0 

    # max number of flights is determined by the FMP, sim sorties are based on simulator availabilty
    
    for t in T:
        model += pl.lpSum(X[(p,t,m)] for p in P for m in M) + \
                 pl.lpSum(FTC[(p,t,m)] for p in P for m in M) \
                 <= math.ceil(sorties[t + period]*cf.sortie_correct)
        model += pl.lpSum(S[(p,t,m)] for p in P for m in M) + \
                 pl.lpSum(FTC_s[(p,t,m)] for p in P for m in M) \
                 <= cf.SC_max*cf.sim_waves*A_SC[t + period]
        #model += pl.lpSum(X[(p,t,m)] for p in P for m in M_night) + \
            #     pl.lpSum(FTC[(p,t,m)] for p in P for m in M_night) \
             #    <= mc_AC[t + period]

    # Missions in YTP are limited to max executions 
    for p in P:
        model += cf.FTC_max[P[(p)][1]-1] + math.ceil((cf.FTC_extra[P[p][1]-1] * (cf.tau-period)/cf.tau)) - ftc_count[p] +2 >= pl.lpSum(FTC[(p,t,m)] for t in T for m in M)    
                     
        model += cf.FTC_max[P[(p)][3]-1] + math.ceil((cf.FTC_extra[P[p][3]-1] * (cf.tau-period)/cf.tau)) - ftcs_count[p] +2 >= pl.lpSum(FTC_s[(p,t,m)] for t in T for m in M)        
        #V[(p)] <= #W[(p)] <= 

        for m in M:
            model += M[m][P[(p)][0]] <= \
                    pl.lpSum(X[(p,t,m)] for t in T) + \
                    hist_p_live[p][m] <= M[m][P[p][1]]
            model += M[m][P[(p)][2]] <= \
                    pl.lpSum(S[(p,t,m)] for t in T) + \
                    hist_p_sim[p][m] <= M[m][P[p][3]]
                    
    #disallow night missions during summer period
    for t in T:
        model += pl.lpSum(X[(p,t,m)] for p in P for m in M_night) + \
                 pl.lpSum(FTC[(p,t,m)] for p in P for m in M_night) \
                 <= A_n[t + period] * mc_AC[t + period]

    # force aux var F to indicate mission execution      
    for t in T:
        model += F[(t,'BFM')] == pl.lpSum(X[(p,t,'BFM')] for p in P) + \
                 pl.lpSum(FTC[(p,t,'BFM')] for p in P)
        model += F_s[(t,'BFM')] == pl.lpSum(S[(p,t,'BFM')] for p in P) + \
                 pl.lpSum(FTC_s[(p,t,'BFM')] for p in P)
        for m in M2:
            model += F[(t,m)] * 2 == pl.lpSum(X[(p,t,m)] for p in P) + \
                     pl.lpSum(FTC[(p,t,m)] for p in P)
            model += F_s[(t,m)] * 2 == pl.lpSum(S[(p,t,m)] for p in P) + \
                     pl.lpSum(FTC_s[(p,t,m)] for p in P)
        for m in M4:
            model += F[(t,m)] * 4 == pl.lpSum(X[(p,t,m)] for p in P) + \
                     pl.lpSum(FTC[(p,t,m)] for p in P)
            model += F_s[(t,m)] * 4 == pl.lpSum(S[(p,t,m)] for p in P) + \
                     pl.lpSum(FTC_s[(p,t,m)] for p in P)

    # if the free FL4 policy is active apply the following constraint
    if cf.free_FL4:
        for t in T:
            model += pl.lpSum(X[(p,t,m)] for p in Pfl4 for m in M) + \
                     pl.lpSum(FTC[(p,t,m)] for p in Pfl4 for m in M) + \
                     pl.lpSum(S[(p,t,m)] for p in Pfl4 for m in M) + \
                     pl.lpSum(FTC_s[(p,t,m)] for p in Pfl4 for m in M) \
                     <= len(Pfl4.keys()) - 1
    
    if len(args) != 0:
        print('prestarting solution with previous schedule')
        for p in P:
            for t in T:
                for m in M:
                    if x_start.loc[p][t+period] == m:
                        X[(p,t, m)].setInitialValue(1)
                    else:
                        X[(p,t, m)].setInitialValue(0)
                    if s_start.loc[p][t+period] == m:
                        S[(p,t, m)].setInitialValue(1)
                    else:
                        S[(p,t, m)].setInitialValue(0)
    
    #--------------------RUN MODEL------------------------
    t1 = time.perf_counter()
    status = model.solve(cf.solver)
    #--------------------WRITE SOLUTION-------------------
    #TODO write func to anutomate this messy data formatting

    t2 = time.perf_counter() - t1

    if cf.verbose:
        print(pl.LpStatus[status])
        print('\nruntime: %f sec' %t2) 

    F_solve = np.array([[F[t,m].varValue for t in T] for m in M])
    F  = pd.DataFrame(F_solve, index=M)

    b = np.full((len(P),len(T)), '')
    x = pd.DataFrame(b, index=P, dtype = 'str')
    for m in M:
        for p in P:
            for t in T:
                if X[p,t,m].varValue == 1:
                    x.at[p, t] += m
    d = np.full((len(P),len(T)), '')
    ftc = pd.DataFrame(d, index=P, dtype = 'str')
    for m in M:
        for p in P:
            for t in T:
                if FTC[p,t,m].varValue == 1:
                    ftc.at[p, t] += m

    e = np.full((len(P),len(T)), '')
    s = pd.DataFrame(e, index=P, dtype = 'str')
    for m in M:
        for p in P:
            for t in T:
                if S[p,t,m].varValue == 1:
                    s.at[p, t] += m
    g = np.full((len(P),len(T)), '')
    ftcs = pd.DataFrame(g, index=P, dtype = 'str')
    for m in M:
        for p in P:
            for t in T:
                if FTC_s[p,t,m].varValue == 1:
                    ftcs.at[p, t] += m
    
    return x, ftc, s, ftcs, F