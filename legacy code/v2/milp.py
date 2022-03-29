import time

import numpy as np
import pandas as pd
import pulp as pl

def mip(hist_p_live, A, cf, M, M2, M4, P, period, Pfl2, Pfl4, QRA, sorties, *args, **kwargs):

    for p in P:
        for m in M:
            hist_p_live[p][m] = min(hist_p_live[p][m], M[m][P[p][1]])


    T = range(cf.tau - period)
    cg = np.linspace(1, 0, num=(cf.tau - period))

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

    X_a = {(p,t,m):# pilot p assigned at time t to live mission m 
    pl.LpVariable(cat=pl.LpBinary, 
                name='assignedoverYTP_{0}_{1}_{2}'.format(p,t,m)) 
    for p in P for t in T for m in M}

    S_a = {(p,t,m):# pilot p assigned at time t to simulator mission m 
    pl.LpVariable(cat=pl.LpBinary, 
                name='SassignedoverYTP_{0}_{1}_{2}'.format(p,t,m)) 
    for p in P for t in T for m in M}

    O = {(p,t):# pilot p assigned to office shift at time t
    pl.LpVariable(cat=pl.LpBinary, 
                name='office_{0}_{1}'.format(p,t)) 
    for p in P for t in T}  

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
    #--------------------MODEL----------------------------
    model = pl.LpProblem('CT_opt', pl.LpMaximize)
            
    #--------------------GAMMA CONSTRAINTS-------------
    #--------------------Objective---------------
    model += C

    # Aim for max number of in YTP flights, penalty is incurred for out of YTP flights 
    model += C == pl.lpSum(cg[t] * X[(p,t,m)] for p in P for t in T for m in M) + \
                  pl.lpSum(S[(p,t,m)] for p in P for t in T for m in M) - \
                  0.9 * pl.lpSum(X_a[(p,t,m)] for p in P for t in T for m in M2)- \
                  0.3 * pl.lpSum(X_a[(p,t,m)] for p in P for t in T for m in M4)

    #--------------------ALPHA CONSTRAINTS-------------
    # a pilot is either flying or in the office 
    for p in P:
        for t in T:
            model += pl.lpSum(X[(p,t,m)] for m in M) + \
                     pl.lpSum(S[(p,t,m)] for m in M) + \
                     O[(p,t)] + pl.lpSum(X_a[(p,t,m)] for m in M) + \
                     pl.lpSum(S_a[(p,t,m)] for m in M) + \
                     pl.lpSum(FTC[(p,t,m)] for m in M) \
                     == A[(p)][t]
                
    # A mission has minimum 1 certified flight lead       
    for t in T:
        for m in M:
            if M[m][0] == 2:
                model += F[(t,m)] <= pl.lpSum(X[(p,t,m)] for p in Pfl2) + \
                         pl.lpSum(X_a[(p,t,m)] for p in Pfl2) + \
                         pl.lpSum(FTC[(p,t,m)] for p in Pfl2)
                model += F_s[(t,m)] <= pl.lpSum(S[(p,t,m)] for p in Pfl2) + \
                         pl.lpSum(S_a[(p,t,m)] for p in Pfl2) + \
                         pl.lpSum(FTC_s[(p,t,m)] for p in Pfl2)
                
    for t in T:
        for m in M:
            if M[m][0] == 4:
                model += F[(t,m)] <= pl.lpSum(X[(p,t,m)] for p in Pfl4) + \
                         pl.lpSum(X_a[(p,t,m)] for p in Pfl4) + \
                         pl.lpSum(FTC[(p,t,m)] for p in Pfl4)  
                model += F_s[(t,m)] <= pl.lpSum(X[(p,t,m)] for p in Pfl4) + \
                         pl.lpSum(S_a[(p,t,m)] for p in Pfl4) + \
                         pl.lpSum(FTC_s[(p,t,m)] for p in Pfl4)   
                
    #--------------------BETA CONSTRAINTS-------------
            
    # max number of flights is max AC
    for t in T:
        model += pl.lpSum(X[(p,t,m)] for p in P for m in M) + \
                 pl.lpSum(X_a[(p,t,m)] for p in P for m in M) + \
                 pl.lpSum(FTC[(p,t,m)] for p in P for m in M) <= sorties[t]*cf.sortie_correct
        model += pl.lpSum(S[(p,t,m)] for p in P for m in M) + \
                 pl.lpSum(S_a[(p,t,m)] for p in P for m in M) + \
                 pl.lpSum(FTC_s[(p,t,m)] for p in P for m in M) <= cf.SC_tmax

    # Missions in YTP are limited to max executions 
    for p in P:
        model += cf.FTC_max[P[(p)][0]-1] <= pl.lpSum(FTC[(p,t,m)] for t in T for m in M) \
                 <= cf.FTC_max[P[(p)][1]-1]
        model += cf.FTC_max[P[(p)][2]-1] <= pl.lpSum(FTC_s[(p,t,m)] for t in T for m in M) \
                 <= cf.FTC_max[P[(p)][3]-1]
        for m in M:
            model += M[m][P[(p)][0]] <= pl.lpSum(X[(p,t,m)] for t in T) + hist_p_live[p][m] <= M[m][P[p][1]]
            model += M[m][P[(p)][2]] <= pl.lpSum(S[(p,t,m)] for t in T) <= M[m][P[p][3]]


    # force aux var F to indicate mission execution      
    for t in T:
        model += F[(t,'BFM')] == pl.lpSum(X[(p,t,'BFM')] for p in P) + \
                 pl.lpSum(FTC[(p,t,'BFM')] for p in P)
        model += F_s[(t,'BFM')] == pl.lpSum(S[(p,t,'BFM')] for p in P) + \
                 pl.lpSum(FTC_s[(p,t,'BFM')] for p in P)
        for m in M2:
            model += F[(t,m)] * 2 == pl.lpSum(X[(p,t,m)] for p in P) + \
                     pl.lpSum(X_a[(p,t,m)] for p in P) + \
                     pl.lpSum(FTC[(p,t,m)] for p in P)
            model += F_s[(t,m)] * 2 == pl.lpSum(S[(p,t,m)] for p in P) + \
                     pl.lpSum(S_a[(p,t,m)] for p in P) + \
                     pl.lpSum(FTC_s[(p,t,m)] for p in P)
        for m in M4:
            model += F[(t,m)] * 4 == pl.lpSum(X[(p,t,m)] for p in P) + \
                     pl.lpSum(X_a[(p,t,m)] for p in P) + \
                     pl.lpSum(FTC[(p,t,m)] for p in P)
            model += F_s[(t,m)] * 4 == pl.lpSum(S[(p,t,m)] for p in P) + \
                     pl.lpSum(S_a[(p,t,m)] for p in P) + \
                     pl.lpSum(FTC_s[(p,t,m)] for p in P)


    #--------------------RUN MODEL------------------------
    t1 = time.perf_counter()

    #print(model)

    status = model.solve(cf.solver)

    #--------------------WRITE SOLUTION-------------------
    #TODO write func to anutomate this messy data formatting

    t2 = time.perf_counter() - t1

    if cf.verbose:
        print(pl.LpStatus[status])
        print('\nruntime: %f sec' %t2) 

    O_solve = np.array([[O[p, t].varValue for t in T] for p in P])
    O  = pd.DataFrame(O_solve, index=P)
        
    F_solve = np.array([[F[t,m].varValue for t in T] for m in M])
    F  = pd.DataFrame(F_solve, index=M)

    b = np.full((len(P),len(T)), '')
    x = pd.DataFrame(b, index=P, dtype = 'str')
    for m in M:
        for p in P:
            for t in T:
                if X[p,t,m].varValue == 1:
                    x.at[p, t] += m
    c = np.full((len(P),len(T)), '')
    x_a = pd.DataFrame(c, index=P, dtype = 'str')
    for m in M:
        for p in P:
            for t in T:
                if X_a[p,t,m].varValue == 1:
                    x_a.at[p, t] += m
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
    f = np.full((len(P),len(T)), '')
    s_a = pd.DataFrame(f, index=P, dtype = 'str')
    for m in M:
        for p in P:
            for t in T:
                if S_a[p,t,m].varValue == 1:
                    s_a.at[p, t] += m
    g = np.full((len(P),len(T)), '')
    ftcs = pd.DataFrame(g, index=P, dtype = 'str')
    for m in M:
        for p in P:
            for t in T:
                if FTC_s[p,t,m].varValue == 1:
                    ftcs.at[p, t] += m
    
    return x, x_a, ftc, s, s_a, ftcs, O, F