'''
this file contains the functions that initialize the datasets if parameters
are defined in config.py and randomize input is TRUE the model input is
generated 
'''
import warnings
import csv

import numpy as np
import pandas as pd

from config import Configuration
from utils import date_indices

cf = Configuration() # read config settings as cf

def AC_availability(cf):
    '''A_AC is used to determine what days are used to schedule sorties
    
    Returns
    ------- 
    A_AC: array'''

    A_AC = np.full(cf.tau,1)
    i = 6
    for k in range(len(A_AC)):
        if i%7 == 0:
            A_AC[k] = 0
            if k+1 < cf.tau:
                A_AC[k+1] = 0
        i += 1
    return A_AC

def aircraft():
    '''
    A/C data for the FMP module. Each A/C has an array to init t=0 in the FMP:
    Returns
    ------- 
    AC: dict
    values:
    |--SVC--|--RFT--|--RMT--|
    [0,        1,       2   ]
    '''
    AC = {
    'J-003': [1,200,0],
    'J-006': [1,176,0],
    'J-020': [1,159,0],
    'J-021': [1,145,0],
    'J-055': [1,142,0],
    'J-062': [1,134,0],
    'J-063': [1,124,0],
    'J-136': [1,100,0],
    'J-368': [1,87,0],
    'J-514': [1,79,0],
    'J-515': [1,61,0],
    'J-641': [1,53,0],
    'J-882': [1,25,0],
    'J-914': [0,0,15]
    }
    return AC  

def FMP_sortiecount(cf):
    '''sortie info is loaded from csv files. The SVC variable from the FMP is used 
    to calculate the number of mission capable A/C for each t in the planning period
    Returns
    -------
    FMP_sorties: array
    mc_AC: array
    '''
    FMP_sorties = np.genfromtxt(cf.FMP_file, delimiter=',')
    _svc = np.genfromtxt(cf.svc_file, delimiter=',')
    mc_AC = np.around(_svc*cf.percentage_AC_mc, decimals=0)
    ft = pd.read_csv('data/FT.csv', index_col=0, header=0).fillna('')
    ft.columns = ft.columns.astype(int)
    return ft, FMP_sorties, mc_AC

def maintenance_activities():
    #transition chances [no change, 1st SMALL, 1st LARGE, 2nd LARGE]
    transitions = {
        'pre_flight' : [.98, .01, .01, 0],
        'post_flight' : [.88, .07, .03, .02],
        'start_up' : [.97, .01, .01, .01],
        'engine_inspection' : [.95, 0, 0, .05],
        'landinggear_inspection' : [.97, 0, .01, .02]
    }
    
    transition_activities = ['pass', '1_small', '1_large', '2_large']
    
    #duration [mean (hours), variance (hours), distribution]
    activity_duration = {
        '1_small' : [4, 1, 'gamma'],
        '1_large' : [10, 2, 'normal'],
        '2_large' : [30, 5, 'gamma'],
    }
    for k in transitions.keys():
        if sum(transitions[k]) > 1:
            warnings.warn('error transition odds greater than 1 for: %s' % k)
    return transitions, transition_activities, activity_duration

def missions(cf):
    '''generate misson data from csv. M is the full mission set, M2, M4 and M_night
    are subsets containing respectively all: fl2, fl4 and night missions. The A_n array 
    determines if execution of night mission is allowed for each t in the planning period
    
    Returns
    -------
    M: dict 
    M2: dict
    M4: dict
    M_night: dict
    A_n: array'''

    M = {}
    if cf.verbose:
        print('aquire mission data')
    with open(cf.mission_file, mode='r',encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for row in reader:
            M[row[0]] = np.genfromtxt(row[1:10], dtype='int')
    M2 = {k:v for k,v in M.items() if v[0] == 2}
    M4 = {k:v for k,v in M.items() if v[0] == 4}
    M_night = {k:v for k,v in M.items() if '_n_'in k}

    n_ = date_indices(cf.startdate, cf.night_allowed)
    A_n = np.full((cf.tau), 1)
    A_n[n_[0]:n_[1]] = 0

    return M, M2, M4, M_night, A_n

def pilots(cf):
    '''initialize for the pilots
    
    config.py lets you select either imported pilots from CSV or randomly generated
    pilots using various settings in config. Three dicts are initialized:

    P (dict) keys are either imported names or generic P<int> assignments. Every value
    consists of an array with 5 values:

    0 : index for minimum number of live YTP missions
    1 : index for desired number of live YTP missions
    2 : index for minimum number of sim YTP missions
    3 : index for desired number of sim YTP missions
    4 : flight lead status of pilot

    These values relate to the mission defined in the YTP. 

    Table YTP by dict (M) value index
    -size-|------------- Desired YTP------|-----------Minimum YTP---------|
    ------|--Exp pilots--|--inexp pilots--|--Exp pilots--|--inexp pilots--|
    ------|--Live-|--Sim-|--Live--|--Sim--|--Live-|--Sim-|--Live--|--Sim--|
    [0,        1,     2,      3,      4,       5,     6,      7,      8   ]

    '''
    P = {}
    if cf.random_pilots:
        if cf.verbose:
            print('get randos')
        if cf.no_FL4pilots > cf.no_EXPpilots and cf.verbose:
            print('More FL4 pilots than EXP pilots assigned. FL4 reduced to max EXP')
        for i in range(cf.no_pilots):
            if i+1<=cf.no_EXPpilots and i+1<=cf.no_FL4pilots:
                P['P%d'%(i+1)] = [5,1,6,2,4]
            elif i+1<=cf.no_EXPpilots and i+1>cf.no_FL4pilots:
                P['P%d'%(i+1)] = [5,1,6,2,2]
            else:
                P['P%d'%(i+1)] = [7,3,8,4,0]
    else:
        if cf.verbose:
            print('aquire pilot data')
        with open(cf.pilot_file, mode='r',encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row in reader:
                P[row[0]] = np.genfromtxt(row[1:6], dtype='int')
    Pfl2 = {k:v for k,v in P.items() if v[4] >= 2}
    Pfl4 = {k:v for k,v in P.items() if v[4] >= 4}

    return P, Pfl2, Pfl4
 
def pilot_availability(cf, P): 
    '''Pilot availability is determined in threee steps. First weekends are 
    deducted from the availability list. If the toggel is set in config step 2
    deducts vacation days for each pilot. 2 consecutive weeks during the summer
    3 random weeks throughout the year. In the final step all pilots are blocked for
    4 manditory sqn training days
    
    Returns
    ------- 
    A: dict
    P_status: pandas Series
    Markov_p: list(2x2)'''

    A = dict.fromkeys(P)
    for k in A.keys():
        A[k] = np.full(cf.tau, 1)
    for k in A.keys():
        i = 6
        for t in range(cf.tau):
            if i%7 == 0:
                A[(k)][t] = 0
                if t+1 < cf.tau:
                    A[(k)][t+1] = 0
            i += 1
    if cf.vacation:
        for k in A.keys():
            summer = np.random.choice(cf.weeks_off[1],1,replace=False)
            not_summer = np.random.choice(cf.weeks_off[0],3,replace=False)
            summer = date_indices(cf.startdate, summer)
            not_summer = date_indices(cf.startdate, not_summer)
            for i in not_summer:
                A[k][i:i+5] = np.full((5), 0)        
            for j in summer:
                A[k][j:j+13] = np.full((13), 0)
    if cf.training:
        train = date_indices(cf.startdate, cf.team_training)
        for k in A.keys():
            for i in train:
                A[k][i] = 0
    '''At the start of the year all pilots are assumed to be fit for duty
    a discrete Markov chain is used to determine wheter the pilot health 
    state is modified at each timestep'''
    P_status = pd.Series(np.full(len(P), 1), index=P.keys(), dtype=int)
    #discrete Markov state
    # [[P_ill_ill, P_ill_notill], [P_notill_ill. P_notill_notill]]
    Markov_P = [[.822, .178], [.0059, .9941]]
    return A, P_status, Markov_P

def QRA_availability(cf, P):
    '''QRA is scheduled for each pilot in order over the full QRA period
    each time a pilot has QRA duties he/she cannot perform other duties 1 
    day prior and 1 day after the QRA duty
    
    Returns
    ------- 
    QRA: dict'''

    QRA = dict.fromkeys(P)
    for k in QRA.keys():
        QRA[k] = np.full(cf.tau, 0)   
    sjaak = list(QRA.keys())
    for i in range(cf.QRA_start, cf.QRA_start+cf.QRA_length):
        QRA[sjaak[0]][i:i+3] = 1
        QRA[sjaak[1]][i:i+3] = 1
        del sjaak[0:2]
        if len(sjaak)<2:
            sjaak = sjaak + list(QRA.keys())
    return QRA

def schedule_init():
    x = pd.read_csv('data/x.csv', index_col=0, header=0).fillna('')
    x.columns = x.columns.astype(int)
    ftc = pd.read_csv('data/ftc.csv', index_col=0, header=0).fillna('')
    ftc.columns = ftc.columns.astype(int)
    s = pd.read_csv('data/s.csv', index_col=0, header=0).fillna('')
    s.columns = s.columns.astype(int)
    ftcs = pd.read_csv('data/ftcs.csv', index_col=0, header=0).fillna('')
    ftcs.columns = ftcs.columns.astype(int)

    return x, ftc, s, ftcs

def simulator_availability(cf):
    '''Availability for simulators
    
    Returns
    -------
    A_SC: array'''

    A_SC = np.full((cf.tau), 1)
    prev_maint = date_indices(cf.startdate, cf.SC_preventive_maintenance)
    for i in prev_maint:
        A_SC[i] = 0
    return A_SC

class Sim_history():
    '''class contains trackers'''

    def __init__(self, P, M, cf, activity):

        self.P_live = {p: {m: int(0) for m in M.keys()} for p in P.keys()}
        self.P_live2 = {p: {m: int(0) for m in M.keys()} for p in P.keys()}
        self.P_sim = {p: {m: int(0) for m in M.keys()} for p in P.keys()}
        self.FTC_live = {p: {m: int(0) for m in M.keys()} for p in P.keys()}
        self.FTC_sim = {p: {m: int(0) for m in M.keys()} for p in P.keys()}
        self.FTC_count = {p: int(0) for p in P.keys()}
        self.FTCs_count = {p: int(0) for p in P.keys()}
        self.maintenance = {a: int(0) for a in activity.keys()}
        self.FHR_realized = 0
        self.weather_cancelations = 0
        self.weather_sorties = 0
        self.sortie_cancel = 0
        self.sortie_executed = 0 
        self.sim_sortie_cancel = 0
        self.sim_sortie_executed = 0 
        self.FHR_accumulated = np.full((cf.tau), 0, dtype=np.float64)
        self.regen = []
        self.regen_period = []

        self.sortielist = np.full((cf.tau), 0, dtype=np.float64)
        self.sortiecount = np.full((cf.tau), 0, dtype=np.float64)
