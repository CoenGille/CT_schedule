'''
this file contains the functions that initialize the datasets if parameters
are defined in config.py and randomize input is TRUE the model input is
generated 
'''

from glob import glob
import warnings
import csv

import numpy as np
import pandas as pd

from config import Configuration
from utils import check_folder

cf = Configuration() # read config settings as cf

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

def aircraft():
    AC = {
    'J-003': [1,200,0],
    'J-006': [1,180,0],
    'J-020': [1,163,0],
    'J-021': [1,150,0],
    'J-055': [1,142,0],
    'J-062': [1,134,0],
    'J-063': [1,125,0],
    'J-136': [1,100,0],
    'J-368': [1,87,0],
    'J-514': [1,85,0],
    'J-515': [1,63,0],
    'J-641': [1,53,0],
    'J-882': [1,32,0],
    'J-914': [0,0,15]
    }
    return AC   

def FMP_sortiecount(cf):
    FMP_sorties = np.genfromtxt(cf.FMP_file, delimiter=',')
    _svc = np.genfromtxt(cf.svc_file, delimiter=',')
    svc = np.around(_svc*cf.percentage_AC_svc, decimals=0)
    return FMP_sorties, svc

def schedule_init():
    x = pd.read_csv('x.csv', index_col=0, header=0).fillna('')
    x.columns = x.columns.astype(int)
    x_a = pd.read_csv('x_a.csv', index_col=0, header=0).fillna('')
    x_a.columns = x_a.columns.astype(int)
    ftc = pd.read_csv('ftc.csv', index_col=0, header=0).fillna('')
    ftc.columns = ftc.columns.astype(int)
    s = pd.read_csv('s.csv', index_col=0, header=0).fillna('')
    s.columns = s.columns.astype(int)
    s_a = pd.read_csv('s_a.csv', index_col=0, header=0).fillna('')
    s_a.columns = s_a.columns.astype(int)
    ftcs = pd.read_csv('ftcs.csv', index_col=0, header=0).fillna('')
    ftcs.columns = ftcs.columns.astype(int)

    return x, x_a, ftc, s, s_a, ftcs

def pilot_availability(cf, P): #TODO import holidays and training dates

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
    return A

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


def QRA_availability(cf, P):
    QRA = dict.fromkeys(P)
    for k in QRA.keys():
        QRA[k] = np.full(cf.tau, 0)   
    sjaak = list(QRA.keys())
    #print(sjaak)
    for i in range(cf.QRA_start, cf.QRA_start+30):
        QRA[sjaak[0]][i:i+3] = 1
        QRA[sjaak[1]][i:i+3] = 1
        del sjaak[0:2]
        if len(sjaak)<2:
            sjaak = sjaak + list(QRA.keys())
    return QRA

def AC_availability(cf):
    '''returns set A_AC: array'''
    A_AC = np.full(cf.tau,1)
    i = 6
    for k in range(len(A_AC)):
        if i%7 == 0:
            A_AC[k] = 0
            if k+1 < cf.tau:
                A_AC[k+1] = 0
        i += 1
    return A_AC

def missions(config):
    M = {}
    if cf.verbose:
        print('aquire mission data')
    with open(cf.mission_file, mode='r',encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for row in reader:
            M[row[0]] = np.genfromtxt(row[1:10], dtype='int')
    M2 = {k:v for k,v in M.items() if v[0] == 2}
    M4 = {k:v for k,v in M.items() if v[0] == 4}

    return M, M2, M4

class Sim_history():
    '''class contains trackers'''

    def __init__(self, P, M, cf):

        self.P_live = {p: {m: 0 for m in M.keys()} for p in P.keys()}
        self.P_sim = {p: {m: 0 for m in M.keys()} for p in P.keys()}
        self.FTC_live = {p: {m: 0 for m in M.keys()} for p in P.keys()}
        self.FTC_sim = {p: {m: 0 for m in M.keys()} for p in P.keys()}
        self.FTC_count = {p: 0 for p in P.keys()}
        self.recover = 0
        self.regenerate = 0
        self.FHR_realized = 0
        self.weather_cancelations = 0
        self.weather_sorties = 0
        self.sortie_cancel = 0
        self.sortie_executed = 0 
        self.sim_sortie_cancel = 0
        self.sim_sortie_executed = 0 
        self.FHR_accumulated = np.full((cf.tau), 0)
