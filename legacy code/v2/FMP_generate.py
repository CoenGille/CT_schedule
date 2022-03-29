import os
import sys
import time
import itertools

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from config import Configuration
from data import pilots, missions, QRA_availability, pilot_availability, Sim_history,\
                 aircraft, AC_availability
from datetime import datetime, timedelta
from FMP import fmp
from milp import mip
from utils import excel_writer

class Simulation():

    def __init__(self, *args, **kwargs):

        
        self.cf = Configuration() #load default data for Config
        self.period = 0           #start index for simulation 

        #initialize sim data
        self.pilots_init()
        self.missions_init()
        self.availability_init()
        self.ac_init()

        self.history = Sim_history(self.P, self.M, self.cf)
    
    def pilots_init(self):
        '''function to initialize pilot data.
        settings in config allow for data init from
        csv or randomly generated pilot data'''
        self.P, self.Pfl2, self.Pfl4 = pilots(self.cf)

    def ac_init(self):
        #TODO write init for AC from csv
        self.AC = aircraft()  

    def missions_init(self):
        '''function to initialize misson data'''
        self.M, self.M2, self.M4 = missions(self.cf)
    
    def availability_init(self):
        '''construct availability data for pilots. A 
        contains general availability and QRA defines
        what days a pilot is unavailable due to QRA duties
        A_AC '''
        self.A = pilot_availability(self.cf, self.P)
        self.QRA = QRA_availability(self.cf, self.P)
        self.A_AC = AC_availability(self.cf)

    def generate_fmp(self):
        self.svc, self.ft

    def step(self, x, x_a, ftc):
        print(self.period)
        t1 = time.perf_counter()
        sortie, idle_pilots, schedule = self.sortie_sort(x, x_a, ftc, self.cf, self.period, self.P, self.M)
        t2 = time.perf_counter() - t1
        print('\nruntime: %f sec' %t2)

        #print(self.period)
        self.period += 1

    def run(self):
        '''
        run 1 planning period of simulation
        '''
        t1_init = time.perf_counter()
        
        #init the FMP parameters for time t=0
        svc_init = dict.fromkeys(self.AC)
        rft_init = dict.fromkeys(self.AC)
        rmt_init = dict.fromkeys(self.AC)

        for n in self.AC.keys():
            svc_init[n] = self.AC[n][0]
            rft_init[n] = self.AC[n][1]
            rmt_init[n] = self.AC[n][2]

        #--------------------GENERATE FMP-------------------
        #TODO move to FMP.py?
        #due to runtime complexity FMP gen is limited to aprox 200 periods 
        #for 22 AC, cf.sp_tau is the default max subproblem length 

        svc, rft, rmt, ft = fmp(
            svc_init, rft_init, rmt_init, self.AC,
            self.A_AC, self.cf, self.cf.tau
            )
        print(svc)
        #calculate number of sorties required to achieve FMP flight hours for time t
        map = range(0, len(ft.columns))
        ft.columns = map
        FMP_sorties = np.zeros(len(ft.columns))
        svc_sorties = np.zeros(len(svc.columns))

        for t in ft.keys():
            FMP_sorties[t] = round(sum(ft[t])/1.5)
            svc_sorties[t] = sum(svc[t])


        np.savetxt('FMP_sorties.csv', FMP_sorties, delimiter=',')
        np.savetxt('svc_sorties.csv', svc_sorties, delimiter=',')

        t2_init = time.perf_counter() - t1_init
        print('\nruntime: %f sec' %t2_init)

if __name__ == '__main__':

    #initialise sim for Simulation class
    sim = Simulation()

    if sim.cf.fixed_seed:
        np.random.seed(sim.cf.seed)

    #overwrite number of iterations of  simulation
    #sim.cf.iterations = 1


    #run simulation, hold CTRL+C in terminal to terminate
    sim.run()