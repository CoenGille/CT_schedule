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

        self.history = Sim_history(self.P, self.M)
    
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
    
    def date_list(self, start_date, i):
        date_generator = (((datetime.strptime(start_date,"%d/%m/%Y") + \
                          timedelta(days=i)).date()).strftime('%d/%m/%Y') \
                          for i in itertools.count()
        )
        j = itertools.islice(date_generator,i)
        k = pd.Series(list(j))
        return k

    def generate_fmp(self):
        self.svc, self.ft

    def sortie_sort(self, x, x_a, ftc, cf, period, P, M):
        '''function takes dataframes of sortie schedule on the current timestep
        to assign live missions to n waves of aproximately equal size.
        
        The Longest Processing Time (LPT) dispatching rule is applied to the 
        partition problem. In descending order of formation size the formations
        are assigned to the wave with the least number of scheduled sorties
        
        n.b. night missions are by default assigned to the last wave in the day
        '''
        #build list of live missions
        _l = x[period] + x_a[period] + ftc[period]
        FL = pd.Series([P[x][4] for x in _l.index], index=_l.index)
        _l = pd.concat([_l, FL], axis=1)
        _l.columns = range(_l.shape[1])
        
        #Separate live missions from idle pilots 
        sortie = _l[_l[0]!='']
        idle_pilots = _l[_l[0]=='']
        
        #retrieve mission characteristics
        MS = pd.Series([M[x][0] for x in sortie[0]], index=sortie.index)
        sortie = pd.concat([sortie, MS], axis=1)
        cols = ['mission', 'FL-status', 'size']
        sortie.columns = cols
    
        #LPT heuristic
        #sort live missions on formation size first and mission type second
        #construct daily waves, number of waves can be set in Configuration.py 
        sortie = sortie.sort_values(by=['size', 'mission'], ascending=False)
        schedule = {}
        for i in range(cf.waves):
            schedule[f'wave_{i+1}'] = pd.DataFrame(columns=cols)  
        
        #night missions are automatically assigned to the last wave in the day 
        if len(sortie[sortie['mission'].str.contains('_n_')]) > 0:
            schedule[f'wave_{len(schedule)}'] = \
                sortie[sortie['mission'].str.contains('_n_')]
            sortie = sortie[~sortie['mission'].str.contains('_n_')]
            
        #iteratively assign a formation to a wave using the LPT rule
        while len(sortie) > 0:
            wave = self.find_smallest_value(schedule)
            if len(sortie.loc[sortie['mission'] == sortie.iloc[0]['mission']]) != \
                sortie.iloc[0]['size'] and sortie.iloc[0]['size'] != 1:
                
                sortie = self.FL_assign(sortie, cols)
            schedule[wave] = schedule[wave].append(sortie.head(sortie.iloc[0]['size']))
            sortie = sortie.iloc[sortie.iloc[0]['size']:]
        return sortie, idle_pilots, schedule

    def find_smallest_value(self, ds):
        _s = 1e9  #sufficiently large number
        wave = ''
        for k, v in ds.items():
            size = len(v)
            if size <= _s:
                _s = size
                wave = k
        return wave

    def FL_assign(self, sortie, cols):
        '''FL_assign is used when more than one formation of a mission is
        scheduled. all identical missions in the sortie list are sorted in 
        descending order on the FL-status. For each formation of the mission 
        type a placeholder bin is created. The Pilots are assigned to the least
        filled bin ensuring the correct FL-status combination for each bin. Bins
        are appended and reinserted at the top of the sortie list.
        '''
        _s = sortie[sortie['mission'].str.contains(sortie.iloc[0]['mission'])]
        _s = _s.sort_values(by=['FL-status'], ascending=False)

        i = int(len(_s)/sortie.iloc[0]['size'])

        
        sortie = sortie.iloc[len(_s):]
        _schedule = {}
        for j in range(i):
            _schedule[f'bin_{j}'] = pd.DataFrame(columns=cols)

        while len(_s) > 0:
            _bin = self.find_smallest_value(_schedule)       
            _schedule[_bin] = _schedule[_bin].append(_s.iloc[0])
            _s = _s.iloc[1:]

        new_order = pd.concat(_schedule.values())
        sortie = pd.concat([new_order, sortie])
        return sortie

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
        if self.cf.tau > self.cf.sp_tau:

            #first run FMP
            A_AC_partial = np.delete(self.A_AC, slice(self.cf.sp_tau, self.cf.tau), 0)
            svc,rft,rmt,ft = fmp(svc_init, rft_init, rmt_init, self.AC, A_AC_partial,
                                 self.cf, self.cf.sp_tau
                                 )
            #second run FMP
            A_AC_partial = np.delete(self.A_AC, slice(0, (self.cf.tau-self.cf.sp_tau)), 0)

            for n in self.AC.keys():
                svc_init[n] = svc.loc[n][(self.cf.tau-self.cf.sp_tau)]
                rft_init[n] = rft.loc[n][(self.cf.tau-self.cf.sp_tau)]
                rmt_init[n] = rmt.loc[n][(self.cf.tau-self.cf.sp_tau)]

            svc2, rft2, rmt2, ft2 = fmp(svc_init, rft_init, rmt_init, self.AC,
                                        A_AC_partial, self.cf, self.cf.sp_tau
                                        )    

            #trim first run results !NB first run generates more periods
            #due to model behaviour, FMP underschedules phase maint. towards
            #the planning horizon end. generating and trimming excess periods
            #fixes this behaviour.
            drop_range = np.array(range((self.cf.tau-self.cf.sp_tau),self.cf.sp_tau))
            svc = svc.drop(drop_range, axis=1)
            rmt = rmt.drop(drop_range, axis=1)
            rft = rft.drop(drop_range, axis=1)
            ft = ft.drop(drop_range, axis=1)

            cols = range(self.cf.tau)
            #merge FMP run subproblems
            svc_tot = pd.concat([svc, svc2], axis=1, ignore_index=True) 
            rmt_tot = pd.concat([rmt, rmt2], axis=1, ignore_index=True)
            rft_tot = pd.concat([rft, rft2], axis=1, ignore_index=True) 
            ft_tot = pd.concat([ft, ft2], axis=1, ignore_index=True) 

        #if the planning horizon is less than the subproblem length go nuts and
        #plan everything in one go
        else: 
            svc_tot, rft_tot, rmt_tot, ft_tot = fmp(
                svc_init, rft_init, rmt_init, self.AC,
                self.A_AC, self.cf, self.cf.tau
                )

        #calculate number of sorties required to achieve FMP flight hours for time t
        map = range(0, len(ft_tot.columns))
        ft_tot.columns = map
        FMP_sorties = np.zeros(len(ft_tot.columns))

        for t in ft_tot.keys():
            FMP_sorties[t] = round(sum(ft_tot[t])/1.5)

        #generate the initial sortie schedule for the Sim
        x, x_a, ftc, s, s_a, ftcs, O, F = mip(
            self.A, self.cf, self.M, self.M2,
            self.M4, self.P, self.period,
            self.Pfl2, self.Pfl4, self.QRA, FMP_sorties
            )
        print(rft_tot[0])
        #write sortie solution to sim history
        for i in range(len(x)):
            for k in x.keys():
                if x.loc[x.index[i]][k] != '':
                    self.history.P_live[x.index[i]][x.loc[x.index[i]][k]] += 1
        #print(self.history.P_live)

        #print schedule for preventive maint. and sortie planning to xlsx
        if self.cf.print_schedule: 
            dates = self.date_list(self.cf.startdate, self.cf.tau)
            excel_writer(
                dates, 
                'Fighter_schedule.xlsx',
                ['live', x],
                ['sim', s],
                ['FTC live', ftc],
                ['FTC sim', ftcs], 
                ['live out YTP', x_a],
                ['sim out YTP', s_a],
                ['office', O],
                ['formations', F],
                ['AC fight time', ft_tot],
                ['servicable', svc_tot],
                ['rem. flight time', rft_tot],
                ['rem. maintenace time', rmt_tot]
                )

        t2_init = time.perf_counter() - t1_init
        print('\nruntime: %f sec' %t2_init)

        while self.period < self.cf.tau:
            try:
                self.step(x, x_a, ftc)
            except KeyboardInterrupt:
                print('\nTerminate simulation run')
                sys.exit(1)

if __name__ == '__main__':

    #initialise sim for Simulation class
    sim = Simulation()

    if sim.cf.fixed_seed:
        np.random.seed(sim.cf.seed)

    #overwrite number of iterations of  simulation
    #sim.cf.iterations = 1


    #run simulation, hold CTRL+C in terminal to terminate
    sim.run()