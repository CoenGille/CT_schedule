import math
import os
import sys
import time
import itertools
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from config import Configuration
from data import pilots, missions, QRA_availability, pilot_availability, Sim_history,\
                 aircraft, AC_availability, FMP_sortiecount, maintenance_activities
from datetime import datetime, timedelta
from FMP import fmp
from FP import mip
from utils import excel_writer, date_list

class Simulation():

    def __init__(self, *args, **kwargs):

        
        self.cf = Configuration() #load default data for Config
        self.period = 0           #start index for simulation 

        #initialize sim data
        self.pilots_init()
        self.missions_init()
        self.availability_init()
        self.ac_init()
        self.maintenance_init()

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
        self.FMP_sorties, self.svc = FMP_sortiecount(self.cf)
    
    def maintenance_init(self):
        self.transitions, self.transition_activities, self.activity_duration = maintenance_activities()


    #def generate_fmp(self): #TODO find out why this is here?
    #    self.svc, self.ft

    def sortie_sort(self, x, x_a, ftc, cf, period, P, M): #TODO refactor live vars (x, x_a and ftc)
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
        MS = pd.Series([M[x][0] for x in sortie[0]], index=sortie.index, dtype=int)
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
        return idle_pilots, schedule

    def find_largest_value(self, ds):
        _s = 1e-9 #sufficiently small number
        key = ''
        for k, v in ds.items():
            size = len(v)
            if size >= _s:
                _s = size
                key = k
        return key

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
        if i > 1:
            sortie = sortie.iloc[len(_s):]
            _schedule = {}
            for j in range(i):
                _schedule[f'bin_{j}'] = pd.DataFrame(columns=cols)

            while len(_s) > 0:
                if self.cf.verbose:
                    print(_schedule)
                _bin = self.find_smallest_value(_schedule)       
                _schedule[_bin] = _schedule[_bin].append(_s.iloc[0])
                _s = _s.iloc[1:]

            new_order = pd.concat(_schedule.values())
            sortie = pd.concat([new_order, sortie])
        return sortie

    def sim_state(self, checks, wave, schedule, transitions, transition_activities, activity_duration, svc_sortie, period):
        #sim_state checks to see if state changes due to CM checks occur.
        available_for_day = int(svc_sortie[period]) #no of AC available for day
        available_for_wave = 99 #sufficiently large number
        #print(schedule[wave])
        
        #build sorted list of waves to enable check if wave is last
        _l = list(k for k in schedule.keys())
        _l.sort()
        
        _w = wave == _l[-1]
        #print(_w)
        
        if svc_sortie[period] <= 0: return 0

        #pre_flight checks are used to ready enough AC for day, other checks use no of AC in current wave
        if len(checks[checks == 'pre_flight']) != 0:
            day_max = self.find_largest_value(schedule)
            _c = min(len(schedule[day_max]), available_for_day)
        else:
            _c = min(len(schedule[wave]), available_for_day) 

        #perform checks on all AC in current wave or for day 
        for k in checks:
            
            state_AC = []
            for i in range(_c):
                state_AC.append(np.random.choice(transition_activities, p=transitions[k]))
            state_AC = np.asarray(state_AC)
            available_for_wave = min(len(state_AC[state_AC == 'pass']), available_for_wave)
            #print(state_AC)
            
            if 'landinggear_inspection' not in checks and 'engine_inspection' not in checks:
                if len(schedule[wave]) > available_for_wave and len(schedule[wave]) <= svc_sortie[period]\
                        and not _w:
                    if self.cf.verbose:
                        print('prep more AC')

                    add_prep = (len(schedule[wave]) - available_for_wave)
                    if self.cf.verbose:
                        print(add_prep)
                    for i in range(add_prep):
                        state_AC = np.append(state_AC, np.random.choice(transition_activities, p=transitions[k]))
                    #print(state_AC)
                available_for_wave = len(state_AC[state_AC == 'pass'])
            
            if len(state_AC[state_AC != 'pass']) != 0:
                unique_activity = [j for j in state_AC[state_AC != 'pass']]
                for i in unique_activity:
                    if self.cf.verbose:
                        print(f'{i} maintenance required after {k} inspection')
                    if _w and i != '1_small':
                        _n = math.ceil(activity_duration[i][0]/8)
                        for _o in range(1,_n+1):
                            if period+_o < len(svc_sortie) and svc_sortie[period+_o]>=1:
                                svc_sortie[period+_o] -= 1
                                if self.cf.verbose:
                                    print(f'AC availability changes in day {_o} from current period')
                    elif not _w and i != '1_small':
                        _n = math.ceil(activity_duration[i][0]/8)
                        for _o in range(_n):
                            if period+_o < len(svc_sortie) and svc_sortie[period+_o]>=1:
                                svc_sortie[period+_o] -= 1
                                if self.cf.verbose:
                                    print(f'AC availability changes in day {_o} from current period')
            available_for_day = int(svc_sortie[period])        

            _c = available_for_wave

        return available_for_wave, available_for_day, svc_sortie

    def prioritize(self, schedule, AC_wave):
        no_sorties_planned = len(schedule)
        line_up = schedule.copy()

        prio_total = pd.Series([sum(sim.history.P_live[i].values()) for i in schedule.index], index=schedule.index, dtype=int)
        prio_mission = pd.Series([sim.history.P_live[i][schedule.loc[i]['mission']] for i in schedule.index], index=schedule.index, dtype=int)
        line_up = pd.concat([line_up,prio_total,prio_mission], axis=1)

        cols = ['mission', 'FL-status', 'size', 'total_sorties', 'total_mission']
        line_up.columns = cols
        
        line_up = line_up.sort_values(by=['total_sorties', 'total_mission'], ascending=True)
        _l_u = pd.DataFrame(columns=cols)
        check = True
        while AC_wave > 0 and len(line_up) and check > 0: 
            if line_up.loc[line_up.head(1).index[0]]['size'] <= AC_wave:
                AC_wave -= len(line_up[line_up['mission'].str.contains(line_up.loc[line_up.head(1).index[0]]['mission'])])
                _l_u = _l_u.append(line_up[line_up['mission'].str.contains(line_up.loc[line_up.head(1).index[0]]['mission'])])
                assigned_pilots = line_up[line_up['mission'].str.contains(line_up.loc[line_up.head(1).index[0]]['mission'])].index
                line_up = line_up.drop(assigned_pilots)

            else:
                
                k = list(line_up)
                for k in line_up.index:
                    P_not_assigned = k in line_up.index
                    #print(P_not_assigned)
                    if P_not_assigned:
                        if line_up.loc[k]['size'] <= AC_wave: 
                            P_not_assigned = k in line_up.index
                            AC_wave -= len(line_up[line_up['mission'].str.contains(line_up.loc[k]['mission'])])
                            _l_u = _l_u.append(line_up[line_up['mission'].str.contains(line_up.loc[k]['mission'])])
                            assigned_pilots = line_up[line_up['mission'].str.contains(line_up.loc[k]['mission'])].index
                            line_up = line_up.drop(assigned_pilots)

                check = False

        sim.history.sortie_cancel += (no_sorties_planned - len(_l_u))
        return _l_u

    def write_live_to_history(self, assigned_sorties): #TODO add FTC counts
        if self.cf.verbose:
            print('write live sorties to history')
        for k in assigned_sorties.index.values:
            sim.history.P_live[k][assigned_sorties.loc[k]['mission']] += 1
        sim.history.FHR_realized += len(assigned_sorties.index.values)*self.cf.ASD
        sim.history.sortie_executed += len(assigned_sorties)
        sim.history.FHR_accumulated[self.period] = sim.history.FHR_realized

    def write_sim_to_history(self, assigned_sorties):
        if self.cf.verbose:
            print('write sim sorties to history')
        for k in assigned_sorties.index.values:
            sim.history.P_sim[k][assigned_sorties.loc[k]['mission']] += 1
        sim.history.sim_sortie_executed += len(assigned_sorties)

    def step(self, x, x_a, ftc, s, s_a, ftcs):
        sys.stdout.write('\r')
        sys.stdout.write('%i / %i'  %(self.period, self.cf.tau))

        #get events for day by constructing schedule for the day
        not_live_pilots, schedule = self.sortie_sort(x, x_a, ftc, self.cf, self.period, self.P, self.M)
        not_sim_pilots, schedule_sim = self.sortie_sort(s, s_a, ftcs, self.cf, self.period, self.P, self.M)

        if self.cf.verbose:
            print(f'-------------------- Simulate day {self.period+1} --------------------')

        #----------------Schedule Recovery-----------------
        if self.period in self.cf.rescheduling:
            if self.cf.verbose:
                print('------------ Reschedule Periodic ------------')
        
            x_partial, x_a_partial, ftc_partial, s_partial, s_a_partial, ftcs_partial, O, F = mip(
                    sim.history.P_live, sim.history.FTC_live, sim.history.FTC_count, sim.history.P_sim, 
                    self.A, self.cf, self.M, self.M2,
                    self.M4, self.P, self.period,
                    self.Pfl2, self.Pfl4, self.QRA, self.FMP_sorties
                    )
            drop_range = np.array(range(self.period, min(sim.cf.tau,x.shape[1])))

            x = x.drop(drop_range, axis=1)
            x_a = x_a.drop(drop_range, axis=1)
            ftc = ftc.drop(drop_range, axis=1)
            s = s.drop(drop_range, axis=1)
            s_a = s_a.drop(drop_range, axis=1)
            ftcs = ftcs.drop(drop_range, axis=1)

            x = pd.concat([x, x_partial], axis=1, ignore_index=True)
            x_a = pd.concat([x_a, x_a_partial], axis=1, ignore_index=True)
            ftc = pd.concat([ftc, ftc_partial], axis=1, ignore_index=True)
            s = pd.concat([s, s_partial], axis=1, ignore_index=True)
            s_a = pd.concat([s_a, s_a_partial], axis=1, ignore_index=True)
            ftcs = pd.concat([ftcs, ftcs_partial], axis=1, ignore_index=True)

            map = range(len(x.columns))
            x.columns = map
            x_a.columns = map
            ftc.columns = map
            s.columns = map
            s_a.columns = map
            ftcs.columns = map

        if sum([len(schedule[x]) for x in schedule.keys()]) != 0 and self.svc[self.period] != 0:
            #start day, get randomized preceding disrutions for scheduled event list for wave 1
            _sched = list(k for k in schedule.keys())
            _sched.sort()
            
            #-------------Update Sim state--------------
            #pre flight inspection for day
            first_wave = _sched[0]
            if self.cf.verbose:
                print('----- pre_flight for entire day -----')
            AC_wave, AC_day, self.svc = self.sim_state(['pre_flight'], first_wave, schedule, \
                self.transitions, self.transition_activities, self.activity_duration, self.svc, self.period)
            #print(AC_wave)

            #TODO sim state update for pilots (absenteism)

            #-------------Preceding Disruptions--------------
            #start up inspection
            for k in schedule.keys(): #TODO use AC_day input instead of svc[period]
                if np.random.random() > self.cf.weater_abort and self.svc[self.period] != 0:
                    AC_wave, AC_day, self.svc = self.sim_state(['start_up'], k, schedule, \
                        self.transitions, self.transition_activities, self.activity_duration, self.svc, self.period)
                    if self.cf.verbose:
                        print(f'----- {k} start_up -----')

                    assigned_sorties = self.prioritize(schedule[k], AC_wave)
                    pre_sortie_FHR = max(sim.history.FHR_accumulated)

                    self.write_live_to_history(assigned_sorties)
                    self.write_sim_to_history(schedule_sim[self.period])
            


            #-------------Succeeding Disruptions--------------
            #post flight inspection
                    if self.svc[self.period] != 0:
                        AC_wave, AC_day, self.svc = self.sim_state(['post_flight'], k, schedule, \
                            self.transitions, self.transition_activities, self.activity_duration, self.svc, self.period)
                        if self.cf.verbose:
                            print(f'----- {k} post_flight -----')

            #landing gear inspections
                    if math.floor(sim.history.FHR_accumulated[self.period]/30) > math.floor(pre_sortie_FHR/30):
                        LG_checks = math.floor(sim.history.FHR_accumulated[self.period]/30) -  math.floor(pre_sortie_FHR/30)
                        if self.cf.verbose:
                            print(f'Perform landing gear inspection on {LG_checks} AC')
                        LG = pd.Series(f'LG_check {i+1}' for i in range(LG_checks))
                        AC_wave, AC_day, self.svc = self.sim_state(['landinggear_inspection'], k, LG, \
                            self.transitions, self.transition_activities, self.activity_duration, self.svc, self.period)
            #engine check inspections
                    if math.floor(sim.history.FHR_accumulated[self.period]/50) > math.floor(pre_sortie_FHR/50):
                        E_checks = math.floor(sim.history.FHR_accumulated[self.period]/50) -  math.floor(pre_sortie_FHR/50)
                        if self.cf.verbose:
                            print(f'Perform engine inspection on {E_checks} AC')
                        EI = pd.Series(f'EI_check {i+1}' for i in range(E_checks))
                        AC_wave, AC_day, self.svc = self.sim_state(['engine_inspection'], k, EI, \
                            self.transitions, self.transition_activities, self.activity_duration, self.svc, self.period)


                else:
                    if self.cf.verbose:
                        print(f'{k} cancelled due to weather')
                    sim.history.weather_cancelations += 1
                    sim.history.weather_sorties += len(schedule[k])
        #print(self.period)
        self.period += 1
        return x, x_a, ftc, s, s_a, ftcs




    def run(self):
        '''
        run 1 planning period of simulation
        '''
        t1_init = time.perf_counter()
        print('run intial schedule generation')
        #generate the initial sortie schedule for the Sim
        x, x_a, ftc, s, s_a, ftcs, O, F = mip(
                    sim.history.P_live, sim.history.FTC_live, sim.history.FTC_count, sim.history.P_sim, 
                    self.A, self.cf, self.M, self.M2,
                    self.M4, self.P, self.period,
                    self.Pfl2, self.Pfl4, self.QRA, self.FMP_sorties
                    )
        
        def save_schdule(*args):
            for a in args:
                a[1].to_csv(f'{a[0]}.csv', sep=',')

        save_schdule(['x', x], ['x_a', x_a], ['ftc', ftc], ['s', s], ['s_a', s_a], ['ftcs', ftcs])

        #write sortie solution to sim history
        #for i in range(len(x)):
        #    for k in x.keys():
        #        if x.loc[x.index[i]][k] != '':
        #            self.history.P_live[x.index[i]][x.loc[x.index[i]][k]] += 1
        #print(self.history.P_live)

        #print schedule for preventive maint. and sortie planning to xlsx


        if self.cf.print_schedule: 
            dates = date_list(self.cf.startdate, self.cf.tau)
            excel_writer(
                dates, 
                'Fighter_schedule.xlsx',
                ['live', x],
                ['sim', s],
                ['FTC live', ftc],
                ['FTC sim', ftcs], 
                ['live out YTP', x_a],
                ['sim out YTP', s_a]
                )

        #print(json.dumps(sim.history.P_live, sort_keys=True, indent=4))  
        print('realized availability per day')
        print(self.svc[0:self.cf.tau])
        print('actualized FHR for total period for CT')
        print(sim.history.FHR_realized)
        print('weather cancels')
        print(sim.history.weather_cancelations)
        print('sorties cancelled due to weather')
        print(sim.history.weather_sorties)


        print('number of sorties cancelled due to insufficient AC:')
        print(sim.history.sortie_cancel)
        print('number of sorties executed over period:')
        print(sim.history.sortie_executed)

        for p in self.P:
            for m in self.M:
                sim.history.P_live[p][m] -= self.M[m][self.P[p][1]]
        CT_complete = dict.fromkeys(self.P)

        for p in self.P:
            CT_complete[p] = 1- ((-1*(sum(sim.history.P_live[p][m] for m in self.M))) / (sum(self.M[m][self.P[p][1]] for m in self.M)))

        print('desired CT completion ratio per pilot')
        print(CT_complete)

        print('average desired CT completion for SQN')
        avg_completion = sum(CT_complete.values())/len(CT_complete)
        print(avg_completion)

        t2_init = time.perf_counter() - t1_init
        print('\nruntime: %f sec' %t2_init)

        for t in range(self.cf.tau):
            if sim.history.FHR_accumulated[t] == 0:
                sim.history.FHR_accumulated[t] = sim.history.FHR_accumulated[t-1]
        
        FHR_from_FMP = np.copy(self.FMP_sorties[0:self.cf.tau])*self.cf.ASD
        for t in range(1, self.cf.tau):
            FHR_from_FMP[t] = FHR_from_FMP[t]+FHR_from_FMP[t-1]

        plt.plot(sim.history.FHR_accumulated, c='r', label='FHR realized')
        plt.plot(FHR_from_FMP, c='g', label='FHR provided by FMP')
        plt.title('Flight hours realized')
        plt.legend()
        plt.show()

if __name__ == '__main__':

    #initialise sim for Simulation class
    sim = Simulation()

    if sim.cf.fixed_seed:
        np.random.seed(sim.cf.seed)

    #overwrite number of iterations of  simulation
    #sim.cf.iterations = 1


    #run simulation, hold CTRL+C in terminal to terminate
    sim.run()