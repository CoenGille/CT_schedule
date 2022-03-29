import math
import sys
import time
import json
import glob
import copy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import reduce

from config import Configuration
from data import Sim_history, pilots, missions, QRA_availability, pilot_availability,\
                 aircraft, AC_availability, FMP_sortiecount, maintenance_activities,\
                 schedule_init, simulator_availability
from CT import mip
from FMP import fmp
from utils import excel_writer, date_list, dict_writer, find_largest_value, find_smallest_value

class Simulation():

    def __init__(self, *args, **kwargs):
        
        self.cf = Configuration() #load default data from config.py
    def init_run(self):
        print('initialize data and history for run')
        self.period = 0           #start index for simulation 
        #initialize sim data
        self.pilots_init()
        self.missions_init()
        self.availability_init()
        self.ac_init()
        self.maintenance_init()
        self.history = Sim_history(self.P, self.M, self.cf, self.activity_duration)

    def pilots_init(self):
        '''function to initialize pilot data.
        settings in config allow for data init from
        csv or randomly generated pilot data'''
        self.P, self.Pfl2, self.Pfl4 = pilots(self.cf)

    def ac_init(self):
        self.AC = aircraft()  

    def missions_init(self):
        '''function to initialize misson data'''
        self.M, self.M2, self.M4, self.M_night, self.A_n = missions(self.cf)
    
    def availability_init(self):
        '''construct availability data for pilots, A/C and simulators'''
        self.A, self.p_status, self.markov_P = pilot_availability(self.cf, self.P)
        self.QRA = QRA_availability(self.cf, self.P)
        self.A_AC = AC_availability(self.cf)
        self.A_SC = simulator_availability(self.cf)
    
    def maintenance_init(self):
        self.transitions, self.transition_activities, self.activity_duration = maintenance_activities()

    def save_schedule(self, *args):
        for a in args:
            a[1].to_csv(f'data/{a[0]}.csv', sep=',')

    def sortie_sort(self, X_schedule, ftc_schedule, cf, period, P, M, waves, live=True):
        '''function takes dataframes of sortie schedule on the current timestep
        to assign live missions to n waves of aproximately equal size.
        
        The Longest Processing Time (LPT) dispatching rule is applied to the 
        partition problem. In descending order of formation size the formations
        are assigned to the wave with the least number of scheduled sorties
        
        n.b. night missions are by default assigned to the last wave in the day
        if the wave is live. Simulator night missions can be flown in all waves
        '''        
        #build list of live missions
        _l = X_schedule[period] + ftc_schedule[period]
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
        for i in range(waves):
            schedule[f'wave_{i+1}'] = pd.DataFrame(columns=cols)  
        #night missions are automatically assigned to the last wave in the day 
        if len(sortie[sortie['mission'].str.contains('_n_')]) > 0 and live:
            schedule[f'wave_{len(schedule)}'] = \
                sortie[sortie['mission'].str.contains('_n_')]
            sortie = sortie[~sortie['mission'].str.contains('_n_')]
        #iteratively assign a formation to a wave using the LPT rule
        while len(sortie) > 0:
            wave = find_smallest_value(schedule)
            if len(sortie.loc[sortie['mission'] == sortie.iloc[0]['mission']]) != \
                sortie.iloc[0]['size'] and sortie.iloc[0]['size'] != 1:
                sortie = self.FL_assign(sortie, cols)
            schedule[wave] = schedule[wave].append(sortie.head(sortie.iloc[0]['size']))
            sortie = sortie.iloc[sortie.iloc[0]['size']:]
        return idle_pilots, schedule


    def FL_assign(self, sortie, cols):
        '''FL_assign is used when more than one formation of a mission is
        scheduled. all identical missions in the sortie list are sorted in 
        descending order on the FL-status. For each formation flying the mission 
        type a placeholder bin is created. The Pilots are assigned to the least
        filled bin ensuring the correct FL-status combination for each bin. Bins
        are appended and reinserted at the top of the sortie list. This corresponds
        to a least flexible job heuristic
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
                _bin = find_smallest_value(_schedule)       
                _schedule[_bin] = _schedule[_bin].append(_s.iloc[0])
                _s = _s.iloc[1:]
            new_order = pd.concat(_schedule.values())
            sortie = pd.concat([new_order, sortie])
        return sortie

    def AC_state(
                self, checks, wave, schedule, transitions, 
                transition_activities, activity_duration, 
                svc_sortie, period):
        #sim_state checks to see if state changes due to CM checks occur.
        available_for_day = int(svc_sortie[period]) #no of AC available for day
        available_for_wave = 99 #sufficiently large number
        #build sorted list of waves to enable check if wave is last
        _l = list(k for k in schedule.keys())
        _l.sort()
        _w = wave == _l[-1]
        #print(_w)
        if svc_sortie[period] <= 0: return 0
        #pre_flight checks are used to ready enough A/C for day, other checks use number of A/C in current wave
        if len(checks[checks == 'start_up']) != 0:
            day_max = find_largest_value(schedule)
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
            #waves other than the last wave of the day are presented the option to replace A/C that failed the check
            if 'landinggear_inspection' not in checks and 'engine_inspection' not in checks:
                if len(schedule[wave]) > available_for_wave and len(
                            schedule[wave]) <= svc_sortie[period] and not _w:
                    if self.cf.verbose:
                        print('prep more AC')
                    add_prep = (len(schedule[wave]) - available_for_wave)
                    if self.cf.verbose:
                        print(add_prep)
                    for i in range(add_prep):
                        state_AC = np.append(state_AC, np.random.choice(
                                    transition_activities, p=transitions[k]))
                available_for_wave = len(state_AC[state_AC == 'pass'])
            #for ech failure determine what failure occured and adjust future A/C availability accordingly
            if len(state_AC[state_AC != 'pass']) != 0:
                unique_activity = [j for j in state_AC[state_AC != 'pass']]
                for i in unique_activity:
                    sim.history.maintenance[i] += 1
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

        return available_for_wave, svc_sortie

    def prioritize(self, schedule, AC_wave, benched_pilots, p_day, live=True): #TODO refactor to multiple functions 
        '''prioritize reduces the planned schedule based on the available AC for the wave
        pilots are sorted based on the deficit of an individual pilot on a specific mission
        a second sort orders pilots in order of deficit toward the total CT schedule.
        Pilots with greater deficits are assigned prioritized over lower deficit pilots. 
        Absent pilots are replaced if possible. If sorties are cancelled the pilots with
        the lowest deficits are dropped from the schdule in the case of multiple executions.
        In case of remaining availble AC after assignemnt, a prioritized list of pilots 
        are assigned to additional BFM missions for the wave if this policy is selected
        '''
        no_sorties_planned = len(schedule)
        line_up = schedule.copy()

        prio_total = pd.Series([sum(sim.history.P_live[i].values()) \
                for i in schedule.index], index=schedule.index, dtype=int)
        prio_mission = pd.Series([sim.history.P_live[i][schedule.loc[i]['mission']] \
                for i in schedule.index], index=schedule.index, dtype=int)
        line_up = pd.concat([line_up,prio_total,prio_mission], axis=1)
        cols = ['mission', 'FL-status', 'size', 'total_sorties', 'total_mission']
        line_up.columns = cols
        
        line_up = line_up.sort_values(by=['total_mission', 'total_sorties'], ascending=True)
        _l_u = pd.DataFrame(columns=cols)
        if self.cf.verbose:
            print('try to schedule:')
            print(line_up)
        list_ = list(line_up.index)
        #-----------------------Policy for pilot absenteism replacements----------------------------------
        if live:
            for p in list_:
                if p in line_up.index:
                    mission = line_up.loc[p]['mission']
                    if p not in p_day and len(benched_pilots) != 0:
                        if self.cf.verbose:
                            print(f'pilot {p} is unvailable for the wave, find replacement')
                        if line_up.loc[p]['mission'] == 'BFM':
                            line_up = line_up.drop(index=p)
                        else:
                            line_up = line_up.drop(index=p)
                            reserve = pd.DataFrame(columns=cols)
                            for i in benched_pilots:
                                if live:
                                    row_ = pd.Series([mission, self.P[i][4], self.M[mission][0], sum(sim.history.P_live[i].values()),\
                                            sim.history.P_live[i][mission]], index=cols)
                                if not live:
                                        row_ = pd.Series([mission, self.P[i][4], self.M[mission][0], sum(sim.history.P_sim[i].values()),\
                                            sim.history.P_sim[i][mission]], index=cols)
                                index_ = list(reserve.index)
                                index_.append(i)
                                reserve = reserve.append(row_, ignore_index=True)
                                reserve.index = index_
                            if self.cf.verbose:
                                print('unscheduled pilots')
                                print(reserve)
                            if self.M[mission][0] >= 4:
                                if len(line_up[line_up['mission'].str.contains(mission)].loc[line_up['FL-status']==4]) >= 1 * \
                                        math.ceil(len(line_up[line_up['mission'].str.contains(mission)])/self.M[mission][0]) and \
                                        len(line_up[line_up['mission'].str.contains(mission)].loc[line_up['FL-status']>=2]) >= 2 * \
                                        math.ceil(len(line_up[line_up['mission'].str.contains(mission)])/self.M[mission][0]):
                                    if self.cf.verbose:
                                        print('sufficient FL-4 and FL-2 pilots remain, selecting from all available pilots')
                                elif len(line_up[line_up['mission'].str.contains(mission)].loc[line_up['FL-status']==4]) >= 1 * \
                                        math.ceil(len(line_up[line_up['mission'].str.contains(mission)])/self.M[mission][0]):
                                    if self.cf.verbose:
                                        print('sufficient FL-4 remain, insufficient FL-2 or higher pilots for second seat')
                                        print('reducing available replacements to FL-2 and higher')
                                    reserve = reserve.drop(reserve[reserve['FL-status']<2].index)
                                else:
                                    if self.cf.verbose:
                                        print('FL-4 subsitute required')
                                    reserve = reserve.drop(reserve[reserve['FL-status']<4].index)
                            if self.M[mission][0] == 2:
                                if len(line_up[line_up['mission'].str.contains(mission)].loc[line_up['FL-status']>=2]) >= 2 * \
                                        math.ceil(len(line_up[line_up['mission'].str.contains(mission)])/self.M[mission][0]):
                                    if self.cf.verbose:
                                        print('sufficient FL-2 pilots remain, selecting from all available pilots')
                                else:
                                    if self.cf.verbose:
                                        print('insufficient FL-2 or higher pilots remaining')
                                        print('reducing available replacements to FL-2 and higher')
                                    reserve = reserve.drop(reserve[reserve['FL-status']==0].index)
                            if not reserve.empty and not line_up.empty:
                                reserve = reserve.sort_values(by=['total_mission', 'total_sorties'], ascending=True)
                                if self.cf.verbose:
                                    print('candidate replacements')
                                    print(reserve)
                                    print(f'replace pilot {p} with pilot {reserve.index[0]}')
                                line_up = line_up.append(reserve.iloc[0])
                                for x in list(np.intersect1d(benched_pilots, line_up.index)):
                                    if x in benched_pilots:
                                        benched_pilots.remove(x)
                            else:
                                if self.cf.verbose:
                                    print('no viable candidates are available')
                                if len(line_up[line_up['mission'].str.contains(mission)]) <= self.M[mission][0]: 
                                    if self.cf.verbose:
                                        print(f'cancel mission {mission} due to insufficient pilots')
                                        print(line_up[line_up['mission'].str.contains(mission)])
                                    line_up = line_up[~line_up['mission'].str.contains(mission)]
                                elif len(line_up[line_up['mission'].str.contains(mission)]) >= self.M[mission][0]:
                                    if self.cf.verbose:
                                        print(f'multiple executions of mission {mission} are planned, select best sort combination')
                                        print('initial mission line up')
                                        print(line_up[line_up['mission'].str.contains(mission)])
                                    sortie_ = line_up[line_up['mission'].str.contains(mission)].sort_values(by=
                                            ['total_mission', 'total_sorties'], ascending=True)
                                    FL4 = sortie_[sortie_['FL-status']>=4].sort_values(by=
                                            ['total_mission', 'total_sorties'], ascending=True)
                                    FL2 = sortie_[sortie_['FL-status']>=2].sort_values(by=
                                            ['total_mission', 'total_sorties'], ascending=True)
                                    new_execution = pd.DataFrame(columns=cols)      
                                    for i in range(math.floor(len(sortie_)/self.M[mission][0])):
                                        for j in range(self.M[mission][0]):
                                            if self.M[mission][0]==4 and len(FL4.index) != 0:
                                                new_execution = new_execution.append(FL4.loc[FL4.index[0]])
                                                if FL4.index[0] in FL2.index:
                                                    FL2 = FL2.drop(FL4.index[0])
                                                if FL4.index[0] in sortie_.index:
                                                    sortie_ = sortie_.drop(FL4.index[0])
                                                FL4 = FL4.drop(FL4.index[0])
                                            if len(FL2.index) != 0:
                                                new_execution = new_execution.append(FL2.loc[FL2.index[0]])
                                                if FL2.index[0] in FL4.index:
                                                    FL4 = FL4.drop(FL2.index[0])
                                                if FL2.index[0] in sortie_.index:
                                                    sortie_ = sortie_.drop(FL2.index[0])
                                                FL2 = FL2.drop(FL2.index[0])
                                            if self.M[mission][0]==4 and len(sortie_.index) != 0:
                                                new_execution = new_execution.append(sortie_.loc[sortie_.index[0:2]])
                                                for k in sortie_.index[0:2]:
                                                    if k in FL4.index:
                                                        FL4 = FL4.drop(k)
                                                    if k in FL2.index:
                                                        FL2 = FL2.drop(k)
                                                sortie_ = sortie_.drop(sortie_.index[0:2])        
                                            if len(sortie_.index) != 0:
                                                new_execution = new_execution.append(sortie_.loc[sortie_.index[0]])
                                                if sortie_.index[0] in FL4.index:
                                                    FL4 = FL4.drop(sortie_.index[0])
                                                if sortie_.index[0] in FL2.index:
                                                    FL2 = FL2.drop(sortie_.index[0])
                                                sortie_ = sortie_.drop(sortie_.index[0])    
                                        if self.cf.verbose:
                                            print(f'reduced mission list for mission {mission}')
                                            print(new_execution)
                                        line_up = line_up[~line_up['mission'].str.contains(mission)]
                                        line_up = line_up.append(new_execution)
                    elif p not in p_day and len(benched_pilots) == 0: 
                        if self.cf.verbose:
                            print(f'pilot {p} is unvailable for the wave, find replacement')
                            print('no viable candidates are available')
                            print(f'cancel mission {mission} due to insufficient pilots')
                            print(line_up[line_up['mission'].str.contains(mission)])
                        line_up = line_up[~line_up['mission'].str.contains(mission)]
        #---------------------------------Policy for prioritized mission assignments---------------------------
        check = True
        if self.cf.verbose:
            if live:
                print('A/C available for wave')
                print(AC_wave)
            else:
                print('simulators available')
                print(AC_wave)
        while AC_wave > 0 and not line_up.empty and check: 
            if line_up.loc[line_up.head(1).index[0]]['size'] <= AC_wave: 
                AC_wave -= len(line_up[line_up['mission'].str.contains(
                        line_up.loc[line_up.head(1).index[0]]['mission'])])
                _l_u = _l_u.append(line_up[line_up['mission'].str.contains(
                        line_up.loc[line_up.head(1).index[0]]['mission'])])
                assigned_pilots = line_up[line_up['mission'].str.contains(
                        line_up.loc[line_up.head(1).index[0]]['mission'])].index
                line_up = line_up.drop(assigned_pilots)
            else:  
                for k in line_up.index:
                    P_not_assigned = k in line_up.index
                    if P_not_assigned:
                        if line_up.loc[k]['size'] <= AC_wave: 
                            P_not_assigned = k in line_up.index
                            AC_wave -= len(line_up[line_up['mission'].str.contains(
                                    line_up.loc[k]['mission'])])
                            _l_u = _l_u.append(line_up[line_up['mission'].str.contains(
                                    line_up.loc[k]['mission'])])
                            assigned_pilots = line_up[line_up['mission'].str.contains(
                                    line_up.loc[k]['mission'])].index
                            line_up = line_up.drop(assigned_pilots)
                #toggle exit for loop
                check = False
        #unassigned pilots from wave get assigned to the bench
        if not line_up.empty:
            if len(benched_pilots)!= 0:
                benched_pilots = benched_pilots + list(line_up.index)
            else:
                benched_pilots = list(line_up.index)
        #----------------------------policy for additional BFM sorties----------------------------
        #sort in additional BFM missions to fill in wave if pilot has BFM executions left
        if self.cf.additional_BFM:
            if AC_wave >= 1 and len(benched_pilots) != 0:
                if live:
                    BFM = pd.Series([sim.history.P_live[i]['BFM'] - self.M['BFM'][self.P[i][1]] \
                            for i in benched_pilots], index = benched_pilots, dtype=int)
                    for i in BFM.index:
                        if sim.history.P_live[i]['BFM'] >= self.M['BFM'][self.P[i][1]]:
                            BFM = BFM.drop(i)
                if not live:
                    BFM = pd.Series([sim.history.P_sim[i]['BFM'] - self.M['BFM'][self.P[i][3]] \
                            for i in benched_pilots], index = benched_pilots, dtype=int)
                    for i in BFM.index:
                        if sim.history.P_sim[i]['BFM'] >= self.M['BFM'][self.P[i][3]]:
                            BFM = BFM.drop(i)
                BFM = BFM.sort_values(ascending=True)
                while AC_wave > 0 and len(BFM) != 0:
                    add_BFM = pd.DataFrame([['BFM', self.P[BFM.index[0]][4], 1, 'NaN', 'NaN']],
                        columns=cols, index=[BFM.index[0]])
                    _l_u = _l_u.append(add_BFM, ignore_index=False)
                    BFM = BFM.drop(BFM.index[0])
                    if self.cf.verbose:
                        print('Fly additional sortie:')
                        print(add_BFM)
                    AC_wave -= 1
        if self.cf.verbose:
            print('final schedule for wave')
            print(_l_u)
            if not line_up.empty:
                if live:
                    print('sorties not assigned due to insufficient A/C')
                    print(line_up)
                else:
                    print('sorties not assigned due to insufficient simulators')
                    print(line_up)
        #write cancelled sorties to history
        if live and len(_l_u) < no_sorties_planned:
            sim.history.sortie_cancel += (no_sorties_planned - len(_l_u))
        return _l_u, benched_pilots

    def write_live_to_history(self, assigned_sorties, ftc, period):
        '''live wave is considred executed and written to pilot mission history
        FTC executions are assigned to FTC history
        '''
        if self.cf.verbose:
            print('write live sorties to history')
        for k in assigned_sorties.index.values:
            sim.history.P_live[k][assigned_sorties.loc[k]['mission']] += 1
            sim.history.P_live2[k][assigned_sorties.loc[k]['mission']] += 1
            m  = assigned_sorties.loc[k]['mission']
           # print(f'pilot {k} finished mission {m}')
            if assigned_sorties.loc[k]['mission'] == ftc[period].loc[k]:
                #print('FTC mission')
                #print(sim.history.FTC_live[k][assigned_sorties.loc[k]['mission']])
                sim.history.FTC_live[k][assigned_sorties.loc[k]['mission']] += 1
                #print(sim.history.FTC_live[k][assigned_sorties.loc[k]['mission']])
                sim.history.FTC_count[k] += 1
        sim.history.FHR_realized += len(assigned_sorties.index.values)*self.cf.ASD
        
        sim.history.sortie_executed += len(assigned_sorties.index.values)
        #print('sorties today:')
        #print(sim.history.sortie_executed)
        sim.history.FHR_accumulated[self.period] = sim.history.FHR_realized
        #print('FHR today')
        #print(sim.history.FHR_realized)

        sim.history.sortiecount[self.period] = sim.history.sortie_executed
        sim.history.sortielist[self.period] += len(assigned_sorties.index.values)

    def write_sim_to_history(self, assigned_sorties, ftcs, period):
        '''simulator wave is considred executed and written to pilot mission history
        FTC executions are assigned to FTC history
        '''
        if self.cf.verbose:
            print('write sim sorties to history')
        for k in assigned_sorties.index.values:
            sim.history.P_sim[k][assigned_sorties.loc[k]['mission']] += 1

            if assigned_sorties.loc[k]['mission'] == ftcs[period].loc[k]:
                sim.history.FTC_sim[k][assigned_sorties.loc[k]['mission']] += 1
                sim.history.FTCs_count[k] += 1
        sim.history.sim_sortie_executed += len(assigned_sorties.index.values)

    def pilot_state(self, A, QRA, period, p_status, markov):
        '''modify pilot availability state using a discrete Markov chain'''
        message = ['fell ill', 'returned to duty']
        index_ = list(A.keys())
        #duty_scheduled = pd.Series([A[k][period] for k in index_], index=index_, dtype=int)
        #has_QRA = pd.Series([QRA[k][period] for k in index_], index=index_, dtype=int)
        can_fly = pd.Series([max((A[k][period] - QRA[k][period]),0) for k in index_], index=index_, dtype=int)
        available_ = can_fly[[can_fly[i] == 1 for i in can_fly.index]].index
        for p in p_status.index:
            if np.random.random() >= markov[p_status[p]][p_status[p]]:
                p_status[p] = abs(p_status[p]-1)
                if self.cf.verbose:
                    print(p, message[p_status[p]])
            if p_status[p] == 0 and self.cf.verbose:
                print(f'{p} is ill')
        available_ = list(can_fly[[can_fly[i] == 1 and p_status[i] == 1 for i in can_fly.index]].index)
        available_ = list(np.intersect1d(list(available_),list(p_status[[p_status[p] == 1 for p in index_]].index),
                list(can_fly[[QRA[i][period] == 1 for i in index_]].index)))
        if self.cf.verbose:
            print('unavailable due to illness:')
            print(list(p_status[[p_status[p] == 0 for p in index_]].index))
            print('unavailable due to QRA:')
            print(list(can_fly[[QRA[i][period] == 1 for i in index_]].index))
            print('available for duty:')
            print(available_)
        return available_, p_status

    def step(self, x, ftc, s, ftcs):
        '''step is the primary loop that is repeated every t in the planning period. Each
        day an event list is compiled based ont he scheduled sorties. Disruptions from
        pilot absenteism and unplanned maintenance are generated. The disruptions are mitigated
        and the events are executed. Succeeding disrutions are generated after the sorties.'''
        sys.stdout.write('\r')
        sys.stdout.write('%i / %i'  %(self.period, self.cf.tau))

        if self.cf.verbose:
            print(f'\n-------------------- Simulate day {self.period+1} --------------------')
        #get events for day by constructing schedule for the day
        not_live_pilots, schedule = self.sortie_sort(
                x, ftc, self.cf, self.period, 
                self.P, self.M, self.cf.waves, live=True)
        not_sim_pilots, schedule_sim = self.sortie_sort(
                s, ftcs, self.cf, self.period, 
                self.P, self.M, self.cf.sim_waves, live=False)
        #update sim state for pilots
        p_day, self.p_status = self.pilot_state(self.A, self.QRA, self.period, self.p_status, self.markov_P)
        benched_pilots = list(reduce(np.intersect1d, (not_live_pilots.index, not_sim_pilots.index, p_day)))
        if self.cf.verbose:
            print('unscheduled pilots:')
            print(benched_pilots)
        #----------------Schedule Recovery-----------------
        copy_p_live = copy.deepcopy(sim.history.P_live)
        track = []
        for p in self.P:
            for m in self.M:
                copy_p_live[p][m] = min(sim.history.P_live[p][m], self.M[m][self.P[p][1]]) 
                track.append(copy_p_live[p][m])
        sim.history.regen.append(sum(track)/((1274/self.cf.tau)*(self.period+1)))
        
        if self.period in self.cf.rescheduling:
            #if sum(track)/((1274/self.cf.tau)*self.period) < 1:
                #sim.history.regen_period.append(self.period)
                #if self.cf.verbose:            
                #    print(f'for period {self.period} the tally is {sum(track)}, \
                #    {sum(track)/((1274/self.cf.tau)*self.period)} of the expected \
                #        {((1274/self.cf.tau)*self.period)} live missions are completed')
            print(f'------------ Periodic rescheduling on day {self.period}------------')
            x_partial, ftc_partial, s_partial, ftcs_partial, F = mip(
                    sim.history.P_live, sim.history.FTC_live, sim.history.FTC_count,
                    sim.history.P_sim, sim.history.FTC_sim, sim.history.FTCs_count, 
                    self.A, self.A_SC, self.A_n, self.cf, self.M, self.M2,
                    self.M4, self.M_night, self.P, self.period,
                    self.Pfl2, self.Pfl4, self.QRA, self.mc_AC, self.FMP_sorties,
                    x, s)
            drop_range = np.array(range(self.period, max(sim.cf.tau,x.shape[1])))
            x = x.drop(drop_range, axis=1)
            ftc = ftc.drop(drop_range, axis=1)
            s = s.drop(drop_range, axis=1)
            ftcs = ftcs.drop(drop_range, axis=1)
            x = pd.concat([x, x_partial], axis=1, ignore_index=True)
            ftc = pd.concat([ftc, ftc_partial], axis=1, ignore_index=True)
            s = pd.concat([s, s_partial], axis=1, ignore_index=True)
            ftcs = pd.concat([ftcs, ftcs_partial], axis=1, ignore_index=True)
            map = range(len(x.columns))
            x.columns = map
            ftc.columns = map
            s.columns = map
            ftcs.columns = map
        #---------------------------- LIVE MISSIONS ----------------------------
        if sum([len(schedule[x]) for x in schedule.keys()]) != 0 and self.mc_AC[self.period] != 0:
            #start day, get randomized preceding disrutions for scheduled event list for wave 1
            _sched = list(k for k in schedule.keys())
            _sched.sort()
            #-------------Update Sim state--------------
            #pre flight inspection for day
            first_wave = _sched[0]
            if self.cf.verbose:
                print('----- start_up for entire day -----')
            AC_wave, self.mc_AC = self.AC_state(
                    ['start_up'], first_wave, schedule, 
                    self.transitions, self.transition_activities,
                    self.activity_duration, self.mc_AC, self.period)
            #-------------Preceding Disruptions--------------
            #start up inspection
            for k in schedule.keys():
                weather_check = np.random.random()
                if weather_check > self.cf.weater_abort and self.mc_AC[self.period] != 0:
                    AC_wave, self.mc_AC = self.AC_state(
                            ['pre_flight'], k, schedule, self.transitions, 
                            self.transition_activities, self.activity_duration, 
                            self.mc_AC, self.period)
                    if self.cf.verbose:
                        print(f'--------------- {k} pre_flight ---------------')
                    if not schedule[k].empty:
                        #modify schedule to mitigate disruptions
                        assigned_sorties, benched_pilots = self.prioritize(schedule[k], AC_wave, 
                                benched_pilots, p_day, live=True)
                        pre_sortie_FHR = max(sim.history.FHR_accumulated)
                        self.write_live_to_history(assigned_sorties, ftc, self.period)
            #-------------Succeeding Disruptions--------------
            #post flight inspection
                        if self.mc_AC[self.period] != 0:
                            AC_wave, self.mc_AC = self.AC_state(
                                    ['post_flight'], k, schedule,
                                    self.transitions, self.transition_activities, 
                                    self.activity_duration, self.mc_AC, self.period)
                            if self.cf.verbose:
                                print(f'----- {k} post_flight -----')
            #landing gear inspections
                        if math.floor(sim.history.FHR_accumulated[self.period]/30) > \
                                    math.floor(pre_sortie_FHR/30):
                            LG_checks = math.floor(sim.history.FHR_accumulated[self.period]
                                    /30) - math.floor(pre_sortie_FHR/30)
                            if self.cf.verbose:
                                print(f'Perform landing gear inspection on {LG_checks} AC')
                            LG = pd.Series(f'LG_check {i+1}' for i in range(LG_checks))
                            if not LG.empty:                              
                                AC_wave, self.mc_AC = self.AC_state(
                                        ['landinggear_inspection'], k, LG, 
                                        self.transitions, self.transition_activities, 
                                        self.activity_duration, self.mc_AC, self.period)
            #engine check inspections
                        if math.floor(sim.history.FHR_accumulated[self.period]/50) > \
                                    math.floor(pre_sortie_FHR/50):
                            E_checks = math.floor(sim.history.FHR_accumulated[self.period]/50) - \
                                    math.floor(pre_sortie_FHR/50)
                            if self.cf.verbose:
                                print(f'Perform engine inspection on {E_checks} AC')
                            EI = pd.Series(f'EI_check {i+1}' for i in range(E_checks))
                            if not EI.empty:                              
                                AC_wave, self.mc_AC = self.AC_state(
                                        ['engine_inspection'], k, EI, 
                                        self.transitions, self.transition_activities, 
                                        self.activity_duration, self.mc_AC, self.period)
                elif weather_check <= self.cf.weater_abort:
                    if self.cf.verbose:
                        print(f'{k} cancelled due to weather')
                    sim.history.weather_cancelations += 1
                    sim.history.weather_sorties += len(schedule[k])
        #---------------------------- SIM MISSIONS ----------------------------    
        if sum([len(schedule_sim[x]) for x in schedule_sim.keys()]) != 0 and \
                    self.mc_AC[self.period] != 0:
            for k in schedule_sim.keys():
                if self.cf.verbose:
                    print(f'--------------- {k} SIM start_up ---------------')
                sim_wave = self.sim_state(self.cf, schedule_sim[k])
                if not schedule_sim[k].empty:
                    #modify schedule to mitigate disruptions
                    assigned_sorties, benched_pilots = self.prioritize(schedule_sim[k], sim_wave, 
                                    benched_pilots, p_day, live=False)
                    self.write_sim_to_history(assigned_sorties, ftcs, self.period)
        self.period += 1
        return x, ftc, s, ftcs

    def sim_state(self, cf, schedule):
        available_simulators = cf.SC_max
        for i in range(cf.SC_max):
            if np.random.random() <= cf.SC_failure:
                if cf.verbose:
                    print('simulator failed during start-up')
                available_simulators -= 1
        return available_simulators

    def run(self):
        '''
        run 1 planning period of simulation
        '''
        t1_init = time.perf_counter()

        for iteration in range(self.cf.iterations):
            print(f'\nsim run {iteration + 1}')
            self.init_run()
            if self.cf.FMP:
                print('run FMP generation')
                #init the FMP parameters for time t=0
                svc_init = dict.fromkeys(self.AC)
                rft_init = dict.fromkeys(self.AC)
                rmt_init = dict.fromkeys(self.AC)

                for n in self.AC.keys():
                    svc_init[n] = self.AC[n][0]
                    rft_init[n] = self.AC[n][1]
                    rmt_init[n] = self.AC[n][2]

                #--------------------GENERATE FMP-------------------
                svc, self.ft = fmp(
                    svc_init, rft_init, rmt_init, self.AC,
                    self.A_AC, self.cf, self.cf.tau
                    )
                #calculate number of sorties required to achieve FMP flight hours for time t
                map = range(0, len(self.ft.columns))
                self.ft.columns = map
                FMP_sorties = np.zeros(len(self.ft.columns))
                svc_sorties = np.zeros(len(svc.columns))

                for t in self.ft.keys():
                    FMP_sorties[t] = round(sum(self.ft[t])/1.5)
                    svc_sorties[t] = sum(svc[t])

                np.savetxt(self.cf.FMP_file, FMP_sorties, delimiter=',')
                np.savetxt(self.cf.svc_file, svc_sorties, delimiter=',')
                self.save_schedule(['FT', self.ft])

            self.ft, self.FMP_sorties, self.mc_AC = FMP_sortiecount(self.cf)
            print('FH assigned by FMP')
            print(self.ft.values.sum())

            if self.cf.gen_first_schedule:
                print('run intial schedule generation')
            #--------------------GENERATE INITIAL SORTIE SCHEDULE-------------------
                x, ftc, s, ftcs, F = mip(
                        sim.history.P_live, sim.history.FTC_live, sim.history.FTC_count,
                        sim.history.P_sim, sim.history.FTC_sim, sim.history.FTCs_count, 
                        self.A, self.A_SC, self.A_n, self.cf, self.M, self.M2,
                        self.M4, self.M_night, self.P, self.period,
                        self.Pfl2, self.Pfl4, self.QRA, self.mc_AC, self.FMP_sorties
                        )
                self.save_schedule(['x', x], ['ftc', ftc], ['s', s], ['ftcs', ftcs])
                print('intial schedule saved to data folder')
            else:
                print('import schedule from data')
                x, ftc, s, ftcs = schedule_init()

            time.sleep(5)
            while self.period < self.cf.tau:
                try:
                    x, ftc, s, ftcs = self.step(x, ftc, s, ftcs)
                except KeyboardInterrupt:
                    print('\nTerminate simulation run')
                    sys.exit(1)
            t2_init = time.perf_counter() - t1_init
            for t in range(self.cf.tau):
                if sim.history.FHR_accumulated[t] == 0:
                    sim.history.FHR_accumulated[t] = sim.history.FHR_accumulated[t-1]
            FHR_from_FMP = np.full((self.cf.tau), 0, dtype=np.float64)
            FHR_from_FMP[0] = self.ft[0].values.sum()
            for t in range(1, self.cf.tau):
                FHR_from_FMP[t] = self.ft[t].values.sum()+FHR_from_FMP[t-1]
            print('\nruntime: %f sec' %t2_init)
            print('normal data var')
            print(sum([sum(sim.history.P_live[k].values()) for k in sim.history.P_live.keys()]))
            print('isolated data var')
            print(sum([sum(sim.history.P_live2[k].values()) for k in sim.history.P_live2.keys()]))
            for p in self.P:
                for m in self.M:
                    sim.history.P_live[p][m] = int(sim.history.P_live[p][m])
                    sim.history.P_sim[p][m] = int(sim.history.P_sim[p][m])
            data_run = dict_writer(
                ['FHR_realized', sim.history.FHR_realized],
                ['waves_weather_cancel', sim.history.weather_cancelations],
                ['sortie_weather_cancel', sim.history.weather_sorties],
                ['insuf_AC_cancel', sim.history.sortie_cancel],
                ['sortie_executed', sim.history.sortie_executed],
                ['sim_sortie_cancel', sim.history.sim_sortie_cancel],
                ['sim_sortie_executed', sim.history.sim_sortie_executed],
                ['FHR_cummulative', sim.history.FHR_accumulated.tolist()],
                ['FT_FMP', FHR_from_FMP.tolist()],
                ['corrective_MX', sim.history.maintenance],
                ['FTC_count', sim.history.FTC_count],
                ['regen', sim.history.regen],
                ['regen_period', sim.history.regen_period],
                ['runtime_simulation', t2_init]
            )
            with open(f'results/run_{iteration}_CTresults_live.txt', 'w', encoding='utf-8') as file:
                json.dump(sim.history.P_live, file, ensure_ascii=False, indent=2) 
            with open(f'results/run_{iteration}_CTresults_sim.txt', 'w', encoding='utf-8') as file:
                json.dump(sim.history.P_sim, file, ensure_ascii=False, indent=2) 
            with open(f'results/run_{iteration}_data.txt', 'w', encoding='utf-8') as file:
                json.dump(data_run, file, ensure_ascii=False, indent=2) 

        if self.cf.print_schedule: 
            dates = date_list(self.cf.startdate, self.cf.tau)
            excel_writer(
                dates, 
                'results/schedule.xlsx',
                ['live', x],
                ['sim', s],
                ['FTC live', ftc],
                ['FTC sim', ftcs]
                )

        print(sim.history.P_live)
        print('realized availability per day')
        print(self.mc_AC[0:self.cf.tau])
        print('FH assigned by FMP')
        print(self.ft.values.sum())
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
        print('maintenance activities over period:')
        for i in sim.history.maintenance.keys():
            print(f'{i} performed {sim.history.maintenance[i]} times')

        for p in self.P:
            for m in self.M:
                sim.history.P_live[p][m] = int(sim.history.P_live[p][m])
        #with open('hist_run.txt', 'w', encoding='utf-8') as file:
           # json.dump(sim.history.P_live, file, ensure_ascii=False, indent=2)
        for p in self.P:
            for m in self.M:
                sim.history.P_live[p][m] -= self.M[m][self.P[p][0]]
        CT_complete = dict.fromkeys(self.P)
        #print(sim.history.P_live)
        for p in self.P:
            CT_complete[p] = 1- ((-1*(sum(sim.history.P_live[p][m] for m in self.M))) 
                    / (sum(self.M[m][self.P[p][0]] for m in self.M)))
        print('minimum CT completion ratio per pilot in relative flight hours')
        print(CT_complete)
        print('average minimum CT completion ratio for SQN')
        avg_completion = sum(CT_complete.values())/len(CT_complete)
        print(avg_completion)
        t2_init = time.perf_counter() - t1_init
        print('\nruntime: %f sec' %t2_init)
        print('sorties per day')
        print(sim.history.sortielist)
        print('cummulative sorties')
        print(sim.history.sortiecount)

        if self.cf.plot:
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