'''
File contains all simulation presets and settings
'''

import numpy as np
import pulp as pl
import multiprocessing

class Configuration():
    def __init__(self, *args, **kwargs):
        #--------------------SIM_SETTINGS------------------------
        self.verbose = False         #toggle to enable print to console for simulation events
        self.print_schedule = True #toggle wheter the sortie schedule is saved to xls after the run
        self.gen_first_schedule = False   #generate schedule at t=0 | False imports initial schedule from data folder
        self.FMP = False             #generate FMP solution on first run | False imports FMP from data folder
        self.plot = False            #toggle FHR print function for last run in simulation set 
        self.fixed_seed = False     #toggle to fix seed for RNG
        self.seed = 69              #seed for RNG init in numpy, to fix RNG for academic and testing purposes
        self.iterations = 30         #number of iterations in simulation 

        #solver is used to modify the pulp LP solver setting in the milp.py file
        #for academic purposes a free licence for gurobi can be obtained at
        #the following adress:
        #https://www.gurobi.com/academia/academic-program-and-licenses/
        
        self.solver = pl.GUROBI_CMD(options=[('MIPFocus', 3), ('NumericFocus', 1), ('MIPGapAbs', 2),('MIPGap',0.05)]) 
        #options=[('MIPGap',0.01), ('Threads', 20)], ('TimeLimit', 500), ('TimeLimit', 600), ('MIPFocus', 3), ('NumericFocus', 1), ('MIPGapAbs', 2), ('OutputFlag', 0)
        #
        #alternative for CPLEX solver
        #self.solver = pl.CPLEX_CMD(options=['set mip tolerances mipgap 0.01']) 
        
        #alternative for the native CBC (COIN-BC) solver in pulp 
        #self.solver = pl.PULP_CBC_CMD(threads=multiprocessing.cpu_count()-1,gapRel = 0.01, presolve=None, mip_start=False, timeLimit=60)   
        #--------------------POLICIES------------------------
        self.additional_BFM = True  #use available unassigned A/C for extra BFM scheduling, if pilots are behind on BFM count 
        self.free_FL4 = True        #leave at least 1 FL4 pilot unassigned each day
        #reschedule [59, 120, 181, 243, 304, 330, 344, 351, 358][90, 151, 181, 212, 243, 273, 304, 337, 344, 351, 358][90, 181, 273, 304, 337, 344, 351, 358][31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 337, 344, 351, 358]
        # KFI1 monthly - [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 337, 344, 351, 358]
        # KFI1 triaal HF close / dynamic [120, 240, 304, 337, 344, 351, 358]
        # KFI1 triaal LF close [120, 240, 304, 334]
        # KFI1 dynamic [120, 239, 246, 253, 260, 267, 274, 281, 288, 295, 302, 309, 316, 323, 330, 337, 344, 351, 358]
        self.rescheduling = [120, 240, 304, 337, 344, 351, 358]
        self.sortie_correct = 1.48   #overschedule rate 

        #--------------------VARIABLES------------------------
        self.tau = 365              #set lenght of simulation period in days 
        self.waves = 2              #number of live waves per day 
        self.sim_waves = 3          #number of sim waves per day
        self.weater_abort = .1      #chance of weather abort for entire wave 
        
        self.ASD = 1.5              #average sortie duration
        self.percentage_AC_mc = .7  #net percentage of mission capable A/C | misson capable = percentage_AC_MC * SVC[t]

        self.startdate = '1/1/2021' #start date for Sim format: "dd/mm/yyyy" used to recalulate date input to time index
        self.QRA_start = 15         #timestep the QRA duties start
        self.QRA_length = 91        #number of days QRA duty is scheduled

        self.night_allowed = [
            '8/3/2021',
            '22/10/2021'
        ]                           #night sorties not allowed between these dates
        self.hollidays = [
            '1/1/2021',             
            '2/4/2021',             
            '4/4/2021',
            '5/4/2021',
            '27/4/2021',
            '5/5/2021',
            '13/5/2021',
            '23/5/2021',
            '24/5/2021',
            '25/12/2021',
            '26/12/2021'
        ]                           #national hollidays where no flights are schedules
        self.vacation = True        #toggle scheduling of vacation days
        self.weeks_off = [[
            '15/2/2021',
            '22/2/2021',
            '29/3/2021',
            '5/4/2021',
            '12/4/2021',
            '26/4/2021',
            '10/5/2021',
            '24/5/2021',
            '14/6/2021',
            '20/9/2021',
            '4/10/2021',
            '18/10/2021',
            '1/11/2021',
            '15/11/2021',
            '29/11/2021',
            '6/12/2021',
            '13/12/2021',
            '20/12/2021',
        ],[ '5/7/2021',
            '19/7/2021',
            '2/8/2021',
            '16/8/2021',
        ]]                          #from set 1, 3 random weeks are selected. From set 2, a 2 week period is selected
                                    #pilots are required to attend 4 mandatory SQN training days per year
        self.training = True        #toggle manditory SQN training      
        self.team_training = [
            '4/1/2021',
            '5/4/2021',
            '1/7/2021',
            '1/10/2021'
        ]

        self.SC_max = 4            #number of simulators
        self.SC_failure = .05      #chance of failure on demand for single simulator 
        self.SC_preventive_maintenance = [
            '29/1/2021',
            '26/2/2021',
            '26/3/2021',
            '30/4/2021',
            '28/5/2021',
            '25/6/2021',
            '30/7/2021',
            '27/8/2021',
            '24/9/2021',
            '29/10/2021',
            '26/11/2021',
            '31/12/2021'
        ]                           #preventive maintenance is performed on these dates

        #--------------------FMP SETTINGS------------------------
        #M. Verhoeff et al. Maximizing Operational Readiness in Military Aviation
        #by Optimizing Flight and Maintenance Planning DOI:10.1016/j.trpro.2015.09.048

        self.RFT_max = 200          #maximum remaining FH for a single AC
        self.RFT_min = 0            #min flight hours where AC can be assigned fase maint.
        self.RMT_max = 15           #number of days a AC is in fase maint.
        self.FHR = 2360             #yearly flight hour requirement for the sqn 

        self.tol_FHR = .05          #daily allowed deviation from req. flight hours
        self.ACR = .6               #ratio of AC req. to be servicable
        self.M_max = 2              #max no. of AC allowed in fase maint.
        self.MT_max = 1             #fase maint. reduction per day (default 1)

        #NB fleet configuration inputs for the FMP model are defined in >data.py>aircraft

        #--------------------DATA INIT------------------------
        #these settings toggle wheter the pilots are randomly generated. Default False
        self.random_pilots = kwargs.get('random_pilots', False)
        self.pilot_file = kwargs.get('pilot_file', 'data/pilots.csv')
        #if random generation True the following settings are used for the pilot
        #population. All FL4 and FL2 pilots are considred EXP. Any EXP pilot that is 
        #not FL4 will automatically be assigned FL2 assigning more FL4 pilots than EXP
        #will automatically reduce the no of FL4 pilots to the no of EXP pilots
        self.no_pilots = 15
        self.no_EXPpilots = 12
        self.no_FL4pilots = 12
        #missions are read from csv file
        self.mission_file = kwargs.get('mission_file', 'data/missions.csv')
        #max FTC mission index:
        #|------------- Desired YTP------|-----------Minimum YTP---------|
        #|--Exp pilots--|--inexp pilots--|--Exp pilots--|--inexp pilots--|
        #|--Live-|--Sim-|--Live--|--Sim--|--Live-|--Sim-|--Live--|--Sim--|
        #[    0,     1,      2,      3,       4,     5,      6,       7  ]
        self.FTC_max = [12, 9, 13, 13, 10, 9, 15, 11]    #number of FTC missions allowed
        self.FTC_extra = [69 * (.49), 
            53 * (.38), 
            73 * (.48), 
            72 * (.39), 
            41 * (self.sortie_correct -1), 
            37 * (self.sortie_correct -1),
            58 * (self.sortie_correct -1), 
            42 * (self.sortie_correct -1)]
        self.FMP_file = kwargs.get('FMP_sortie_file', 'data/FMP_sorties.csv')
        self.svc_file = kwargs.get('svc_file', 'data/svc_sorties.csv')

        #--------------------PLOT SETTINGS------------------------
        self.plot_path = 'figures/' #folder plots are saved to
        self.plot_style = ['science','ieee'] #science IEEE 



