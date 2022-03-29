'''
File contains all simulation presets and settings
'''

import numpy as np
import pulp as pl

class Configuration():
    def __init__(self, *args, **kwargs):
        #--------------------SIM_SETTINGS------------------------
        self.verbose = True         #toggle to enable print to console for print functions 
        self.print_schedule = True  #toggle wheter the sortie schedule is saved to xls
        self.fixed_seed = True      #toggle to fix seed for RNG
        self.seed = 69              #seed for RNG init in numpy, to fix RNG for academia
        self.iterations = 1         #number of iterations in simulation 

        #solver is used to modify the pulp LP solver setting in the milp.py file
        #for academic purposes a free licence for gurobi can be obtained at
        #the following adress:
        #https://www.gurobi.com/academia/academic-program-and-licenses/
        self.solver = pl.GUROBI_CMD(options=[('MIPGap',0.01)]) 

        #--------------------VARIABLES------------------------
        self.tau = 365              #set lenght of simulation period tau 
        self.waves = 2              #number of waves per day 
        self.startdate = '1/1/2021' #start date for Sim format: "dd/mm/yyyy"
        self.QRA_start = 15         #timestep the QRA duties start
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
        ]
        #pilots are required to attend 4 mandatory SQN training days per year
        self.team_training = [
            '4/1/2021',
            '5/4/2021',
            '1/7/2021',
            '1/10/2021'
        ]
        #corrective maintenance settings
        self.MTBF_ac = 80           #mean time between failures for aircraft
        self.MTTR_ac = 12           #mean time to repair for aircraft
        self.AC_tmax = 12           #Number of aircraft
        self.SC_tmax = 4            #Number of simulators
        
        #FMP settings
        # M. Verhoeff et al. Maximizing Operational Readiness in Military Aviation
        #by Optimizing Flight and Maintenance Planning DOI:10.1016/j.trpro.2015.09.048

        self.RFT_max = 200          #maximum remaining FH for a single AC
        self.RFT_min = 0            #min flight hours where AC can be assigned fase maint.
        self.RMT_max = 15           #number of days a AC is in fase maint.
        self.FHR = 4680             #yearly flight hour requirement for the sqn 

        self.tol_FHR = .05          #daily allowed deviation from req. flight hours
        self.ACR = .6               #fraction of AC req. to be servicable
        self.M_max = 3              #max no. of AC allowed in fase maint.
        self.MT_max = 1             #fase maint. reduction per day (default 1)

        self.sp_tau = 200           #set max length of subproblem for the FMP

        #--------------------DATA INIT------------------------
        #these settings toggle wheter the pilots are randomly generated. Default False
        self.random_pilots = kwargs.get('random_pilots', True)
        self.pilot_file = kwargs.get('pilot_file', 'data/pilots.csv')
        #if random generation True the following settings are used for the pilot
        #population. All FL4 and FL2 pilots are considred EXP. Any EXP pilot that is 
        #not FL4 will automatically be assigned FL2 assigning more FL4 pilots than EXP
        #will automatically reduce the no of FL4 pilots to the no of EXP pilots
        self.no_pilots = 52
        self.no_EXPpilots = 30
        self.no_FL4pilots = 28
        #missions are read from csv file
        self.mission_file = kwargs.get('mission_file', 'data/missions.csv')
        #max FTC mission index:
        #|------------- Desired YTP------|-----------Minimum YTP---------|
        #|--Exp pilots--|--inexp pilots--|--Exp pilots--|--inexp pilots--|
        #|--Live-|--Sim-|--Live--|--Sim--|--Live-|--Sim-|--Live--|--Sim--|
        #[    0,     1,      2,      3,       4,     5,      6,       7  ]
        self.FTC_max = [12, 9, 13, 13, 10, 9, 15, 11]    #number of FTC missions allowed


        #--------------------PLOT SETTINGS------------------------
        self.plot_path = 'figures/' #folder plots are saved to
        self.plot_style = ['science','ieee'] #science IEEE 
        self.colorblind_mode = False
        #if colorblind is enabled, set type of colorblindness
        #available: deuteranopia, protanopia, tritanopia. defauld=deuteranopia
        self.colorblind_type = 'deuteranopia'

    def get_palette(self):
        '''returns appropriate color palette

        Uses config.plot_style to determine which palette to pick, 
        and changes palette to colorblind mode (config.colorblind_mode)
        and colorblind type (config.colorblind_type) if required.

        Palette colors are based on
        https://venngage.com/blog/color-blind-friendly-palette/
        '''
        palettes = {'regular': {'default': ['gray', 'red', 'green', 'black'],
                                'dark': ['#404040', '#ff0000', '#00ff00', '#000000']},
                    'deuteranopia': {'default': ['gray', '#a50f15', '#08519c', 'black'],
                                     'dark': ['#404040', '#fcae91', '#6baed6', '#000000']},
                    'protanopia': {'default': ['gray', '#a50f15', '08519c', 'black'],
                                   'dark': ['#404040', '#fcae91', '#6baed6', '#000000']},
                    'tritanopia': {'default': ['gray', '#a50f15', '08519c', 'black'],
                                   'dark': ['#404040', '#fcae91', '#6baed6', '#000000']}
                    }

        if self.colorblind_mode:
            return palettes[self.colorblind_type.lower()][self.plot_style]
        else:
            return palettes['regular'][self.plot_style]