CT_schedule
Decision support tool for CT schedule generation

In order to run the simulation from console run:

>>python simulation.py

For a comprehensive list of events and (re)scheduling decisions run the following from console:

>>python simulation.py > output.txt

-----simulation.py:-----
primary file for the generaton of the CT schedule. The Simulation class handles the simulation of
all events in the planning horizon. No user input in this file is required. All user selected options
are handled in the support files.

Returns:
If selected in config.py an initial FMP schedule is generated for the entrire planning horizon the 
files in */data/ containing the output of the FMP are replaced. If an initial CT schedule is generated
files in */data/ containing the CT schedule rae replaced. These files are used a input for the 
simulation. 

the */results/ folder collects output from each simulation run 


PulP modules for linear programs:

https://www.gurobi.com/academia/academic-program-and-licenses/


-----CT.py-----
File used for the formulation of the CT sortie scheduling MILP. Data management is handled by simulation.py. 
Initial schedule generation and Ct settings can be toggled in config.py

-----FMP.py-----
Adapted from:
M. Verhoeff et al. (2015) Maximizing Operational Readiness in Military Aviation
by Optimizing Flight and Maintenance Planning DOI:10.1016/j.trpro.2015.09.048


support files:

-----config.py-----
Contains all model settings.
-----data.py-----
This file converts the csv input from the */data folder for use in the simulation.
-----utils.py-----