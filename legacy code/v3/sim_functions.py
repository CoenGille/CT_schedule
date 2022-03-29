import pandas as pd


def write_to_history(schedule, wave):
    print('write to history')
    for k in schedule[wave].index.values:
        sim.history.P_live[k][schedule[wave].loc[k]['mission']] += 1
    sim.history.FHR_realized += len(schedule[wave].index.values)*self.cf.ASD