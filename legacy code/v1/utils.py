'''
collection of utility methods shared across files
'''
from datetime import datetime

import os
import pandas as pd


def check_folder(folder='output/file.py'):
    '''check if folder exists, make if not present'''
    if not os.path.exists(folder):
            os.makedirs(folder)

def excel_writer(header, filename, *args, **kwargs):
    '''general purpose pandas xlsx writer
    takes column header and filename as vars, pass df.args in format ['argname', arg]
    '''
    writer = pd.ExcelWriter(filename)
    for a in args:
        _a = a[1].copy()
        _a.columns = header
        _a.to_excel(writer, a[0], index=True)
    writer.save()

def date_indices(start_date, date_array):
    '''convert %d/%m/%y dates array from string to relative index postion. returns same
    length array with index postions relative to the start date == 0 index
    '''
    sd = datetime.strptime(start_date,"%d/%m/%Y")
    indices = []
    for i in date_array:
        j = datetime.strptime(i,"%d/%m/%Y")
        indices.append((j-sd).days)
    return indices