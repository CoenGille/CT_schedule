'''
collection of utility methods shared across files
'''

from datetime import datetime, timedelta

import itertools
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

def date_list(start_date, i):
    date_generator = (((datetime.strptime(start_date,"%d/%m/%Y") + \
                        timedelta(days=i)).date()).strftime('%d/%m/%Y') \
                        for i in itertools.count()
    )
    j = itertools.islice(date_generator,i)
    k = pd.Series(list(j))
    return k

def dict_writer(*args):
    dict_ = {}
    for a in args:
        dict_[a[0]] = a[1]
    return dict_

def find_largest_value(ds):
    _s = 1e-9 #sufficiently small number
    key = ''
    for k, v in ds.items():
        size = len(v)
        if size >= _s:
            _s = size
            key = k
    return key

def find_smallest_value(ds):
    _s = 1e9  #sufficiently large number
    key = ''
    for k, v in ds.items():
        size = len(v)
        if size <= _s:
            _s = size
            key = k
    return key