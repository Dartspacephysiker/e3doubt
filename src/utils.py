import os
import pandas as pd

basepath = os.path.dirname(__file__)

default_site_csv = os.path.abspath(os.path.join(basepath,'../','data','sites.csv'))

# read coefficient file and store in pandas DataFrame - with column names from last row of header:
colnames = ([x for x in open(default_site_csv).readlines() if x.startswith('#')][-1][1:]).strip().split(',') 

get_supported_sites = lambda fn=default_site_csv: pd.read_table(fn, skipinitialspace = True, comment = '#', sep = ',', names = colnames, index_col = [0])
