

import os
import copy
from datetime import datetime, timedelta
from timeit import default_timer as timer
import uuid

import numpy as np; print ("numpy", np.__version__)
import pandas as pd; print ("pandas", pd.__version__)

pd.set_option("display.max_rows", 999)
pd.set_option("display.max_columns", 999)
pd.set_option("display.width", 140)

import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import utils as u

home_dir = '/home/mcdevitt/PycharmProjects/lstm_101/src/'
data_dir = '/home/mcdevitt/PycharmProjects/lstm_101/data/'

# ... read in input files and assemble to df_feature

def make_feature():

# ... load data

    os.chdir(data_dir)
    print(os.getcwd())

    cols_2_keep = ['date', 'adj_close']

# ... S&P 500

    f = "sp500.csv"
    df_sp500 = pd.read_csv(f)
    print('read ', df_sp500.shape[0], 'lines from ', f)

    df_sp500 = df_sp500.dropna()
    df_sp500.columns = u.clean_column_names(df_sp500.columns)
    df_sp500['date'] = pd.to_datetime(df_sp500['date'])
    df_sp500 = df_sp500[cols_2_keep]
    df_sp500.rename({'adj_close': 'sp500'}, axis=1, inplace=True)

# ... DJI

    g = "dji.csv"
    df_dji = pd.read_csv(g)
    print('read ', df_dji.shape[0], 'lines from ', g)
    
    df_dji = df_dji.dropna()
    df_dji.columns = u.clean_column_names(df_dji.columns)
    df_dji['date'] = pd.to_datetime(df_dji['date'])
    df_dji = df_dji[cols_2_keep]
    df_dji.rename({'adj_close': 'dji'}, axis=1, inplace=True)

# ... merge into common data frame

    df_feature = df_sp500.merge(df_dji, on='date', how='left')
    df_feature = df_feature.dropna()

    return df_feature
