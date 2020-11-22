

import os
import copy
from datetime import datetime, timedelta
from timeit import default_timer as timer
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

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

def make_feature(ticker, days = 500, refresh = False, feature_column = 'adj_close'):

# ... load data

    df_feature = pd.DataFrame()

    cols_2_keep = ['date', feature_column]
    first = True

    for t in ticker:
        f = Path(data_dir) / (t + '.pkl')

        if not refresh:
            df_tckr = pd.read_pickle(f)
            print('read ', df_tckr.shape[0], 'lines from ', f)
        else:
            df_tckr = u.get_yahoo_data(days = days, ticker = t)
            print('refreshed ', df_tckr.shape[0], 'lines for ', t)
            df_tckr.to_pickle(f)
    
        df_tckr = df_tckr.dropna()
        df_tckr.columns = u.clean_column_names(df_tckr.columns)
        df_tckr['date'] = pd.to_datetime(df_tckr['date'])
        df_tckr = df_tckr[cols_2_keep]
        df_tckr.rename({feature_column: t}, axis=1, inplace=True)

        if first:
            df_feature = df_tckr
            first = False
        else:
            df_feature = df_feature.merge(df_tckr, on='date', how='left')
            print('feature accumulation : %4s | %06d' % (t, df_feature.shape[0]))

    return df_feature
