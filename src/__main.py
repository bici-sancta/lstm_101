


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... imports
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

import platform; print(platform.platform())
import sys; print("Python", sys.version)
import re; print("re", re.__version__)

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
import metrics as m
import make_feature as mf
import lstm as l


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... some directories
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

home_dir = '/home/mcdevitt/PycharmProjects/lstm_101/src/'
data_dir = '/home/mcdevitt/PycharmProjects/lstm_101/data/'
plot_dir = '/home/mcdevitt/PycharmProjects/lstm_101/plot/'
rprt_dir = '/home/mcdevitt/PycharmProjects/lstm_101/rprt/'

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... main
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

if __name__ == '__main__':

    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ... data params
    n_data_size = 4000
    n_seq = 50
    n_test = 100
    n_future = 5
    n_fwd = 5
    n_feature = 2
    feature_columns = ['sp500', 'dji']

    df_feature = mf.make_feature()

    # ... model params
    batch_size = 64
    n_epochs = 50

    # ... plot switch
    save_plots = True

    df_summary = pd.DataFrame()

    for n_seq in [50]:
        for n_layers in [4]:
            for batch_size in [64]:
                df = l.lstm_001(df_feature, n_data_size, n_seq, n_test, n_future, n_fwd,
                                n_feature, feature_columns,
                                n_layers, batch_size, n_epochs, save_plots)
                df_summary = pd.concat([df_summary, df])
                print(timer())
                print(df)

    print(df_summary)

    os.chdir(rprt_dir)
    f_out = 'df_summary_' + run_time + '.csv'
    df_summary.to_csv(f_out, index = False)

    print(__name__, ' completed.')

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... eof
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
