# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ...
# ...
# ... 01-mai-2020
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


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

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... some directories
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

home_dir = '/home/mcdevitt/PycharmProjects/lstm_101/src/'
data_dir = '/home/mcdevitt/PycharmProjects/lstm_101/data/'
plot_dir = '/home/mcdevitt/PycharmProjects/lstm_101/plot/'
rprt_dir = '/home/mcdevitt/PycharmProjects/lstm_101/rprt/'
# ........................................................


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... some utility functions
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def sprintf(buf, fmt, *args):
    buf.write(fmt % args)


def dup_scaler(scaler, sc2, col=0):
    """
     Returns an object sklearn MinMaxScaler with values extracted from a multi-column MinMaxScaler
     :param scaler : multicolumn scaler, previously defined
     :param col : column number from which to extract scaling values
     :return sc2 : previously instatiated MinMaxScaler in which to copy column specific scaling values
     :rtype: sklearn scaler
     """

    sc2.data_max_ = scaler.data_max_[col]
    sc2.data_min_ = scaler.data_min_[col]
    sc2.data_range_ = scaler.data_range_[col]
    sc2.min_ = scaler.min_[col]
    sc2.scale_ = scaler.scale_[col]

    return sc2


def df_cols_like(df):
    """
    Returns an empty data frame with the same column names and types as df
    https://stackoverflow.com/questions/27467730/is-there-a-way-to-copy-only-the-structure-not-the-data-of-a-pandas-dataframe
    """
    df2 = pd.DataFrame({i[0]: pd.Series(dtype=i[1])
                        for i in df.dtypes.iteritems()},
                       columns=df.dtypes.index)
    return df2


def rmse(y, y_hat):

    root_mean_squared_error = ((y_hat - y) ** 2).mean() ** .5
    return root_mean_squared_error


def clean_column_names(ls_column_names, chars_to_remove=None):

    if chars_to_remove is None:
        chars_to_remove = [' ', '-', '\.']
    ls_column_names = ls_column_names.str.lower()
    for c in chars_to_remove:
        ls_column_names = ls_column_names.str.replace(c, '_')

    return ls_column_names

# ........................................................

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... LSTM model implementation
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def lstm_001(n_data_size = 2000, n_seq = 20, n_test = 90, n_future = 30, n_feature = 1,
             feature_columns = ['adj_close'],
             n_layers = 2, batch_size = 32, n_epochs = 20,
             plot_save = False):
    """
    Provides basic structure for time-series forecasting using LSTM as primary RNN model element

    :return: sets of output plot summarize model fit and forecasts
    """
    # ... generate run id, seed, plot_dir

    uuid6 = str(uuid.uuid4()).lower()[0:6]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid6
    print('run initiated with run_id ', run_id)

    np.random.seed(20200504)
    os.chdir(plot_dir)

    # ... load data

    os.chdir(data_dir)
    print(os.getcwd())

    f = "sp500.csv"
    df_sp500 = pd.read_csv(f)
    print('read ', df_sp500.shape[0], 'lines from ', f)

    df_sp500 = df_sp500.dropna()
    df_sp500.columns = clean_column_names(df_sp500.columns)
    df_sp500['zeros'] = 0
    df_sp500['date'] = pd.to_datetime(df_sp500['date'])

# ... drop leading rows if data set longer than desired size

    if len(df_sp500) > n_data_size:
        df_sp500_red = df_sp500.tail(n_data_size)
    else:
        df_sp500_red = df_sp500
        print('n_data_size exceeds data set length')
        print('n_data_size = ', n_data_size)
        print('data set size ', df_sp500.shape)
    df_sp500_red.reset_index(drop=True, inplace=True)

    # ... input data set : Open + Volume
    df_oz = df_sp500_red

#    plt.figure(figsize=(5, 3))
#    plt.plot(df_oz['date'], df_oz[feature_columns[0]])
#    plt.close()

    # ... set up train and test size

    n_train = len(df_oz) - n_test
    if n_train < 1:
        print('test length exceeds data set length')

# ... n_cols <= n_feature ???
    n_cols = n_feature

    df_oz_train = df_oz.head(n_train)
    df_oz_test = df_oz.tail(n_test)
    df_oz_test.reset_index(drop=True, inplace=True)

 #   plt.figure(figsize=(5, 3))
 #   plt.plot(df_oz_train['date'], df_oz_train[feature_columns[0]])
 #   plt.plot(df_oz_test['date'], df_oz_test[feature_columns[0]])
 #   plt.close()

    df_oz_train_data = df_oz_train[feature_columns]
    oz = df_oz_train_data.to_numpy()

    print('oz shape = ', oz.shape)

    # ... set up scaler in 0,1 range

    scaler = MinMaxScaler(feature_range=(0, 1))
    sc2 = MinMaxScaler(feature_range=(0, 1))

    # ... scale on train set
    oz_scaled = scaler.fit_transform(oz[:, 0: n_cols])

#    plt.figure(figsize=(5, 3))
#    plt.plot(oz_scaled[:, 0])
#    plt.close()

    # ... dup scaler for later single column hack

    sc2 = copy.deepcopy(scaler)
    sc2 = dup_scaler(scaler, sc2, 0)

    # ... use n_seq day scrolling window for feature
    # ... hold out last n_test days for evaluation

    x = []
    y = []
    # ... TODO : this can be done with split and reshape ?
    # .. TODO ... is this lagging properly ???

    for i in range(n_seq, n_train):
        x.append(oz_scaled[i - n_seq: i])
        y.append(oz_scaled[i, 0])

    # ... shape data to fit keras input structure

    x, y = np.array(x), np.array(y)
    x = x.reshape(x.shape[0], x.shape[1], n_cols)

    print('The 3 input dimensions for keras independent data are:')
    print('\tsamples, time steps, and features.')
    print('Current trial,')
    print('\tsamples (n_data_size - n_test - n_seq) = ', n_data_size - n_test - n_seq)
    print('\ttime steps (n_seq) = ', n_seq)
    print('\tfeatures = ', n_cols)
    print('x reshape = ', x.shape)
    print('y reshape = ', y.shape)

    assert (n_seq == x.shape[1]) # time steps
    assert (n_feature == x.shape[2]) # features

    # ... build a model

    model = Sequential()
    model.add(LSTM(units=50,
                   return_sequences=True,
                   input_shape=(n_seq, n_feature)))
    model.add(Dropout(0.2))

    for il in range(n_layers-1):
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

# ... single output model
    model.add(Dense(units=1))

    callback = EarlyStopping(monitor='loss', patience=10)

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mape', 'mse'])

    start_time = timer()
    print('start model fit : ', start_time)
    history = model.fit(x, y,
                        # validation_split = 0.25,
                        epochs= n_epochs,
                        batch_size=batch_size,
                        callbacks=[callback],
                        verbose=0)
    end_time = timer()
    del_time = end_time - start_time
    print('fit complete : ', end_time)
    print('fit time = ', round(del_time/60, 2), ' (minutes)')
    print(model.summary())

    model_train_mse = model.evaluate(x, y, verbose=0)[0]
    model_train_mape = model.evaluate(x, y, verbose=0)[1]
    print('model performance : ', model_train_mape, model_train_mse)

    os.chdir(plot_dir)
    plt.figure(figsize=(5, 3))
    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    if plot_save:
        plt.savefig('sp500_lstm_fit_history_' + run_id + '.png')
    plt.close()

    # ... predict on train set

    y_hat_train = model.predict(x)
    y_hat_train_inv = sc2.inverse_transform(y_hat_train).reshape(-1, 1)
    ls_y_hat_train_inv = list(y_hat_train_inv.reshape(y_hat_train_inv.shape[0]))
    ls_y_hat_train_inv_extend = n_seq * [np.nan] + ls_y_hat_train_inv

    df_new_col = pd.DataFrame({'y_hat_train': ls_y_hat_train_inv_extend})
    df_oz_train = pd.concat([df_oz_train, df_new_col], axis=1)

    # ... error metric on train set

    df_oz_train_rmse = rmse(df_oz_train[feature_columns[0]], df_oz_train['y_hat_train'])

#    plt.figure(figsize=(5, 3))
#    plt.plot(df_oz_train['date'], df_oz_train[feature_columns[0]])
#    plt.plot(df_oz_test['date'], df_oz_test[feature_columns[0]])
#    plt.plot(df_oz_train['date'], df_oz_train['y_hat_train'], marker='o')
#    plt.close()

    # ... predict on test set

    df_oz_test_data = df_oz_test[feature_columns]
    oz_test = df_oz_test_data.to_numpy()

    print('oz shape = ', oz_test.shape)

# ... pre-pend test set w/ n_seq prior points to construct input sequence
    oz_test_scaled = scaler.transform(oz_test[:, 0: n_cols])
    last_train_pts = oz_scaled[-n_seq:]

    oz_test_scaled = np.concatenate((last_train_pts, oz_test_scaled), axis=0)

# ... construct each sample of length n_seq
    x_test = []
    for i in range(n_seq, len(oz_test_scaled)):
        x_test.append(oz_test_scaled[i - n_seq: i])

    x_test = np.array(x_test)

    y_hat_test = model.predict(x_test)

    y_hat_test_inv = sc2.inverse_transform(y_hat_test.reshape(-1, 1))

    ls_y_hat_test_inv = list(y_hat_test_inv.reshape(y_hat_test_inv.shape[0]))
    df_new_col = pd.DataFrame({'y_hat_test': ls_y_hat_test_inv})
    df_oz_test = pd.concat([df_oz_test, df_new_col], axis=1)

# ... error metric on test set

    df_oz_test_rmse = rmse(df_oz_test[feature_columns[0]], df_oz_test['y_hat_test'])
    print(df_oz_test_rmse)

    # ... forward forecast
    # ... what assumptions to make about exogenous columns future data !!!!

    df_oz_future = df_oz.tail(n_seq)
    df_oz_future.reset_index(drop=True, inplace=True)

    # ... create holding column for predicted values
    df_oz_future['y_hat_future'] = np.nan

    print('future predictions')
    for ia in range(n_future):
        # ... retain last n_seq rows of future dataframe
        df_oz_future_data = df_oz_future.tail(n_seq)

        print(40*'=-')
        print('iteration ', ia)
        print(df_oz_future_data)

        # ... select just columns used in model
        df_oz_future_data = df_oz_future_data[feature_columns]

        # ... convert to numpy array, then scale
        oz_future = df_oz_future_data.to_numpy()
        oz_future_scaled = scaler.transform(oz_future[:, 0: n_cols])

        # ... reshape args : samples, time steps, features
        x_future = oz_future_scaled.reshape(1, n_seq, n_cols)

        print('-------------------- model x inputs -----------------------')
        print(x_future)
        y_hat_future = model.predict(x_future)
        y_hat_future_inv = sc2.inverse_transform(y_hat_future)
        sc_y_hat_future_inv = list(y_hat_future_inv.reshape(y_hat_future_inv.shape[0]))[0]
        print('next predicted value : ', sc_y_hat_future_inv)

        df_new_row = df_cols_like(df_oz_future)
        df_new_row['date'] = df_oz_future['date'].tail(1) + timedelta(days=1)
        df_new_row['y_hat_future'] = sc_y_hat_future_inv
        df_new_row[feature_columns[0]] = sc_y_hat_future_inv

        df_oz_future = pd.concat([df_oz_future, df_new_row])

# ... forward fill exogenous columns ... TODO : naïve estimation not the best option !!
        for this_column in feature_columns[1:]:
            df_oz_future[this_column] = df_oz_future[this_column].fillna(method='ffill')

# ... end of loop on stepwise future predictions

    # ... make a plot

    data_params = "n_data_size : %d\n" \
                  "n_seq       : %d\n" \
                  "n_test      : %d\n" \
                  "n_future    : %d" \
                  % (n_data_size, n_seq, n_test, n_future)
    model_params = "n_layers    : %d\n" \
                   "batch_size  : %d\n" \
                   "n_epochs    : %d" \
                   % (n_layers, batch_size, n_epochs)
    error_metric = "test rmse : %.2f" % df_oz_test_rmse


# ... full x-range plot

    df_oz_train_plot = df_oz_train
    df_oz_test_plot = df_oz_test

    plt.figure(figsize=(10, 5))

    plt.plot(df_oz_train_plot['date'], df_oz_train_plot[feature_columns[0]], color='lightcoral')
    plt.scatter(df_oz_train_plot['date'], df_oz_train_plot['y_hat_train'],
                label = 'y_hat_train',
                marker='o',
                color='red',
                s=5)
    plt.plot(df_oz_test_plot['date'], df_oz_test_plot[feature_columns[0]], color='cornflowerblue')
    plt.scatter(df_oz_test_plot['date'], df_oz_test_plot['y_hat_test'],
                label = 'y_hat_test',
                marker='s',
                color='royalblue',
                s=5)
    plt.scatter(df_oz_future['date'], df_oz_future['y_hat_future'],
                label = 'y_hat_future',
                marker='D',
                s=20,
                color='blueviolet')

    plt.title('Market value Prediction - single explanatory column')
    plt.xlabel('date')
    plt.ylabel('Price')
    plt.legend()
#    plt.ylim(100, 350)
    plt.grid(True)

    plt.text(datetime(2011, 10, 1), 2250, data_params, size=9,
             verticalalignment="baseline",
             horizontalalignment="left",
             multialignment="left",
             bbox=dict(fc="lightgrey"))
    plt.text(datetime(2011, 10, 1), 1750, model_params, size=9,
             verticalalignment="baseline",
             horizontalalignment="left",
             multialignment="left",
             bbox=dict(fc="lightgrey"))

    if plot_save:
        plt.savefig('sp500_lstm_single_column_' + run_id + '.png')
    plt.close()

# ... last few months  x-range plot

    df_oz_train_plot = df_oz_train[df_oz_train['date'] >= datetime(2019, 1, 1)]
    df_oz_test_plot = df_oz_test[df_oz_test['date'] >= datetime(2019, 1, 1)]

    plt.figure(figsize=(10, 5))

    plt.plot(df_oz_train_plot['date'], df_oz_train_plot[feature_columns[0]], color='lightcoral')
    plt.scatter(df_oz_train_plot['date'], df_oz_train_plot['y_hat_train'],
                label='y_hat_train',
                marker='o',
                color='red',
                s=5)
    plt.plot(df_oz_test_plot['date'], df_oz_test_plot[feature_columns[0]], color='cornflowerblue')
    plt.scatter(df_oz_test_plot['date'], df_oz_test_plot['y_hat_test'],
                label='y_hat_test',
                marker='s',
                color='royalblue',
                s=5)
    plt.scatter(df_oz_future['date'], df_oz_future['y_hat_future'],
                label='y_hat_future',
                marker='D',
                s=20,
                color='blueviolet')

    plt.title('Market value Prediction - single explanatory column')
    plt.xlabel('date')
    plt.ylabel('Price')
    plt.legend()
    #    plt.ylim(100, 350)
    plt.grid(True)

    plt.text(datetime(2019, 1, 1), 3000, data_params, size=9,
             verticalalignment="baseline",
             horizontalalignment="left",
             multialignment="left",
             bbox=dict(fc="lightgrey"))
    plt.text(datetime(2019, 1, 1), 2850, model_params, size=9,
             verticalalignment="baseline",
             horizontalalignment="left",
             multialignment="left",
             bbox=dict(fc="lightgrey"))
    plt.text(datetime(2019, 1, 1), 2600, error_metric,
             size=10,
             verticalalignment="baseline",
             horizontalalignment="left",
             multialignment="left",
             bbox=dict(fc="lightgrey"))

    if plot_save:
        plt.savefig('sp500_lstm_single_column_' + run_id + '_2019.png')
    plt.close()

    df_results = pd.DataFrame(
        [{"n_data_size" : n_data_size,
             "n_seq" : n_seq,
             "n_test" : n_test,
             "n_future" : n_future,
             "n_layers" : n_layers,
             "batch_size" : batch_size,
             "n_epochs" : n_epochs,
             "train_mape" : model_train_mape,
             "train_mse" : model_train_mse,
             "train_rmse" : df_oz_train_rmse,
             "test_rmse" : df_oz_test_rmse,
             "timer" : del_time}])

    return df_results


# from : https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
# The LSTM input layer must be 3D.
# The meaning of the 3 input dimensions are: samples, time steps, and features.
# The LSTM input layer is defined by the input_shape argument on the first hidden layer.
# The input_shape argument takes a tuple of two values that define the number of time steps and features.
# The number of samples is assumed to be 1 or more.
# The reshape() function on NumPy arrays can be used to reshape your 1D or 2D data to be 3D.
# The reshape() function takes a tuple as an argument that defines the new shape.


if __name__ == '__main__':

    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ... data params
    n_data_size = 2000
    n_seq = 30
    n_test = 120
    n_future = 30
    n_feature = 4
    feature_columns = ['adj_close', 'high', 'low', 'volume']

    # ... model params
    batch_size = 64
    n_epochs = 50

    # ... plot switch
    save_plots = True

    df_summary = pd.DataFrame()

    for n_seq in [2, 5, 10]:
        for n_layers in [1, 4]:
            for batch_size in [64, 128]:
                df = lstm_001(n_data_size, n_seq, n_test, n_future,
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
