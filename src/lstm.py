# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ...
# ...
# ... 01-mai-2020
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... imports
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

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

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping

import utils as u
import metrics as m
import make_feature as mf

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... some directories
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

home_dir = '/home/mcdevitt/PycharmProjects/lstm_101/src/'
data_dir = '/home/mcdevitt/PycharmProjects/lstm_101/data/'
plot_dir = '/home/mcdevitt/PycharmProjects/lstm_101/plot/'
rprt_dir = '/home/mcdevitt/PycharmProjects/lstm_101/rprt/'
rslt_dir = '/home/mcdevitt/PycharmProjects/lstm_101/rslt/'

# ........................................................

# ........................................................

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... LSTM model implementation
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def lstm_001(df_feature, n_data_size = 2000, n_seq = 20, n_test = 90,
             n_future = 30, n_fwd = 1,
             n_feature = 1, feature_columns = None,
             n_layers = 2, batch_size = 32, n_epochs = 20,
             plot_save = False):
    """
    Provides basic structure for time-series forecasting using LSTM as primary RNN model element

    :return: sets of output plot summarize model fit and forecasts
    """
    # ... generate run id, seed, plot_dir

    if feature_columns is None:
        feature_columns = ['adj_close']

    uuid6 = str(uuid.uuid4()).lower()[0:6]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid6
    print('run initiated with run_id ', run_id)

    np.random.seed(20200504)

# ... data frame for model ...
# ... drop leading rows if data set longer than desired size

    if len(df_feature) > n_data_size:
        df_model = df_feature.tail(n_data_size)
    else:
        df_model = df_feature
        print('n_data_size exceeds data set length')
        print('n_data_size = ', n_data_size)
        print('data set size ', df_feature.shape)
    df_model.reset_index(drop=True, inplace=True)

    # ... set up train and test size

    n_train = len(df_model) - n_test
    if n_train < 1:
        print('test length exceeds data set length')

# ... n_cols <= n_feature ???
    n_cols = n_feature

    df_train = df_model.head(n_train)
    df_test = df_model.tail(n_test)
    df_test.reset_index(drop=True, inplace=True)

    df_train_data = df_train[feature_columns]
    ar_train_data = df_train_data.to_numpy()

    print('train array shape = ', ar_train_data.shape)

    # ... set up scaler in 0,1 range

    scaler = MinMaxScaler(feature_range=(0, 1))
    sc2 = MinMaxScaler(feature_range=(0, 1))

    # ... scale on train set
    ar_train_scaled = scaler.fit_transform(ar_train_data[:, 0: n_cols])

    # ... dup scaler for later single column hack
    sc2 = copy.deepcopy(scaler)
    sc2 = u.dup_scaler(scaler, sc2, 0)

    # ... use n_seq day scrolling window for feature
    # ... hold out last n_test days for evaluation

    x = []
    y = []
    # ... TODO : this can be done with split and reshape ?
    # ... TODO ... is this lagging properly ???

    for i in range(n_seq, n_train - n_fwd + 1):
        x.append(ar_train_scaled[i - n_seq: i])
        y.append(ar_train_scaled[i + (n_fwd - 1), 0])

    # ... shape data to fit keras input structure
    # ... TODO : continue to evaluate array size reshape() w/ future changes

    x, y = np.array(x), np.array(y)
    x = x.reshape((x.shape[0], x.shape[1], n_cols))

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
                        validation_split=0.25,
                        epochs=n_epochs,
                        batch_size=batch_size,
                        callbacks=[callback],
                        verbose=2)
    end_time = timer()
    del_time = end_time - start_time
    print('fit complete : ', end_time)
    print('fit time = ', round(del_time/60, 2), ' (minutes)')
    print(model.summary())

    model_train_mse = model.evaluate(x, y, verbose=0)[0]
    model_train_mape = model.evaluate(x, y, verbose=0)[1]
    print('model performance : ', model_train_mape, model_train_mse)

    print(history.history.keys())

    os.chdir(plot_dir)
    plt.figure(figsize=(5, 3))
    plt.plot(history.history['loss'])
    plt.yscale('log')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    if plot_save:
        plt.savefig('lstm_fit_history_' + run_id + '.png')
    plt.close()

    # ... predict on train set

    y_hat_train = model.predict(x)
    y_hat_train_inv = sc2.inverse_transform(y_hat_train).reshape(-1, 1)
    ls_y_hat_train_inv = list(y_hat_train_inv.reshape(y_hat_train_inv.shape[0]))
    ls_y_hat_train_inv_extend = n_seq * [np.nan] + ls_y_hat_train_inv

    df_new_col = pd.DataFrame({'y_hat_train': ls_y_hat_train_inv_extend})
    df_train = pd.concat([df_train, df_new_col], axis=1)

    # ... error metric on train set

    df_train_rmse = m.rmse(df_train[feature_columns[0]], df_train['y_hat_train'])

#    plt.figure(figsize=(5, 3))
#    plt.plot(df_oz_train['date'], df_oz_train[feature_columns[0]])
#    plt.plot(df_oz_test['date'], df_oz_test[feature_columns[0]])
#    plt.plot(df_oz_train['date'], df_oz_train['y_hat_train'], marker='o')
#    plt.close()

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... predict on test set
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    df_test_data = df_test[feature_columns]
    ar_test_data = df_test_data.to_numpy()

    print('test data array shape = ', ar_test_data.shape)

# ... pre-pend test set w/ n_seq prior points to construct input sequence
    ar_test_scaled = scaler.transform(ar_test_data[:, 0: n_cols])
    last_train_pts = ar_train_scaled[-n_seq:]

    ar_test_scaled = np.concatenate((last_train_pts, ar_test_scaled), axis=0)

# ... construct each sample of length n_seq
    x_test = []
    for i in range(n_seq, len(ar_test_scaled)):
        x_test.append(ar_test_scaled[i - n_seq: i])

    x_test = np.array(x_test)

    y_hat_test = model.predict(x_test)

    y_hat_test_inv = sc2.inverse_transform(y_hat_test.reshape(-1, 1))

    ls_y_hat_test_inv = list(y_hat_test_inv.reshape(y_hat_test_inv.shape[0]))
    df_new_col = pd.DataFrame({'y_hat_test': ls_y_hat_test_inv})
    df_test = pd.concat([df_test, df_new_col], axis=1)

# ... error metric on test set

    df_test_rmse = m.rmse(df_test[feature_columns[0]], df_test['y_hat_test'])
    print(df_test_rmse)

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... forward forecast
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... what assumptions to make about exogenous columns future data !!!!

    df_future_data = df_model.tail(n_seq)
    df_future_data.reset_index(drop=True, inplace=True)

    # ... create holding column for predicted values
    df_future_data['y_hat_future'] = np.nan

    print('future predictions')
    for ia in range(n_future):
        # ... retain last n_seq rows of future dataframe
        df_future_this = df_future_data.tail(n_seq)

        print(40*'=-')
        print('iteration ', ia)
        print(df_future_this)

        # ... select just columns used in model
        df_future_this = df_future_this[feature_columns]

        # ... convert to numpy array, then scale
        ar_future_this = df_future_this.to_numpy()
        ar_future_scaled = scaler.transform(ar_future_this[:, 0: n_cols])

        # ... reshape args : samples, time steps, features
        x_future = ar_future_scaled.reshape(1, n_seq, n_cols)

        print('-------------------- model x inputs -----------------------')
        print(x_future)
        y_hat_future = model.predict(x_future)
        y_hat_future_inv = sc2.inverse_transform(y_hat_future)
        sc_y_hat_future_inv = list(y_hat_future_inv.reshape(y_hat_future_inv.shape[0]))[0]
        print('next predicted value : ', sc_y_hat_future_inv)

        df_new_row = u.df_cols_like(df_future_data)
        df_new_row['date'] = df_future_data['date'].tail(1) + timedelta(days=1)
        df_new_row['y_hat_future'] = sc_y_hat_future_inv
        df_new_row[feature_columns[0]] = sc_y_hat_future_inv

        df_future_data = pd.concat([df_future_data, df_new_row])

# ... forward fill exogenous columns ... TODO : naïve estimation not the best option !!
        for this_column in feature_columns[1:]:
            df_future_data[this_column] = df_future_data[this_column].fillna(method='ffill')

# ... end of loop on stepwise future predictions


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
             "train_rmse" : df_train_rmse,
             "test_rmse" : df_test_rmse,
             "timer" : del_time}])

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
    error_metric = "test rmse : %.2f" % df_test_rmse

# ... combine data frames for storage

    df_train.rename(columns={'y_hat_train': 'y_hat'}, inplace=True)
    df_test.rename(columns={'y_hat_test': 'y_hat'}, inplace=True)
    df_future_data.rename(columns={'y_hat_future': 'y_hat'}, inplace=True)
    df_train['period'] = 'train'
    df_test['period'] = 'test'
    df_future_data['period'] = 'predict'
    last_test_date = df_test.loc[df_test.index[-1], 'date']
    df_future_data = df_future_data[df_future_data['date'] > last_test_date]

# ... df_ttp = df of train, test, and predict values

    df_ttp = pd.concat([df_train, df_test, df_future_data])
    f = Path(rslt_dir) / ('lstm_df_ttp_' + run_id + '.pkl')
    df_ttp.to_pickle(f)

# ... full x-range plot

    df_train_plot = df_train
    df_test_plot = df_test

    fig = plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(df_train_plot['date'], df_train_plot[feature_columns[0]], color='lightcoral')
    plt.scatter(df_train_plot['date'], df_train_plot['y_hat_train'],
                label = 'y_hat_train',
                marker='o',
                color='red',
                s=5)
    plt.plot(df_test_plot['date'], df_test_plot[feature_columns[0]], color='cornflowerblue')
    plt.scatter(df_test_plot['date'], df_test_plot['y_hat_test'],
                label = 'y_hat_test',
                marker='s',
                color='royalblue',
                s=5)
    plt.scatter(df_future_data['date'], df_future_data['y_hat_future'],
                label = 'y_hat_future',
                marker='D',
                s=20,
                color='blueviolet')

    plt.title('Market value Prediction - ')
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

    plt.subplot(2, 1, 2)
    plt.plot(df_train_plot['date'], df_train_plot[feature_columns[1]], color='grey')
    plt.grid(True)

    if plot_save:
        plt.savefig('lstm_ts_' + run_id + '.png')
    plt.close()

# ... last few months  x-range plot

    df_train_plot = df_train[df_train['date'] >= datetime(2019, 1, 1)]
    df_test_plot = df_test[df_test['date'] >= datetime(2019, 1, 1)]

    fig = plt.figure(figsize=(12, 9))

    plt.subplot(3, 1, 1)
    plt.plot(df_train_plot['date'], df_train_plot[feature_columns[0]], color='lightcoral')
    plt.scatter(df_train_plot['date'], df_train_plot['y_hat_train'],
                label='y_hat_train',
                marker='o',
                color='red',
                s=5)
    plt.plot(df_test_plot['date'], df_test_plot[feature_columns[0]], color='cornflowerblue')
    plt.scatter(df_test_plot['date'], df_test_plot['y_hat_test'],
                label='y_hat_test',
                marker='s',
                color='royalblue',
                s=5)
    plt.scatter(df_future_data['date'], df_future_data['y_hat_future'],
                label='y_hat_future',
                marker='o',
                s=20,
                color='blue')

    plt.title('Market value Prediction - ')
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

    plt.subplot(3, 1, 2)
    plt.plot(df_train_plot['date'], df_train_plot[feature_columns[1]], color='grey')
    plt.plot(df_test_plot['date'], df_test_plot[feature_columns[1]], color='cornflowerblue')
    plt.scatter(df_future_data['date'], df_future_data[feature_columns[1]],
                label = feature_columns[1] + '_projected',
                marker = 'o',
                s = 10,
                color='blueviolet')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(df_train_plot['date'], df_train_plot[feature_columns[2]], color='grey')
    plt.plot(df_test_plot['date'], df_test_plot[feature_columns[2]], color='cornflowerblue')
    plt.scatter(df_future_data['date'], df_future_data[feature_columns[2]],
                label = feature_columns[1] + '_projected',
                marker = 'o',
                s = 10,
                color='blueviolet')
    plt.legend()
    plt.grid(True)

    if plot_save:
        plt.savefig('lstm_ts_zoom' + run_id + '.png')
    plt.close()


    return df_results


# from : https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
# The LSTM input layer must be 3D.
# The meaning of the 3 input dimensions are: samples, time steps, and features.
# The LSTM input layer is defined by the input_shape argument on the first hidden layer.
# The input_shape argument takes a tuple of two values that define the number of time steps and features.
# The number of samples is assumed to be 1 or more.
# The reshape() function on NumPy arrays can be used to reshape your 1D or 2D data to be 3D.
# The reshape() function takes a tuple as an argument that defines the new shape.

