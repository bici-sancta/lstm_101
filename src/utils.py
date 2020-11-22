# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... some utility functions
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

import pandas as pd
import yfinance as yf
from datetime import date, datetime, timedelta

def clean_column_names(ls_column_names, chars_to_remove=None):
    if chars_to_remove is None:
        chars_to_remove = [' ', '-', '\.']
    ls_column_names = ls_column_names.str.lower()
    for c in chars_to_remove:
        ls_column_names = ls_column_names.str.replace(c, '_')

    return ls_column_names


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


TRADING_DAYS_PER_YEAR = 253
DAYS_PER_YEAR = 365

def get_yahoo_data(days = 500, ticker = 'VIX') :

    yesterday = date.today() - timedelta(days = 1)
    calendar_days = days * DAYS_PER_YEAR / TRADING_DAYS_PER_YEAR * 1.05
    start_date = yesterday - timedelta(days = calendar_days)

    df = pd.DataFrame()

    df = yf.download(ticker, start_date, yesterday)
    df.reset_index(level=0, inplace=True)
    print(ticker)
    print(df.head())

    return df
