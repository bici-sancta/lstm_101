
import os
import sys
import math
import random
import re
from pathlib import Path

import numpy as np
import time
from datetime import datetime
import locale
locale.setlocale(locale.LC_ALL, '')

import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import json

from style_settings import *
import utils as u


# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... time series observed and predict values plots
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def plot_ticker(t):

    data_dir = '/home/mcdevitt/PycharmProjects/lstm_101/data/'
    f = Path(data_dir) / (t + '.pkl')
    df_tckr = pd.read_pickle(f)
    df_tckr = df_tckr.dropna()
    df_tckr.columns = u.clean_column_names(df_tckr.columns)
    df_tckr['date'] = pd.to_datetime(df_tckr['date'])
    print('read ', df_tckr.shape[0], 'lines from ', f)
    
    marker_props = {'size' : 8,
                'opacity' : 0.4,
                'line' : {'color' : colors['paper_background20'], 'width' : 1}
                }


    n_rows = 1
    row_widths = [1]
    irow = 1

    fig_ticker = make_subplots(rows = n_rows,
                             cols = 1,
                             shared_xaxes = True,
                             vertical_spacing = 0.02,
                             row_width = row_widths)

    fig_ticker.add_trace(go.Scatter(x = df_tckr['date'],
                                  y = df_tckr['adj_close'],
                                  name = 'adj_close',
                                  mode = 'markers',
                                  marker = marker_props), row = irow, col = 1)

    plot_height = n_rows * 250 + 200
    plot_width = 900

    fig_ticker.update_layout(
                        # height = plot_height,
                        # width = plot_width,
                        title_text = t,
                        font_color = colors['text'],
                        plot_bgcolor = colors['plot_background'],
                        paper_bgcolor = colors['paper_background']
#                        template = 'seaborn'
                        )

    fig_ticker.update_xaxes(showline=True, linewidth=1, linecolor=colors['plot_grid'])
    fig_ticker.update_yaxes(showline=True, linewidth=1, linecolor=colors['plot_grid'])
    fig_ticker.update_xaxes(showgrid=True, gridwidth=1, gridcolor=colors['plot_grid'])
    fig_ticker.update_yaxes(showgrid=True, gridwidth=1, gridcolor=colors['plot_grid'])

    return fig_ticker


def make_plots(df, p3_selected):

    print('\tmake_plots(%s)' % p3_selected)

    if len(df) == 0:
        fig_pred = {}
    else:
# ... subset data frames to selected part triple
        df_predict_part = df.copy()
        df_predict_part = df_predict_part[df_predict_part['ticker'] == p3_selected]

        cols_2_keep = ['date', 'ticker', 'orp_code', 'model_type',
                       'value', 'wgt_avg', 'sum_quantity',
                       'price_stability_code', 'ppi']
        df_predict_part = df_predict_part[cols_2_keep]

    # ... some plot element characteristics

        marker_props = {'size' : 8,
                        'opacity' : 0.5,
                        'line' : {'color' : colors['paper_background20'], 'width' : 1}
                        }


        n_rows = 2
        row_widths = [0.4, 0.6]

        fig_pred = make_subplots(rows = n_rows,
                                 cols = 1,
                                 shared_xaxes = True,
                                 vertical_spacing = 0.02,
                                 row_width = row_widths)

    # ... material cost - observed and model predictions
        irow = 1

        subsets = ['Input', 'LSTM']

        for s in subsets:
            df_subset = df_predict_part[df_predict_part['model_type'] == s]
            df_subset.reset_index(drop = True, inplace = True)
            fig_pred.add_trace(go.Scatter(x = df_subset['date'],
                                          y = df_subset['value'],
                                          name = s,
                                          mode = 'markers',
                                          marker = marker_props), row = irow, col = 1)
        irow += 1

        # ... wgt_avg
        df_subset = df_predict_part[df_predict_part['model_type'] == 'Input']
        df_subset.reset_index(drop = True, inplace = True)
        fig_pred.add_trace(go.Scatter(x = df_subset['date'],
                                      y = df_subset['wgt_avg'],
                                      name = 'PA wgtd avg cost',
                                      mode = 'markers',
                                      marker = marker_props), row = irow, col = 1)
        irow += 1

        plot_height = n_rows * 250 + 200
        plot_width = 900

        # fig_pred.add_annotation(
        #         font=dict(
        #             color=colors['text']
        #             ),
        #         align="center",
        #         bordercolor=colors['paper_background'],
        #         borderwidth=2,
        #         borderpad=4,
        #         bgcolor=colors['plot_background'],
        #         opacity=0.8
        #         )

        fig_pred.update_layout(
                            height = plot_height,
                            width = plot_width,
                            title_text = p3_selected,
                            font_color = colors['text'],
                            plot_bgcolor = colors['plot_background'],
                            paper_bgcolor = colors['paper_background']
    #                        template = 'seaborn'
                            )

        fig_pred.update_xaxes(showline=True, linewidth=1, linecolor=colors['plot_grid'])
        fig_pred.update_yaxes(showline=True, linewidth=1, linecolor=colors['plot_grid'])
        fig_pred.update_xaxes(showgrid=True, gridwidth=1, gridcolor=colors['plot_grid'])
        fig_pred.update_yaxes(showgrid=True, gridwidth=1, gridcolor=colors['plot_grid'])

    return fig_pred

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... Linear regression (OLS) plot
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def plot_ols(df, p3_selected, engine_id):

    print('\tmake_ols_plot(%s)' % p3_selected)

    # ... subset data frames to selected part triple
    df_predict_part = df.copy()
    df_predict_part = df_predict_part[df_predict_part['ticker'] == p3_selected]

    cols_2_keep = ['date', 'ticker', 'orp_code', 'model_type',
                   'value', 'wgt_avg', 'sum_quantity',
                   'price_stability_code', 'ppi']
    df_predict_part = df_predict_part[cols_2_keep]

    this_orp = df_predict_part['orp_code'].iloc[0]

    fig_ols = go.Figure()

    marker_props = {'size' : 8,
                    'opacity' : 0.5,
                    'line' : {'color' : colors['paper_background20'], 'width' : 1}
                    }
    marker_props_opaque = {'size' : 12,
                    'opacity' : 1.0,
                    'line' : {'color' : colors['paper_background20'], 'width' : 1}
                    }

    if this_orp >= 600 and this_orp <= 699:
        df_subset = df_predict_part[df_predict_part['model_type'] == 'Input']
        df_subset.reset_index(drop=True, inplace=True)
        fig_ols.add_trace(go.Scatter(x=df_subset['sum_quantity'],
                                      y=df_subset['value'],
                                      name='Observed',
                                      mode='markers',
                                      marker=marker_props
                             ))
        df_sub_ols = df_predict_part[df_predict_part['model_type'] == 'OLS']
        df_sub_ols = df_sub_ols[['date', 'value']]
        df_sub_inp = df_predict_part[df_predict_part['model_type'] == 'Input']
        df_sub_inp = df_sub_inp[['date', 'sum_quantity']]
        df_subset = df_sub_ols.merge(df_sub_inp)
        df_subset.reset_index(drop=True, inplace=True)
        fig_ols.add_trace(go.Scatter(x=df_subset['sum_quantity'],
                                      y=df_subset['value'],
                                      name='OLS',
                                      mode='markers',
                                      marker=marker_props_opaque))
    elif this_orp == 513:
        df_subset = df_predict_part[df_predict_part['model_type'] == 'Input']
        df_subset.reset_index(drop=True, inplace=True)
        fig_ols.add_trace(go.Scatter(x=df_subset['wgt_avg'],
                                     y=df_subset['value'],
                                     name='Observed',
                                     mode='markers',
                                     marker=marker_props,
                                     hovertext=df_subset['date']
                                     ))
        df_sub_ols = df_predict_part[df_predict_part['model_type'] == 'OLS']
        df_sub_ols = df_sub_ols[['date', 'value']]
        df_sub_inp = df_predict_part[df_predict_part['model_type'] == 'Input']
        df_sub_inp = df_sub_inp[['date', 'wgt_avg']]
        df_subset = df_sub_ols.merge(df_sub_inp)
        df_subset.reset_index(drop=True, inplace=True)
        fig_ols.add_trace(go.Scatter(x=df_subset['wgt_avg'],
                                     y=df_subset['value'],
                                     name='OLS',
                                     mode='markers',
                                     marker=marker_props_opaque,
                                     hovertext=df_subset['date']))
    else:
        fig_ols.add_trace(go.Scatter(x = [0,1,0,1], y = [0,1,1,0]))

    fig_ols.update_layout(
        height = 650,
        title_text= engine_id + ' - ' + p3_selected,
        font_color=colors['text'],
        plot_bgcolor=colors['plot_background'],
        paper_bgcolor=colors['paper_background']
        #                        template = 'seaborn'
    )

    fig_ols.update_xaxes(showline=True, linewidth=1, linecolor=colors['plot_grid'])
    fig_ols.update_yaxes(showline=True, linewidth=1, linecolor=colors['plot_grid'])
    fig_ols.update_xaxes(showgrid=True, gridwidth=1, gridcolor=colors['plot_grid'])
    fig_ols.update_yaxes(showgrid=True, gridwidth=1, gridcolor=colors['plot_grid'])

    return fig_ols

