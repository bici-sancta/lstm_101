
# ... source :
# ... https://raw.githubusercontent.com/Coding-with-Adam/Dash-by-Plotly/master/Other/Dash_Introduction/intro.py
# ... https://www.youtube.com/watch?v=hSPmj7mK6ng

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

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import json

import dash_plot as dp
from style_settings import *

# ...   -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ...   some global values ... that need to be defined locally and dynamically (later)
# ...   -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# ... some initial conditions

# ...   -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ...   initiate app
# ...   -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

app = dash.Dash(__name__)

time_init = time.time()


DOW_30 = [
    'mmm',
    'axp',
    'amgn',
    'aapl',
    'ba',
    'cat',
    'cvx',
    'csco',
    'ko',
    'dis',
    'dow',
    'gs',
    'hd',
    'hon',
    'ibm',
    'intc',
    'jnj',
    'jpm',
    'mcd',
    'mrk',
    'msft',
    'nke',
    'pg',
    'crm',
    'trv',
    'unh',
    'vz',
    'v',
    'wba',
    'wmt']

def lx_ticker_dropdown():

    lx_ticker = []
    for t in DOW_30:
        lx_new = {'label': t, 'value': t}
        lx_ticker.append(lx_new)

    return lx_ticker

def df_plot():
    return pd.DataFrame()

# ...   -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ...   App layout
# ...   -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

app.layout = html.Div(
    style = {'backgroundColor' : colors['big_background']},
    children = 
        [
            html.H1("lstm forecasts dashboard", style = text_style),
            dcc.Dropdown(id="select_ticker",
                     options=lx_ticker_dropdown(),
                     multi=False,
                     value = lx_ticker_dropdown()[0]['value'],
                     style=menu_style
                     ),
            html.Button('Previous', id='prev_button', style = button_style),
            html.Button('Next', id='next_button', style = button_style),
            html.Br(),
            dcc.Tabs(id="tabs", value = 'run_id_input',
            children=
                [
                    dcc.Tab(label='input data',
                        style = tab_style, selected_style = tab_selected_style,
                        children=
                            [
                                html.P("historic", style=text_style),
                                dcc.Graph(id='ticker_plot',
                                          figure={},
                                          config = {})
                            ]),
                    dcc.Tab(label='forecast model results', value = 'forecast',
                        style = tab_style, selected_style = tab_selected_style,
                        children=
                        [
                            html.P("forecasts", style=text_style),
                            dcc.Graph(id='predict_plot',
                                      figure={},
                                      config = {}),
                            html.Br(),
                            dcc.Graph(id='error_plot', figure={})
                        ]),
                ], style = tabs_style)
        ])

# ...   -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ...   Next and Prev buttons - updates dropdown selection, to then update plots
# ...   -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

@app.callback(
    output = Output(component_id = 'select_ticker', component_property = 'value'),
    inputs = [Input(component_id = 'prev_button', component_property = 'n_clicks'),
              Input(component_id = 'next_button', component_property = 'n_clicks')],
    state = [State(component_id = 'select_ticker', component_property = 'value')]
)
def next_requested(prev_click, next_click, this_part_ticker):

    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    print('\tnext_requested(%s, %s' % (button_id, this_part_ticker))

    next_p3 = None

    this_indx = [i for i, d in enumerate(lx_ticker_dropdown()) if this_part_ticker in d.values()][0]

    if button_id == 'prev_button':
        next_indx = this_indx - 1
        next_indx = next_indx if next_indx  >= 0 else len(lx_ticker_dropdown())-1
    elif button_id == 'next_button':
        next_indx = this_indx + 1
        next_indx = next_indx if next_indx < len(lx_ticker_dropdown()) else 0
    else:
        print('unknown button_id in next_requested()')
        next_indx = 0

    next_p3 = lx_ticker_dropdown()[next_indx]['label']

    return next_p3

# ...   -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... update forecast plots
# ...   -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

@app.callback(
    [Output(component_id = 'predict_plot', component_property = 'figure'),
     Output(component_id='predict_plot', component_property='config')],
    [Input(component_id = 'select_ticker', component_property = 'value')],
    [State(component_id = 'select_ticker', component_property = 'options')]
)
def update_predict(p3_selected, p3_options):
    print('\tupdate_predict(%s)' % p3_selected)

    fig_pred = dp.make_plots(df_plot(), p3_selected)
    cnfg_pred = {
        'toImageButtonOptions': {
            'format': 'png',
            'filename': p3_selected + '_chr',
#                'width': 800,
            'scale': 1
        }
    }
    # fig_pred = px.bar(x=['lstm', 'arima', 'ols'], y=[1, 1, 1],
    #                  title='default error plot')
    # cnfg_pred = {}

    print('\t\t ... return plots')
    return fig_pred, cnfg_pred

# ...   -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... update historic time series plots
# ...   -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

@app.callback(
    [Output(component_id = 'ticker_plot', component_property = 'figure'),
     Output(component_id='ticker_plot', component_property='config')],
    [Input(component_id = 'select_ticker', component_property = 'value')],
    [State(component_id = 'select_ticker', component_property = 'options')]
)
def update_ticker_plot(p3_selected, p3_options):
    print('\tupdate_ticker_plot(%s)' % p3_selected)

    fig_tckr = dp.plot_ticker(p3_selected)
    cnfg_tckr = {
        'toImageButtonOptions': {
            'format': 'png',
            'filename': p3_selected + '_chr',
#                'width': 800,
            'scale': 1
        }
    }
    # fig_pred = px.bar(x=['lstm', 'arima', 'ols'], y=[1, 1, 1],
    #                  title='default error plot')
    # cnfg_pred = {}

    print('\t\t ... return plots')
    return fig_tckr, cnfg_tckr

# ...   -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... main
# ...   -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def main(server_port = 8888, debug_bool = True):
    app.run_server(port=int(server_port), debug=debug_bool)


if __name__ == '__main__':
    print('begin main() ...')

    port_val = None
    debug_bool = None

    if port_val is None: port_val = 8888
    if debug_bool is None: debug_bool = True

    main(port_val, debug_bool)


# ...   -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ...   eof
# ...   -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-