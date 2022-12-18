# only refrence
# textarea for comments from user.
# grate notes

import pandas as pd
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly
import numpy as np
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy import linalg as LA
import statsmodels.api as sm
from scipy import stats
from scipy.stats import shapiro
from scipy.stats import normaltest
import dash as dash
import dash_daq as daq
from dash import dcc, html
from dash.dependencies import Input, Output, State

folder = r'/Users/yaxin/Documents/input_com/data'
def text_ar_val(file='Textbox1.txt'):
    with open(folder + fr'/{file}', 'r' ) as f:
        return f.read()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

my_app = dash.Dash('my-app', external_stylesheets=external_stylesheets)

my_app.layout = html.Div([
    # header box
    html.Div([
    html.H1('Reference',style={'fontSize':40, 'textAlign':'center'}),
    html.Br(),
    html.Br(),
    html.P('https://www.kaggle.com/datasets/usdot/flight-delays?select=flights.csv',
        style={'fontSize': 12, 'font-weight': 'bold','color':'blue'}),
    html.P('https://plotly.com/python/legend/',
               style={'fontSize': 12, 'font-weight': 'bold', 'color': 'blue'}),
    html.P('https://plotly.com/python/plotly-express/',
               style={'fontSize': 12, 'font-weight': 'bold', 'color': 'blue'}),
    html.P('https://stackoverflow.com/questions/62045601/update-the-plotly-dash-dcc-textarea-value-from-what-user-inputs',
               style={'fontSize': 12, 'font-weight': 'bold', 'color': 'blue'}),
    html.P('https://plotly.com/python/plotly-express/',
               style={'fontSize': 12, 'font-weight': 'bold', 'color': 'blue'}),
    html.P('https://dash.plotly.com/dash-core-components/upload',
               style={'fontSize': 12, 'font-weight': 'bold', 'color': 'blue'}),
    html.P('https://dash.plotly.com/dash-core-components/textarea',
               style={'fontSize': 12, 'font-weight': 'bold', 'color': 'blue'}),
    html.P('https://dash.plotly.com/dash-core-components/radioitems',
               style={'fontSize': 12, 'font-weight': 'bold', 'color': 'blue'}),
    html.P('https://dash.plotly.com/datatable',
               style={'fontSize': 12, 'font-weight': 'bold', 'color': 'blue'}),
    html.P('https://dash.plotly.com/dash-html-components/button',
               style={'fontSize': 12, 'font-weight': 'bold', 'color': 'blue'}),
    html.P('https://dash.plotly.com/dash-core-components/download',
               style={'fontSize': 12, 'font-weight': 'bold', 'color': 'blue'}),


    ]),
        html.Br(),
        html.Br(),
    html.Div([
        html.H2('Thanks for using this app:-)',
                style={'color': 'green','fontSize':25, 'textAlign':'center','font-weight': 'bold'})
    ]),
        html.Br(),
        html.Br(),
        html.Br(),

        html.Div([
            html.P('Please leave any comments and suggestion using box below',
                   style={'color': 'black','fontSize':15}),
            html.Br(),
            dcc.Textarea(
            id='textarea1', className='textarea1',
            value=text_ar_val(),
            persistence=True, persistence_type='local'),
        html.Div(daq.StopButton(id='save1', className="button",
                                    buttonText='Save', n_clicks=0)),
        ]),
])

@my_app.callback(Output('textarea1', 'children'),
              [Input('save1', 'n_clicks'), Input('textarea1','value')])
def textbox1(n_clicks, value):
    if n_clicks>0:
        global folder
        with open(folder+r'/TextBox1.txt','w') as file:
            file.write(value)
            return value


my_app.run_server(
        port=8093,
        host='0.0.0.0')