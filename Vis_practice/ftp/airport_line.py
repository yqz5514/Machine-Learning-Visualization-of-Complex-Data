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
from dash import dcc, html
from dash.dependencies import Input, Output, State
#%%
df_clean = pd.read_csv('clean_data.csv')
num_f = ['ARRIVAL_DELAY', 'SCHEDULED_TIME', 'ELAPSED_TIME','DEPARTURE_DELAY']
airlines = ['AS', 'AA', 'US', 'DL', 'UA', 'NK', 'HA', 'B6', 'EV', 'OO', 'F9', 'WN', 'VX', 'MQ']

#----------------------------------------------------------------------------------------------------
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

my_app = dash.Dash('dateboard', external_stylesheets=external_stylesheets)
my_app.layout = html.Div([

    html.H1('Airline Overall Brief Look - Line Plot', style={'fontSize': 20, 'font-weight': 'bold', 'textAlign': 'center'}),
    html.Br(),
    html.Br(),

    html.H3('Pick a airline(s)',style={'fontSize': 15, 'font-weight': 'bold'}),
    dcc.Dropdown(id='air',
                 options=[{'label': i, 'value': i} for i in airlines],
                 multi=True,
                 placeholder='Select Symbol...'
                 ),

    html.Br(),
    html.Br(),

    html.H3('Pick a feature',style={'fontSize': 15, 'font-weight': 'bold'}),
    dcc.Dropdown(id='feature',
                 options=[{'label': i, 'value': i} for i in num_f],
                 multi=False,
                 placeholder='Select Symbol...'
                 ),

    html.Br(),
    html.Br(),


    html.Br(),
    dcc.Graph(id='airline-line'),

], style={'width': '60%'})


@my_app.callback(
    Output(component_id='airline-line', component_property='figure'),
    [Input(component_id='air', component_property='value'),
     Input(component_id='feature', component_property='value'),

     ]
)
def update(a1, a2):

    df1 = pd.DataFrame()
    #df1['datetime'] = df_clean['datetime']

    for i in a1:
        df = df_clean[df_clean['AIRLINE'] == i]
        df = df[a2]
        df1 = pd.concat([df1, df], axis=1)

    df1.columns = a1



    fig = px.line(df1,
                  #x='datetime',
                  y=a1,
                  width=800,
                  height=500,  # change the size of the figure
                  title=f'The {a2} of the {a1} airline(s)',
                  labels={'index': 'count', 'value': 'time'})

    return fig


my_app.run_server(
    port=8013,
    host='0.0.0.0')