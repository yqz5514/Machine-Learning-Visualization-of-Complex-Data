import pandas as pd
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly
import numpy as np
# import seaborn as sns
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
import datetime

# %%
df_clean = pd.read_csv('clean_data.csv')
df_final = pd.read_csv('Final_data.csv')

# %%
df_clean.columns
features = ['ARRIVAL_DELAY', 'SCHEDULED_TIME', 'ELAPSED_TIME']
num_f = ['ARRIVAL_DELAY', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'DEPARTURE_DELAY']
# %%
import numpy as np
from scipy.stats import shapiro
from scipy.stats import kstest
from scipy.stats import normaltest

# %%
# outlier
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

my_app = dash.Dash('My App', external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

# phase three
# Divioison is a section of app
my_app.layout = html.Div([

    html.H1(f'Outlier detection of dependent variable',
            style={'fontSize': 20, 'font-weight': 'bold', 'textAlign': 'left'}),

    html.Br(),
    html.Br(),
    html.P("Options:"),
    dcc.RadioItems(
        id='r1',
        options=['Before outlier removal', 'After outlier removal'],
        value='',
        style={'fontSize': 15, 'font-weight': 'bold'},
        inline=True
    ),

    dcc.Graph(id='graph1'),
    html.Br(),
    html.Br(),
    html.P('Outlier_Boundary_Value'),
    html.Div(id='out1')

])


@my_app.callback(
    Output(component_id='graph1', component_property='figure'),
    Output(component_id='out1', component_property='children'),
    [Input(component_id='r1', component_property='value'), ]
)
def display(a1):
    if a1 == 'Before outlier removal':

        fig = px.box(df_final, y="DEPARTURE_DELAY",
                     width=800, height=400, )
        # w_clean ='The low limit of dependent variable is -28.5, the high limit is 31.5'
        return fig, (html.Br(),
                     html.P(f'The low limit of dependent variable is -28.5, the high limit is 31.5',
                            style={'fontSize': 20, 'font-weight': 'bold', 'color': 'coral'}))
    elif a1 == 'After outlier removal':

        fig = px.box(df_clean, y="DEPARTURE_DELAY",
                     width=800, height=400, )
        # clean = 'All outliers has been removed'
        return fig, (html.Br(),
                     html.P(f'Outlier between -28.5 and 31.5 all removed',
                            style={'fontSize': 20, 'font-weight': 'bold', 'color': 'coral'}))


my_app.run_server(
    port=8015,
    host='0.0.0.0')
