
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
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State

df_clean = pd.read_csv('clean_data.csv')
# df_final = pd.read_csv('Final_data.csv')

# %%
# df_clean.columns
# features = ['ARRIVAL_DELAY', 'SCHEDULED_TIME', 'ELAPSED_TIME']
# num_f = ['ARRIVAL_DELAY', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'DEPARTURE_DELAY']

missing_df = df_clean.isnull().sum(axis=0).reset_index()
missing_df.columns = ['variable', 'missing values']
missing_df['filling factor (%)']=(df_clean.shape[0]-missing_df['missing values'])/df_clean.shape[0]*100
missing_df.sort_values('filling factor (%)').reset_index(drop = True)

df_decribe = df_clean.describe()
df_5 = df_clean[0:5]
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

my_app = dash.Dash('my-app', external_stylesheets=external_stylesheets)

my_app.layout = html.Div([
    html.H1('Data Description',
            style={'fontSize':40, 'textAlign':'center'}),
    html.Br(),
    html.Br(),
    html.P('      Data that will be used in this project is originally from 2015 Flight Delays and Cancellations.'
           'This data includes the on-time performance of domestic flights operated by large air carriers, and summary'
           'information on the number of on-time, delayed, canceled, and diverted flights.                            '
           '      The original data has over 500K records and 31 features, because of the large amount of data, and   '
           'unnecessary features, after data preprocessing, I decide to use data from January to February with 7 main '
           'features, including two categorical features and four numerical features. The final data have the number of'
           'total records more than 60K. The dependent variable is departure delay and all others are independent variables.',
           style = {'textAlign':'left'}),
    html.Br(),
    html.Br(),
    html.Br(),
    html.P('The top 5 rows of cleaned dataset'),
    html.Br(),
    dash_table.DataTable(df_5.to_dict('records'),
                         [{"name": i, "id": i} for i in df_5.columns]),
    html.Br(),
    html.Br(),
    html.P('The missing value table of cleaned data'),
    html.Br(),
    dash_table.DataTable(missing_df.to_dict('records'),
                         [{"name": i, "id": i} for i in missing_df.columns]),
    html.Br(),
    html.Br(),
    html.P('The describe statistic of cleaned data'),
    html.Br(),
    dash_table.DataTable(df_decribe.to_dict('records'),
                         [{"name": i, "id": i} for i in df_decribe.columns]),
    html.Br(),
    html.Br(),


    ])
my_app.run_server(
        port=8070,
        host='0.0.0.0')