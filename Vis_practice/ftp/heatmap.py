import pandas as pd
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly
import numpy as np
#import seaborn as sns
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
#%%
df_clean = pd.read_csv('clean_data.csv')
df_final = pd.read_csv('Final_data.csv')
num_f = ['ARRIVAL_DELAY', 'SCHEDULED_TIME', 'ELAPSED_TIME','DEPARTURE_DELAY']

fig_pearson = px.imshow(df_clean[num_f].corr(),
                        text_auto=True,
                        color_continuous_scale='Blues',
                        title='Heatmap of Pearson Correlation Coefficient Matrix of Numerical Features',
                        width=800,
                        height=800
                        )
#%%

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

my_app = dash.Dash('My App', external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

my_app.layout = html.Div([
                html.Br(),
                html.Br(),
                html.Div([
                    dcc.Graph(id='graph-pearson', figure=fig_pearson),
                    html.Br(),
                ]),
                html.Br(),
                html.Div([
                    html.Button("Download Image", id="btn_image",
                                style={'backgroundColor' : '#3aaab2','color':'yellow','font-size': '20px'}),
                    dcc.Download(id="download-image")
                        ]),
])
@my_app.callback(
    Output("download-image", "data"),
    Input("btn_image", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    return dcc.send_file(r'/Users/yaxin/Downloads/pic.png')

my_app.run_server(
    port=8011,
    host='0.0.0.0')