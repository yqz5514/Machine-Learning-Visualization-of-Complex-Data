# sumamry

# From the above thing

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



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

my_app = dash.Dash('my-app', external_stylesheets=external_stylesheets)

my_app.layout = html.Div([
    # header box

    html.H1('Summary and Recommendations',
            style={'fontSize':40, 'textAlign':'center'}),

    html.Br(),
    html.Br(),
    html.H4('From this '),


])


my_app.run_server(
        port=8080,
        host='0.0.0.0')