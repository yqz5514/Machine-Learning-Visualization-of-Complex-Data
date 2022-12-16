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
import datetime
#%%
df_clean = pd.read_csv('clean_data.csv')
df_final = pd.read_csv('Final_data.csv')

#%%
df_clean.columns
features = ['ARRIVAL_DELAY', 'SCHEDULED_TIME', 'ELAPSED_TIME']
num_f = ['ARRIVAL_DELAY', 'SCHEDULED_TIME', 'ELAPSED_TIME','DEPARTURE_DELAY']
#%%
import numpy as np
from scipy.stats import shapiro
from scipy.stats import kstest
from scipy.stats import normaltest

def shapiro_test(x, title):
    stats, p = shapiro(x)
    print('=' * 50)
    print(f'Shapiro test : {title} dataset : statistics = {stats:.2f} p-vlaue of ={p:.2f}' )
    alpha = 0.01
    if p > alpha :
        print(f'Shapiro test: {title} dataset is Normal')
    else:
        print(f'Shapiro test: {title} dataset is NOT Normal')
    print('=' * 50)
#%%

fig_qq = plt.figure(figsize=(10, 16))
for i in range(1, 5):
    ax = fig_qq.add_subplot(2,2,i)
    sm.graphics.qqplot(df_clean[num_f[i - 1]], line='s', ax=ax)
    plt.title(f'Q-Q Plot of {num_f[i - 1]}')
plotly_fig = mpl_to_plotly(fig_qq)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

my_app = dash.Dash('My App', external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

# phase three
# Divioison is a section of app
my_app.layout = html.Div([
    html.Br(),
    html.Br(),
    html.P('Normality Test - Plot', style={'fontSize': 20, 'font-weight': 'bold', 'textAlign': 'left'}),


    dcc.Dropdown(id = 'drop_norm',
                 options = [
                   {'label':'Histogram','value':'Histogram'},
                   {'label':'QQ Plot','value':'QQ Plot'},
                   ],
                 value = 'Histogram',
                 clearable = False,
                 style={'fontSize': 15, 'font-weight': 'bold'}),
    html.Br(),
    dcc.Graph(id='graph-normality'),
    html.Br(),


])


@my_app.callback(
    Output(component_id='graph-normality', component_property='figure'),
    #Output(component_id='out3', component_property='children'),
    [Input(component_id='drop_norm', component_property='value')]
)
def update_normal_plot(a1):
    if a1 == 'Histogram':


        fig_normal_hist = make_subplots(rows=2,
                                        cols=2,
                                        subplot_titles=("Distribution of Departure_delay",
                                                        "Distribution of Arrival_delay",
                                                        "Distribution of Schedule_time",
                                                        "Distribution of Elapsed_time"))
        fig_normal_hist.add_trace(go.Histogram(x=df_clean['DEPARTURE_DELAY'], name = 'DEPARTURE_DELAY'), row=1, col=1)
        fig_normal_hist.add_trace(go.Histogram(x=df_clean['ARRIVAL_DELAY'],name = 'ARRIVAL_DELAY'), row=1, col=2)
        fig_normal_hist.add_trace(go.Histogram(x=df_clean['SCHEDULED_TIME'], name='SCHEDULED_TIME'), row=2, col=1)
        fig_normal_hist.add_trace(go.Histogram(x=df_clean['ELAPSED_TIME'],name='ELAPSED_TIME'), row=2, col=2)

        fig_normal_hist.update_layout(title_text='Subplots of Histogram for Numerical Features',height=600, width=1200,)
        return fig_normal_hist

    else:

        return plotly_fig



my_app.run_server(
    port=8010,
    host='0.0.0.0')

# fig_normal_hist = make_subplots(rows=2, cols=2, y_title='Count')
# fig_normal_hist.add_trace(go.Histogram(x=df_clean['DEPARTURE_DELAY']), row=1, col=1)
# fig_normal_hist.add_trace(go.Histogram(x=df_clean['ARRIVAL_DELAY']), row=1, col=2)
# fig_normal_hist.add_trace(go.Histogram(x=df_clean['SCHEDULED_TIME']), row=2, col=1)
# fig_normal_hist.add_trace(go.Histogram(x=df_clean['ELAPSED_TIME']), row=2, col=2)
#
# fig_normal_hist.update_layout(title_text='Subplots of Histogram for Numerical Features')
# fig_normal_hist.write_html('first_figure.html', auto_open=True)