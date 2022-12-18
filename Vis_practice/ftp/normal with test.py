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
#%%

fig_qq = plt.figure()
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
    html.P('Normality Test Analysis', style={'fontSize': 20, 'font-weight': 'bold', 'textAlign': 'center'}),

    html.Div([
        html.P('Plots of Normality Test', style={'fontSize': 20,'textAlign': 'left'}),

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
        #html.Div(id='drop-out', style={'fontSize': 20, 'font-weight': 'bold'}),
        html.Br(),], className='six columns'),
    html.Br(),
    html.Br(),
    html.Br(),


    html.Div([html.Div([
                html.P('Shapiro Test', style={'fontSize': 20, 'font-weight': 'bold'}),
                html.Br(),
                html.Button('Calculate Shapiro Test Result',
                            id='button-st', n_clicks=0, style={'color':'yellow','background-color':'#3aaab2','fontSize': 12, 'font-weight': 'bold','height': '50px','width': '400px'}),
                html.Div(id='st-out'),
                ],
                className='six columns')


]),
])

@my_app.callback(
    Output(component_id='graph-normality', component_property='figure'),
    #Output(component_id='drop-out', component_property='children'),
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

        fig_normal_hist.update_layout(title_text='Subplots of Histogram for Numerical Features')
        return fig_normal_hist

    else:

        return plotly_fig

@my_app.callback(
    Output('st-out', 'children'),
    [Input('button-st', 'n_clicks')]
)
def update_st(n_clicks):

    if n_clicks > 0:
        return (html.Br(),
                html.P(f"Shapiro test of DEPARTURE_DELAY: statistics = {shapiro(df_clean['DEPARTURE_DELAY'])[0]}",
                       style={'fontSize': 20, 'font-weight': 'bold','color':'coral'}),
                html.Br(),
                html.P(f"p-value = {shapiro(df_clean['DEPARTURE_DELAY'])[1]}",
                       style={'fontSize': 20, 'font-weight': 'bold','color':'coral'}),

                html.Br(),
                html.P(f"Shapiro test of ARRIVAL_DELAY: statistics = {shapiro(df_clean['ARRIVAL_DELAY'])[0]}",
                       style={'fontSize': 20, 'font-weight': 'bold', 'color': 'coral'}),
                html.Br(),
                html.P(f"p-value = {shapiro(df_clean['ARRIVAL_DELAY'])[1]}",
                       style={'fontSize': 20, 'font-weight': 'bold', 'color': 'coral'}),

                html.Br(),
                html.P(f"Shapiro test of SCHEDULED_TIME: statistics = {shapiro(df_clean['SCHEDULED_TIME'])[0]}",
                       style={'fontSize': 20, 'font-weight': 'bold', 'color': 'coral'}),
                html.Br(),
                html.P(f"p-value = {shapiro(df_clean['SCHEDULED_TIME'])[1]}",
                       style={'fontSize': 20, 'font-weight': 'bold', 'color': 'coral'}),

                html.Br(),
                html.P(f"Shapiro test of ELAPSED_TIME: statistics = {shapiro(df_clean['ELAPSED_TIME'])[0]}",
                       style={'fontSize': 20, 'font-weight': 'bold', 'color': 'coral'}),
                html.Br(),
                html.P(f"p-value = {shapiro(df_clean['ELAPSED_TIME'])[1]}",
                       style={'fontSize': 20, 'font-weight': 'bold', 'color': 'coral'}),
                html.Br(),
                html.P(f"If p-value less than 0.05, indicate that the data is more likely be not normally distributed.",
                       style={'fontSize': 20, 'font-weight': 'bold', 'color': 'blue'}),
                html.Br(),
                html.P(f"If p-value more than 0.05, indicate that the data is more likely be normally distributed.",
                       style={'fontSize': 20, 'font-weight': 'bold', 'color': 'blue'}),

                )


my_app.run_server(
    port=8001,
    host='0.0.0.0')

# fig_normal_hist = make_subplots(rows=2, cols=2, y_title='Count')
# fig_normal_hist.add_trace(go.Histogram(x=df_clean['DEPARTURE_DELAY']), row=1, col=1)
# fig_normal_hist.add_trace(go.Histogram(x=df_clean['ARRIVAL_DELAY']), row=1, col=2)
# fig_normal_hist.add_trace(go.Histogram(x=df_clean['SCHEDULED_TIME']), row=2, col=1)
# fig_normal_hist.add_trace(go.Histogram(x=df_clean['ELAPSED_TIME']), row=2, col=2)
#
# fig_normal_hist.update_layout(title_text='Subplots of Histogram for Numerical Features')
# fig_normal_hist.write_html('first_figure.html', auto_open=True)