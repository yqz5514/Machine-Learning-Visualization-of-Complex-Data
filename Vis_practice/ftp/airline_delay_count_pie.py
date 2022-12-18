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
from dash import dcc, html,dash_table
from dash.dependencies import Input, Output, State
##preprocessing for airline delay analysis
df_clean = pd.read_csv('clean_data.csv')
df_final = pd.read_csv('Final_data.csv')
df_delay = df_clean[df_clean['DEPARTURE_DELAY']>0]
df_delay_detail = df_final[df_final['DEPARTURE_DELAY']>0]
num_f = ['ARRIVAL_DELAY', 'SCHEDULED_TIME', 'ELAPSED_TIME','DEPARTURE_DELAY']
airlines = ['AS', 'AA', 'US', 'DL', 'UA', 'NK', 'HA', 'B6', 'EV', 'OO', 'F9', 'WN', 'VX', 'MQ']
delay_type = lambda x:((0,1)[x > 5],2)[x > 45]
df_delay_detail['DELAY_LEVEL'] = df_delay_detail['DEPARTURE_DELAY'].apply(delay_type)
fig_air_counts = px.histogram(df_delay_detail,
                              y='AIRLINE',
             color="DELAY_LEVEL",
             barmode='group',
             title="Airline Delay Counts",
             width=800,
             height=800)

def get_stats(group):
    return {'min': group.min(), 'max': group.max(),
            'count': group.count(), 'mean': group.mean()}
air_stats = df_final['DEPARTURE_DELAY'].groupby(df_final['AIRLINE']).apply(get_stats).unstack()
air_stats = air_stats.sort_values('count')
air_stats.reset_index(inplace=True)
#---------------------------------------------------------------------------
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

my_app = dash.Dash('My App', external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

# phase three
# Divioison is a section of app
my_app.layout = html.Div([
    html.Br(),
    html.Br(),
    html.P('Airline Delay Analysis', style={'fontSize': 20, 'font-weight': 'bold', 'textAlign': 'center'}),

    html.Div([
        html.P('Pie Chart of Airline Delay', style={'fontSize': 20,'textAlign': 'left'}),

        dcc.Dropdown(id = 'airline-pie',
                     options=[{'label': i, 'value': i} for i in num_f],
                     value = 'DEPARTURE_DELAY',
                     clearable = False,
                     style={'fontSize': 15, 'font-weight': 'bold'}),
        html.Br(),
        html.Br(),
        dcc.Graph(id='pie-airline-plot'),
        #html.Div(id='drop-out', style={'fontSize': 20, 'font-weight': 'bold'},
        # ),
        html.Br(),], className='six columns'),
    html.Br(),
    html.Br(),
    html.Br(),

        html.Div([
    html.P('The describe stats between departure delay and airlines'),
    html.Br(),
    dash_table.DataTable(air_stats.to_dict('records'),
                         [{"name": i, "id": i} for i in air_stats.columns]),
        html.Br(),
        html.Br(),
        html.P(f'DELAY_LEVEL[0] indicates On time (t < 5 mins)',
               style={'fontSize': 10, 'font-weight': 'bold', 'color': 'blue'}),
        # html.Br(),
        html.P(f'DELAY_LEVEL[1] indicates Small delay (5 < t < 45 mins)',
               style={'fontSize': 10, 'font-weight': 'bold', 'color': 'red'}),
        # html.Br(),
        html.P(f'DELAY_LEVEL[3] indicates Significant delay (t > 45 mins)',
               style={'fontSize': 10, 'font-weight': 'bold', 'color': 'green'}),
                    dcc.Graph(id='air-count', figure=fig_air_counts),
        # html.Br(),
                ] ,className='six columns'),

                html.Br(),
                html.Br(),
                html.Div([
                    html.Button("Download Image", id="btn_image",
                                style={'backgroundColor' : '#88d8c0','color':'white','font-size': '20px'}),
                    dcc.Download(id="download-image1")
                        ],className='six columns'),
])

@my_app.callback(
    Output(component_id='pie-airline-plot', component_property='figure'),
    #Output(component_id='drop-out', component_property='children'),
    [Input(component_id='airline-pie', component_property='value')]
)
def update_pie_plot(a1):
    fig = px.pie(df_delay,
                 values=a1,
                 names='AIRLINE',
                 hole=.3,
                 title='Pie chart of Airlines analysis')
    return fig




@my_app.callback(
        Output("download-image1", "data"),
        Input("btn_image", "n_clicks"),
        prevent_initial_call=True,)

def func(n_clicks):
        return dcc.send_file(r'/Users/yaxin/Downloads/well_labeled_plot.png')


my_app.run_server(
        port=8033,
        host='0.0.0.0' )


