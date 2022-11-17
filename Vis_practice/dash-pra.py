# %%
###########################################1116###############################################
# html.img insert image
import dash as dash
from dash import dcc 
from dash import html
from dash.dependencies import Input, Output 
import plotly.express as px 
import pandas as pd
import numpy as np
import pandas_datareader as web
import datetime
from datetime import date
#%%
company = ['AAPL','YELP','TSLA','IBM','MSFT','GOOG','FRD','ORCL']
df = web.DataReader(company[0], data_source = 'yahoo',
                    start='2000-01-01',
                    end=date.today())#extract features

#%%
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


my_app = dash.Dash('1116', external_stylesheets = external_stylesheets)
my_app.layout = html.Div([
    
    html.H4('Pick a company(s)'),
    dcc.Dropdown(id = 'company',
                 options = [{'label':i, 'value':i} for i in company],
                 multi = True,
                 placeholder = 'Select Symbol...'
                 ),
    
    html.Br(),
    html.Br(),
    
    html.H4('Pick a feature'),
    dcc.Dropdown(id = 'feature',
                 options = [{'label':i, 'value':i} for i in df.columns],
                 multi = False,
                 placeholder = 'Select Symbol...'
                 ),
    
    html.Br(),
    html.Br(),
    
    html.H4('Pick the date range'),
    dcc.DatePickerRange(id = 'start',
                        min_date_allowed = date(2000, 1, 1),
                        max_date_allowed = date.today(),
                        ),
    html.Br(),
    html.Br(),
    dcc.Graph(id = 'graph')
    
], style={'width':'30%'})# change the size of the app

@my_app.callback(
    Output(component_id='graph', component_property='figure'),
    [Input(component_id='company', component_property='value'),
     Input(component_id='feature', component_property='value'),
     Input(component_id='start', component_property='start_date'),
     Input(component_id='start', component_property='end_date'),
     ]
)

def update(a1,a2,a3,a4):
    df1 = pd.DataFrame()
    
    for i in a1:
        df = web.DataReader(i, 
                            data_source = 'yahoo',#######
                            start = a3,
                            end = a4,
                            )
        
        df = df[a2]
        df1 = pd.concat([df1,df], axis=1)
        
    df1.columns = a1
    
    if a2!= 'Volume':
        fig = px.line(df1,
                      y = a1,
                      width = 800,
                         height = 500, # change the size of the figure
                      title = f'The {a2} price of the {a1} company(s)',
                      labels ={'index':'date', 'value':'USD{$}'})
    else:
        fig = px.line(df1,
                  y = a1,
                  width = 800,
                  height = 500,
                  title = f'The {a2} of the {a1} company(s)',
                  labels ={'index':'date', 'value':'QTY'})
    
    return fig
    
my_app.run_server(
        port=8004,
        host='0.0.0.0')