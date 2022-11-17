#%%
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
import matplotlib.pyplot as plt
#%%
tip = px.data.tips()
tip
# %%
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

my_app = dash.Dash('Quiz1', external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css'])

#phase three
# Divioison is a section of app
my_app.layout=html.Div([
    html.H4('Select the predictor'),
    dcc.Dropdown(id = 'predictor',
                 options = [{'label':'total_bill', 'value':'total_bill'},
                            {'label':'tip', 'value':'tip'}],
                 multi = False,
                 value='total_bill', 
                 clearable=False,
                 ),
    
    html.Br(),
    html.Br(),
    
    html.H4('Select the legend'),
    dcc.Dropdown(id = 'legend',
                 options = [{'label':'day', 'value':'day'},
                            {'label':'time', 'value':'time'},
                            {'label':'sex', 'value':'sex'},
                            {'label':'smoker', 'value':'smoker'},
                            {'label':'size', 'value':'size'}],
                 multi = False, 
                 value = 'day',
                 clearable=False,
                 
                 ),
    html.Br(),
    html.Br(),
    dcc.Graph(id = 'graph')
    
    
])

@my_app.callback(
    Output(component_id='graph', component_property='figure'),
    [Input(component_id='predictor', component_property='value'),
     Input(component_id='legend', component_property='value')
     ]
)

def update(a1,a2):
    
    df = px.data.tips()
    fig = px.pie(df, values=a1, names=a2)
    #fig.show()
    
    return fig
           
           
        
my_app.run_server(
        port=8021,
        host='0.0.0.0')
    

# %%
