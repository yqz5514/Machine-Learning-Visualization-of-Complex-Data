#%%

import dash as dash
from dash import dcc 
from dash import html
from dash.dependencies import Input, Output 
import plotly.express as px 
import pandas as pd
import numpy as np

#%%
df = pd.read_csv('clean_data.csv')
df.head()
#%%
df = pd.read_csv('https://drive.google.com/file/d/1QX2RGe3xGa8ds3TXqqJ41MmmeOExrMq_/view?usp=share_link')
 #%%
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

my_app = dash.Dash('Q3', external_stylesheets=external_stylesheets)
my_app.layout = html.Div([
                          html.H1('Caculator', style={'textAlign': 'center'}),
                          html.Br(),
                          html.Br(),
                          
                          
                          html.P('Please enter he first number'),
                          html.P('input:'),
                          dcc.Input(id='input1',type='number'),
                          
                          html.Br(),
                          html.Br(),
                          
                          
                          dcc.Dropdown(id='my_drop',options=[
                              {'label': '+','value':'+'},
                              {'label': '-','value':'-'},
                              {'label': '*','value':'*'},
                              {'label': '/','value':'/'},
                              {'label': 'log','value':'log'},
                              {'label': 'square','value':'square'},
                              {'label': 'square_root','value':'square_root'},
                              ],clearable = False),
                          
                          html.Br(),
                          html.Br(),
                          
                          html.P('Please enter he second number'),
                          html.P('input:'),
                          dcc.Input(id='Input2',type='number'),
                        
                          html.Br(),
                          html.Br(),
                          html.Div(id='my_out')
                          ])

                    



@my_app.callback(Output(component_id='my_out', component_property='children'),
                 [Input(component_id='input1', component_property='value'),
                  Input(component_id='Input2', component_property='value'),
                  Input(component_id='my_drop', component_property='value'),])


def update_output(a1,a2,a3):
    if a3 == '+':
        # n = a1+a2
        return f'The output value is {a1+a2}'
    elif a3 == '-':
        n = np.subtract(a1,a2)
        return f'The output value is {n}'
    elif a3 == '*':
        n = np.multiply(a1,a2)
        return f'The output value is {n}'
    elif a3 == '/':
        n = np.divide(a1,a2)
        return f'The output value is {n}'
    elif a3 == 'log':
        n = a1* np.log(a2)#???????????????
        return f'The output value is {n}'
    elif a3 == 'square':
        #n = a1**a2
        return f'The output value is {a1**a2}'
    elif a3 == 'square_root':
        # n = a1**(1/a2)
        return f'The output value is {a1**(1/a2)}'


my_app.run_server(
    port=8024,
    host='0.0.0.0'
)