#%%

import dash as dash
from dash import dcc 
from dash import html
from dash.dependencies import Input, Output 
import plotly.express as px 
import pandas as pd
import numpy as np
from scipy.fft import fft
import scipy
#%%
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

my_app = dash.Dash('HW5', external_stylesheets=external_stylesheets)
my_app.layout = html.Div([html.H1('HW5', style={'textAlign': 'center'}),
                          html.Br(),
                          dcc.Tabs(id='hw-questions',
                          children=[
                          dcc.Tab(label='Question 1', value='q1'),
                          dcc.Tab(label='Question 2', value='q2'),
                          
                          ]),

                          html.Div(id='layout')])
#####################3
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

question1_layout= html.Div([
                        
                          
                          
                          html.P('Please enter the number of sinusoidal cycle'),
                          dcc.Input(id='input1',
                                    type='number',
                                    value = 4),
                          html.Br(),
                          html.P('Please enter the mean of white noise'),
                          dcc.Input(id='input2',
                                    type='number',
                                    value = 1),
                          html.Br(),
                          html.P('Please enter the standard deviation of white noise'),
                          dcc.Input(id='input3',
                                    type='number',
                                    value = 1),
                          html.Br(),
                          html.P('Please enter the number of sample'),
                          dcc.Input(id='input4',
                                    type='number',
                                    value = 1000),
                        
                          html.Br(),
                          dcc.Graph(id = 'graph1'),
                           html.Br(),
                          html.P('The fast foutier transform of above data'),
                          dcc.Graph(id = 'graph2'),
])
                        
                         
# html.Div([ ### FIGURES Divs
#             html.Div([
#                 dcc.Graph(id = 'graph1'),
                
#             ], className = 'six columns'),
#             html.Div([
#                 html.P('The fast foutier transform of above data'),
#                 dcc.Graph(id = 'graph2'),
#             ], className = 'six columns')
#         ], className = 'row')
# ])
                    



@my_app.callback([Output(component_id='graph1', component_property='figure'),
                 Output(component_id='graph2', component_property='figure')],
                 [Input(component_id='input1', component_property='value'),
                  Input(component_id='input2', component_property='value'),
                  Input(component_id='input3', component_property='value'),
                  Input(component_id='input4', component_property='value')])

def update_q1(a1,a2,a3,a4):
    
    x = np.linspace(-np.pi,np.pi,a4)
    noise = np.random.normal(a2, a3, a4)
    #fx = np.sin(2 * np.pi * a1 * x /a4) + noise
    
    fx = np.sin(a1*x) + noise
    
    q1 = pd.DataFrame({'x_v': x, 'y_v': fx})
    fig1 = px.line(q1,
              x = q1['x_v'],
              y = q1['y_v'],
              width = 600, height = 300,
              #labels = {'value': 'USD($)'}
              )
    
    #fig2
    
    y = fft(fx)
    #y_1 = np.abs(y)
    
    q11 = pd.DataFrame({'x_v':x, 'y_v': np.abs(y)})
    fig2 = px.line(q11,
              x = q11['x_v'],
              y = q11['y_v'],
              width = 600, height = 300,
              #labels = {'value': 'USD($)'}
              )
                   

    return  fig1, fig2


# question2
steps= 0.001
image_path = 'assets/hw5_11.png'
image_path2 = 'assets/hw5_1.JPG'
# need to create assets folder at local app and add local img to the folder 
# lack of the above step will cause error
question2_layout = html.Div([
    html.Img(src=image_path,
             style={'width': '30%', 
                    'display': 'inline-block', 
                    'vertical-align': 'middle'}),
    html.Br(),
    html.Img(src=image_path2,
             style={'width': '30%', 
                    'display': 'inline-block', 
                    'vertical-align': 'middle'}),
    html.Br(),
    dcc.Graph(id = 'g_q2'),
    html.Br(),
    html.P('b1,1'),
    dcc.Slider(id = 'a',
               min = -10, # can not be negative number 
               max = 10,
               value = 4,
               step = steps,# samples can not be float
               marks = {i:f'{i}' for i in range(-10,10)}),
    html.P('b1,2'),
    dcc.Slider(id = 'b',
               min = -10, # can not be negative number 
               max = 10,
               value = 3,
               step = steps,# samples can not be float
               marks = {i:f'{i}' for i in range(-10,10)}),
    html.P('w1,1,1'),
    dcc.Slider(id = 'c',
               min = -10, # can not be negative number 
               max = 10,
               value = 5,
               step = steps,# samples can not be float
               marks = {i:f'{i}' for i in range(-10,10)}),
    html.P('w1,1,2'),
    dcc.Slider(id = 'd',
               min = -10, # can not be negative number 
               max = 10,
               value = 7,
               step = steps,# samples can not be float
               marks = {i:f'{i}' for i in range(-10,10)}),
    html.P('b2,1'),
    dcc.Slider(id = 'e',
               min = -10, # can not be negative number 
               max = 10,
               value = 4,
               step = steps,# samples can not be float
               marks = {i:f'{i}' for i in range(-10,10)}),
    html.P('w2,1,1'),
    dcc.Slider(id = 'f',
               min = -10, # can not be negative number 
               max = 10,
               value = 6,
               step = steps,# samples can not be float
               marks = {i:f'{i}' for i in range(-10,10)}),
    html.P('w2,1,2'),
    dcc.Slider(id = 'g',
               min = -10, # can not be negative number 
               max = 10,
               value = 4,
               step = steps,# samples can not be float
               marks = {i:f'{i}' for i in range(-10,10)}),
    
    
    
])
@my_app.callback(Output(component_id='g_q2', component_property='figure'),
                 [Input(component_id='a', component_property='value'),
                  Input(component_id='b', component_property='value'),
                  Input(component_id='c', component_property='value'),
                  Input(component_id='d', component_property='value'),
                  Input(component_id='e', component_property='value'),
                  Input(component_id='f', component_property='value'),
                  Input(component_id='g', component_property='value')])
                 
def update_nn(a1,a2,a3,a4,a5,a6,a7):
    p = np.linspace(-5, 5, 1000)
    a11 = 1/(1+np.exp(-(p*a3+a1)))
    a12 = 1/(1+np.exp(-(p*a4+a2)))
    a2 = a6*a11+a7*a12+a5
    q2 = pd.DataFrame({'x_v': p, 'y_v': a2})
    fig = px.line(q2,
              x = q2['x_v'],
              y = q2['y_v'],
              width = 1400, height = 700,
              #labels = {'value': 'USD($)'}
              )
    return fig

@my_app.callback(Output(component_id='layout', component_property='children'),
                 [Input(component_id='hw-questions', component_property='value')])

def update_layout(ques):
    if ques == 'q1':
        return question1_layout
    elif ques == 'q2':
        return question2_layout
    


my_app.run_server(
        port=8071,
        host='0.0.0.0')
    
                 

# %%
