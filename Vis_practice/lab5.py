#%%

import dash as dash
from dash import dcc 
from dash import html
from dash.dependencies import Input, Output 
import plotly.express as px 
import pandas as pd
import numpy as np
#%%
# 1 
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/CONVENIENT_global_confirmed_cases.csv'
df = pd.read_csv(url)
df.head(5)
# %%
df.isnull().sum().to_string()
#%%
df.dropna(how ='any', inplace = True)
#%%
df.isnull().sum()

#%%
df.iloc[0:,58:90]
#%%
df['China_sum'] = df.iloc[0:,58:90].astype(float).sum(axis=1)
#%%
df['China_sum']
#%%
##3. Repeat step 2 for the “United Kingdom”.

df.iloc[0:,250:260]
#%%
df['UK_sum'] = df.iloc[0:,250:260].astype(float).sum(axis=1)
#%%
df['UK_sum']

#%%
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

steps = 0.01

my_app = dash.Dash('My App', external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css'])

#phase three
# Divioison is a section of app
my_app.layout=html.Div([
    dcc.Graph(id = 'my_graph'),
    html.H1('COVID global confirmed cased by country'),
    html.P('Country'),
    dcc.Dropdown(id='my_drop',options=[
                              {'label': 'US','value':'US'},
                              {'label': 'UK_sum','value':'UK_sum'},
                              {'label': 'China_sum','value':'China_sum'},
                              {'label': 'Germany','value':'Germany'},
                              {'label': 'Brazil','value':'Brazil'},
                              {'label': 'India','value':'India'},
                              {'label': 'Italy','value':'Italy'},
                              ],clearable = False),
    
])

@my_app.callback(
    Output(component_id = 'my_graph', component_property = 'figure'),
    [Input(component_id = 'my_drop', component_property = 'value'),]
    )


def display(a1):
    
    fig = px.line(df,
              x = df['Country/Region'],
              y = a1,#['US','UK_sum','China_sum','Germany','Brazil','India','Italy'],
              width = 1400, height = 700,
              
              #labels = {'value': 'USD($)'}
              )
    return fig

my_app.run_server(
        port=8070,
        host='0.0.0.0')
#df_11 = df[['Country/Region','US','UK_sum','China_sum','Germany','Brazil','India','Italy']]



#%%
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

steps = 0.01

my_app = dash.Dash('My App', external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css'])

#phase three
# Divioison is a section of app
my_app.layout=html.Div([
    dcc.Graph(id = 'my_graph'),
    html.H1('The quadratic function f(x)=ax2+bx+c'),
                          
    html.P('choose value of a'),
    dcc.Slider(id = 'a',
               min = -3,
               max = 3,
               value = 0,#The value of the input
               step =steps,
               marks = {i:f'{i}' for i in range(-3,3)}),
    html.Br(),
    html.Br(),
  
    
    html.P('choose value of b'),
    dcc.Slider(id = 'b',
               min = -5, # can not be negative number 
               max = 5,
               value = 0,
               step= steps,
               marks = {i:f'{i}' for i in range(-5,5)}),
    
    html.Br(),
    html.Br(),
    
    html.P('choose value of c'),
    dcc.Slider(id = 'c',
               min = -10, # can not be negative number 
               max = 10,
               value = 0,
               step = steps,# samples can not be float
               marks = {i:f'{i}' for i in range(-10,10)}),
    
])

@my_app.callback(
    Output(component_id = 'my_graph', component_property = 'figure'),
    [Input(component_id = 'a', component_property = 'value'),
     Input(component_id = 'b', component_property = 'value'),
     Input(component_id = 'c', component_property = 'value'),]
    )


def display(a1,a2,a3):
    x1 = np.linspace(-2,2,1000)
    y1 = a1 * (x1 **2) + a2*x1 +a3
    q2 = pd.DataFrame({'x_v': x1, 'y_v': y1})
    fig = px.line(q2,
              x = q2['x_v'],
              y = q2['y_v'],
              width = 1400, height = 700,
              #labels = {'value': 'USD($)'}
              )
    return fig

my_app.run_server(
        port=8055,
        host='0.0.0.0')




# %%
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

my_app = dash.Dash('Q3', external_stylesheets=external_stylesheets)
my_app.layout = html.Div([
                          html.H1('Caculator', style={'textAlign': 'center'}),
                          html.Br(),
                          html.Br(),
                          
                          html.H5('Please enter he first number'),
                          
                          html.Br(),
                          
                          html.P('Input:'),
                          dcc.Input(id='input1',type='number'),
                          
                          html.Br(),
                          
                          #html.P('Country')
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
                          
                          html.P('Input:'),
                          dcc.Input(id='Input2',type='number'),
                        
                          html.Br(),
                          html.Div(id='my_out')
                          ])

                    



@my_app.callback(Output(component_id='my_out', component_property='children'),
                 [Input(component_id='input1', component_property='value'),
                  Input(component_id='input2', component_property='value'),])


def update_output(a1,a3,a2):
    if a2 == '+':
        n = a1+a3
        return n
    elif a2 == '-':
        n = a1-a3
        return n
    elif a2 == '*':
        n = a1*a3
        return n
    elif a2 == '/':
        n = a1/a3
        return n
    elif a2 == 'log':
        n = np.loga3(a1)
        return n
    elif a2 == 'square':
        n = a1**a3
        return n
    elif a2 == 'square_root':
        n = a1**(1/a3)
        return n


my_app.run_server(
    port=8023,
    host='0.0.0.0'
)


# %%
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

steps = 1

my_app = dash.Dash('My App', external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css'])

#phase three
# Divioison is a section of app
my_app.layout=html.Div([
    dcc.Graph(id = 'my_graph'),
    html.H1('gaussian distribution.'),
                          
    html.P('Mean'),
    dcc.Slider(id = 'mean',
               min = -2,
               max = 2,
               value = 0,#The value of the input
               step =steps,
               marks = {i:f'{i}' for i in range(-2,2)}),
    html.Br(),
    html.Br(),
  
    
    html.P('Std'),
    dcc.Slider(id = 'std',
               min = 1, # can not be negative number 
               max = 3,
               value = 1,
               step= steps,
               marks = {i:f'{i}' for i in range(0,3)}),
    
    html.Br(),
    html.Br(),
    
    html.P('number of samples'),
    dcc.Slider(id = 'samples',
               min = 100, # can not be negative number 
               max = 10000,
               value = 100,
               step = 500,
               #step = steps,# samples can not be float
               marks = {500:'500',1500:'1500',3000:'3000',4500:'4500', 6000:'6000', 7500:'7500', 9000:'9000',10000:'10000'}),
    
    html.Br(),
    html.Br(),
    html.P('bins'),
    dcc.Dropdown(id = 'bins',
               options = [
                   {'label':20,'value':20},
                   {'label':30,'value':30},
                   {'label':40,'value':40},
                   {'label':50,'value':50},
                   {'label':60,'value':60},
                   {'label':70,'value':70},
                   {'label':80,'value':80},
                   {'label':90,'value':90},
                   {'label':100,'value':100},],
                   value = 20,
                   #step = 10,
                   clearable = False),
    
])

@my_app.callback(
    Output(component_id = 'my_graph', component_property = 'figure'),
    [Input(component_id = 'mean', component_property = 'value'),
    Input(component_id = 'std', component_property = 'value'),
    Input(component_id = 'samples', component_property = 'value'),
    Input(component_id = 'bins', component_property = 'value')]
    )

def display(a1,a2,a3,a4):
    x = np.random.normal(a1, a2, size=a3)
    fig = px.histogram(x = x, nbins=a4, range_x=[-6,6])
    return fig

my_app.run_server(
        port=8075,
        host='0.0.0.0')






# %%
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

my_app = dash.Dash('My App', external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css'])

#phase three
# Divioison is a section of app
my_app.layout=html.Div([
    dcc.Graph(id = 'my_graph'),
   
    html.P('Please enter the polynomial order'),
    dcc.Dropdown(id='my_drop',options=[
                              {'label': 1,'value':1},
                              {'label': 2,'value':2},
                              {'label': 3,'value':3},
                              {'label': 4,'value':4},
                              {'label': 5,'value':5},
                              {'label': 6,'value':6},
                              {'label': 7,'value':7},
                              ],clearable = False),
    
    
])

@my_app.callback(
    Output(component_id = 'my_graph', component_property = 'figure'),
    [Input(component_id = 'my_drop', component_property = 'value'),]
    )


def display(a1):
    
    x2 = np.linspace(-2,2,1000)
    y2 = (x2**a1)
    q5 = pd.DataFrame({'x_v': x2, 'y_v': y2})
    fig = px.line(q5,
              x = q5['x_v'],
              y = q5['y_v'],
              width = 1400, height = 700,
              
              )
    return fig

my_app.run_server(
        port=8004,
        host='0.0.0.0')


# %%
df_bar = pd.DataFrame({ "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"], 
                       "Amount": [4, 1, 2, 2, 4, 5],
                       "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"] })
