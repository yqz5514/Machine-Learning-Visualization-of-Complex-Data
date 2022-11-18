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
######################tab######################
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


# my_app = dash.Dash('My App', external_stylesheets = external_stylesheets)
# my_app.layout = html.Div([html.H1('Lab 5', style={'textAlign': 'center'}),
#                           html.Br(),
#                           dcc.Tabs([
#                               dcc.Tab(label='Question 1', children =[
#                                   dcc.Graph(id = 'my_graph'),
#                                   html.H3('COVID global confirmed cased by country'),
#                                   html.P('Country'),
#                                   dcc.Dropdown(id='my_drop',options=[
#                                   {'label': 'US','value':'US'},
#                                   {'label': 'UK_sum','value':'UK_sum'},
#                                   {'label': 'China_sum','value':'China_sum'},
#                                   {'label': 'Germany','value':'Germany'},
#                                   {'label': 'Brazil','value':'Brazil'},
#                                   {'label': 'India','value':'India'},
#                                   {'label': 'Italy','value':'Italy'},
#                                   ],clearable = False)                        
                                                      
#                                   @my_app.callback(
#                                   Output(component_id = 'my_graph', component_property = 'figure'),
#                                   [Input(component_id = 'my_drop', component_property = 'value')])
                              
#                                   def display(a1):
#                                                fig = px.line(df,
#                                                x = df['Country/Region'],
#                                                y = a1,#['US','UK_sum','China_sum','Germany','Brazil','India','Italy'],
#                                                width = 1400, height = 700,
                                           
#                                                #labels = {'value': 'USD($)'}
#                                                )
#                                      return fig
                                  
                              
#                           ])
#                           ])
                          
#                           ])

                             
                                 
# my_app.run_server(
#         port=8013,
#         host='0.0.0.0')

#                         #   dcc.Tab(label='Question 4', value='q4'),
#                         #   dcc.Tab(label='Question 5', value='q5'),
#                         #   dcc.Tab(label='Question 6', value='q6'),
                          
                    
                          

                         

# @my_app.callback(Output(component_id='layout', component_property='children'),
#                  [Input(component_id='hw-questions', component_property='value')])


#%%
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

my_app = dash.Dash('Lab5', external_stylesheets=external_stylesheets)
my_app.layout = html.Div([html.H1('Lab5', style={'textAlign': 'center'}),
                          html.Br(),
                          dcc.Tabs(id='hw-questions',
                          children=[
                          dcc.Tab(label='Question 1', value='q1'),
                          dcc.Tab(label='Question 2', value='q2'),
                          ]),

                          html.Div(id='layout')])
#q1
question1_layout = html.Div([
    dcc.Graph(id = 'graph1'),
    html.H3('COVID global confirmed cased by country'),
    html.P('Country'),
    dcc.Dropdown(id='drop1',options=[
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
    Output(component_id = 'graph1', component_property = 'figure'),
    [Input(component_id = 'drop1', component_property = 'value'),]
    )


def display(a1):
    
    fig = px.line(df,
              x = df['Country/Region'],
              y = a1,#['US','UK_sum','China_sum','Germany','Brazil','India','Italy'],
              width = 1400, height = 700,
              
              #labels = {'value': 'USD($)'}
              )
    return fig  



#qestion2
question2_layout=html.Div([
    dcc.Graph(id = 'my_graph'),
    html.H1('The quadratic function f(x)=ax2+bx+c'),
                          
    html.P('choose value of a'),
    dcc.Slider(id = 'a',
               min = -3,
               max = 3,
               value = 0,#The value of the input
               step =1,
               marks = {i:f'{i}' for i in range(-3,3)}),
    html.Br(),
    html.Br(),
  
    
    html.P('choose value of b'),
    dcc.Slider(id = 'b',
               min = -5, # can not be negative number 
               max = 5,
               value = 0,
               step= 1,
               marks = {i:f'{i}' for i in range(-5,5)}),
    
    html.Br(),
    html.Br(),
    
    html.P('choose value of c'),
    dcc.Slider(id = 'c',
               min = -10, # can not be negative number 
               max = 10,
               value = 0,
               step = 1,# samples can not be float
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

@my_app.callback(Output(component_id='layout', component_property='children'),
                 [Input(component_id='hw-questions', component_property='value')])
def update_layout(ques):
    if ques == 'q1':
        return question1_layout
    elif ques == 'q2':
        return question2_layout


my_app.run_server(
        port=8018,
        host='0.0.0.0')

#############################################################

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
   
   html.H1(f'f(x)= x^n'),
   html.Br(),
   html.Br(),
    html.P('Please enter the polynomial order n'),
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
    fig5 = px.line(q5,
              x = q5['x_v'],
              y = q5['y_v'],
              width = 1400, height = 700,
              
              )
    return fig5

my_app.run_server(
        port=8005,
        host='0.0.0.0')





#%%
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

my_app = dash.Dash('Q6', external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df_bar = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df_bar, x="Fruit", y="Amount", color="City", barmode="group")

my_app.layout = html.Div(children=[
    # All elements from the top of the page
    html.Div([
        html.Div([
            html.H1(children='Hello Dash 1'),

            html.Div(children='''
                Dash: A web application framework for Python.
            '''),

            dcc.Graph(
                id='graph1',
                figure=fig
            ),  
            html.P('Slider 1'),
            dcc.Slider(id = 'mean',
               min = 0,
               max = 20,
               value = 0,#The value of the input
               step =1,
               marks = {i:f'{i}' for i in range(0,20)}),
        ], className='six columns'),
        html.Div([
            html.H1(children='Hello Dash 2'),

            html.Div(children='''
                Dash: A web application framework for Python.
            '''),

            dcc.Graph(
                id='graph2',
                figure=fig
            ),  
            html.P('Slider 2'),
            dcc.Slider(id = 'mean',
               min = 0,
               max = 20,
               value = 0,#The value of the input
               step =1,
               marks = {i:f'{i}' for i in range(0,20)}),
        ], className='six columns'),
    ], className='row'),
    # New Div for all elements in the new 'row' of the page
   html.Div([
        html.Div([
            html.H1(children='Hello Dash 3'),

            html.Div(children='''
                Dash: A web application framework for Python.
            '''),

            dcc.Graph(
                id='graph3',
                figure=fig
            ),  
            html.P('Slider 3'),
            dcc.Slider(id = 'mean',
               min = 0,
               max = 20,
               value = 0,#The value of the input
               step =1,
               marks = {i:f'{i}' for i in range(0,20)}),
        ], className='six columns'),
        html.Div([
            html.H1(children='Hello Dash 4'),

            html.Div(children='''
                Dash: A web application framework for Python.
            '''),

            dcc.Graph(
                id='graph4',
                figure=fig
            ),  
            html.P('Slider 4'),
            dcc.Slider(id = 'mean',
               min = 0,
               max = 20,
               value = 0,#The value of the input
               step =1,
               marks = {i:f'{i}' for i in range(0,20)}),
        ], className='six columns'),
    ], className='row'),
])


my_app.run_server(
        port=8111,
        host='0.0.0.0')

# %%
