import dash as dash
from dash import dcc
from dash import html
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
my_app = dash.Dash('My app')
import dash as dash
from dash import dcc
from dash import html
import plotly.express as px
import numpy as np
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

my_app = dash.Dash('HW9', external_stylesheets=external_stylesheets)
my_app.layout = html.Div([html.H1('Homework 9', style={'textAlign': 'center'}),
                          html.Br(),
                          dcc.Tabs(id='hw-questions',
                          children=[
                          dcc.Tab(label='Question 1', value='q1'),
                          dcc.Tab(label='Question 2', value='q2'),
                          ]),

                          html.Div(id='layout')])

# Question 1
question1_layout = html.Div([html.H1('Question 1'),
                             html.H5('Test'),
                             html.P('Input:'),
                             dcc.Input(id='input1',type='text')])

question2_layout = html.Div([
    html.H1('Complex Data Visualization'),
    dcc.Dropdown(id='drop3',
                 options=[{'label': 'Introduction', 'value': 'Introduction'},
                          {'label': 'Pandas Package', 'value': 'Pandas Package'}],
                 value='Introduction'),
    html.Br(),
    html.Div(id='output3')
])

@my_app.callback(Output(component_id='layout', component_property='children'),
                 [Input(component_id='hw-questions', component_property='value')])
def update_layout(ques):
    if ques == 'q1':
        return question1_layout
    elif ques == 'q2':
        return question2_layout


my_app.run_server(
    port=8023,
    host='0.0.0.0'
)
# my_app.layout = html.Div([html.H1('Homework 9', style={'textAlign':'cetner'}),
#                           html.Br(),
#                           dcc.Tabs(id = 'hw-questions',
#                                    children=[
#                                        dcc.Tab(label='Question 1', value='q1'),
#                                        dcc.Tab(label='Question 2', value='q2')
#                                    ]),
#                           html.Div(id = 'layout')
#
#
# ]
#
# )


# my_app.layout = html.Div([
#     dcc.Checklist(id = 'my-checklist',
#                   options=[
#                       {'label':'Lecture 1', 'value': 'Lecture 1'},
#                       {'label':'Lecture 2', 'value': 'Lecture 2'},
#                       {'label':'Lecture 3', 'value': 'Lecture 3'},
#                       {'label':'Lecture 4', 'value': 'Lecture 4'},
#                   ],value=['Lecture 1']),
#
#     html.Br(),
#     html.Div(id = 'my-out')]
# )
# my_app.layout = html.Div([
#         dcc.Slider(id = 'my-input',
#             min = -10,
#             max = 90,
#             step = 2,
#             value=70       ),
#     html.Br(),
#     html.Br(),
#     dcc.Slider(id='my-out',
#                min=-25,
#                max=35,
#                step=2,
#                ),
#
# ]
# )

# @my_app.callback(
#     Output(component_id='my-out', component_property='children'),
#     [Input(component_id='my-checklist',component_property='value')]
# )
# #
# def update_Reza(input):
#     return f'The selected item {input}'



# my_app.layout = html.Div([html.H1('Complex Data Vis.'),
#                           dcc.Dropdown(id = 'my_drop', options=[
#                               {'label':'Introduction', 'value':'Introduction'},
#                               {'label':'Panda', 'value':'Panda'},
#                               {'label': 'Seaborn', 'value': 'Seaborn'},
#                               {'label': 'Plotly', 'value': 'Plotly'},
#                           ]),
#                           html.Br(),
#                           html.Br(),
#                           html.Div(id='my_out')
#
# ]
#
# )

# @my_app.callback(
#     Output(component_id='my_out', component_property='children'),
#     [Input(component_id='my_drop',component_property='value')]
# )
#
# def update_Reza(input):
#     return f'The selected item is {input}'
# # Phase 4 Callback

# my_app.layout = html.Div([html.H1('Assignment 1'),
#                           html.Button('Submit Assignment 1', id = 'a1'),
#
#                           html.H1('Assignment 2'),
#                           html.Button('Submit Assignment 2', id = 'a2'),
#
#                           html.H1('Assignment 3'),
#                           html.Button('Submit Assignment 3', id='a3'),
#
#                           html.H1('Assignment 4'),
#                           html.Button('Submit Assignment 2', id='a4')
# ]
#
# )

# Phase 3
# my_app.layout = html.Div([html.H1('Hello World! with html.H1', style={'textAlign':'center'}),
#                          html.H2('Hello World! with html.H2',style={'textAlign':'center'}),
#                          html.H3('Hello World! with html.H3',style={'textAlign':'center'}),
#                          html.H4('Hello World! with html.H4',style={'textAlign':'center'}),
#                          html.H5('Hello World! with html.H5',style={'textAlign':'center'}),
#                          html.H6('Hello World! with html.H6',style={'textAlign':'center'})
# ])
my_app.run_server(
    port = 8012,
    host = '0.0.0.0'
)