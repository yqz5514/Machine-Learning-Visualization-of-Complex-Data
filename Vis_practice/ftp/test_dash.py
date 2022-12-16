
#%%

import pandas as pd
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly
import numpy as np
#import seaborn as sns
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

fig_qq = plt.figure()

for i in range(1, 5):
    ax = fig_qq.add_subplot(2,2,i)
    sm.graphics.qqplot(df_clean[num_f[i - 1]], line='s', ax=ax)
    plt.title(f'Q-Q Plot of {num_f[i - 1]}')
plotly_fig = mpl_to_plotly(fig_qq)

  
#%%
#outlier 
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

my_app = dash.Dash('My App', external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css'])

#phase three
# Divioison is a section of app
my_app.layout=html.Div([
    
    html.H1(f'Outlier detection of dependent variable',style={'fontSize': 20, 'font-weight': 'bold', 'textAlign':'left'}),
  
   html.Br(),
   html.Br(),
   html.P("Options:"),
    dcc.RadioItems(
        id='r1', 
        options=['Before outlier removal', 'After outlier removal'],
        value='', 
        style={'fontSize': 15, 'font-weight': 'bold'},
        inline=True
    ),
    
    dcc.Graph(id = 'graph1'),
    html.Br(),
                          html.Br(),
                          html.P('Outlier_Boundary_Value'),
                          html.Div(id='out1')
    
])

@my_app.callback(
    Output(component_id = 'graph1', component_property = 'figure'),
    Output(component_id='out1', component_property='children'),
    [Input(component_id = 'r1', component_property = 'value'),]
    )


def display(a1):
    
    if a1 == 'Before outlier removal':
        
        fig = px.box(df_final, y="DEPARTURE_DELAY",
                      width = 800, height = 400,)
        #w_clean ='The low limit of dependent varible is -28.5, the high limit is 31.5'
        return fig, (html.Br(),
                     html.P(f'The low limit of dependent varible is -28.5, the high limit is 31.5',
                            style={'fontSize': 20, 'font-weight': 'bold','color':'coral'}))
    elif a1 == 'After outlier removal':
        
        fig = px.box(df_clean, y="DEPARTURE_DELAY",
                     width = 800, height = 400,)
        #clean = 'All outliers has been removed'
        return fig, (html.Br(),
                     html.P(f'Outlier between -28.5 and 31.5 all removed',
                            style={'fontSize': 20, 'font-weight': 'bold','color':'coral'}))

my_app.run_server(
        port=8015,
        host='0.0.0.0')

#%%
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

my_app = dash.Dash('My App', external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css'])

#phase three
# Divioison is a section of app
my_app.layout=html.Div([
                html.Br(),
                html.Br(),
                html.P('Principal Component Analysis', style={'fontSize': 20, 'font-weight': 'bold', 'textAlign':'left'}),
                html.Div([
                    dcc.Graph(id='graph2'),
                    html.P('Select the Number of Features', style={'fontSize': 20}),
                    dcc.Slider(
                        id='slider1', 
                        min=1, 
                        max=3, 
                        step=1, 
                        value=1,
                        marks={1: '1', 2: '2', 3: '3'}),
                    html.Div(id='slider-out', style={'fontSize': 20, 'font-weight': 'bold'})
                ], className='six columns'),

                html.Div([html.Div([
                html.P('SVD Analysis and Condition Number', style={'fontSize': 20, 'font-weight': 'bold'}),
                html.Br(),
                html.Button('Calculate PCA and SVD',
                            id='button1', n_clicks=0, style={'color':'white','background-color':'#3aaab2','fontSize': 20, 'font-weight': 'bold','height': '50px','width': '400px'}),
                html.Div(id='pca-out'),
                ],
                className='six columns')
    ]),
html.Br(),
html.Br(),
])
@my_app.callback(
    Output(component_id='graph2', component_property='figure'),
    Output(component_id='slider-out', component_property='children'),
    [Input(component_id='slider1', component_property='value')]
)

def update_pca_graph(a1):
    X = df_clean[features].values
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=int(a1), svd_solver='full')
    pca.fit(X)
    x0 = np.arange(1, len(np.cumsum(pca.explained_variance_ratio_)) + 1, 1)
    fig = px.line(x=x0, y=np.cumsum(pca.explained_variance_ratio_), 
                  title=f'PCA plot of {a1} Numerical Features')

    num = 0
    for i in range(a1):
        num = num + pca.explained_variance_ratio_[i]
    return fig, (html.Br(),
                 html.P(f'Explained variance ratio is: {pca.explained_variance_ratio_}', style={'color':'coral'}),
                 html.Br(),
                 html.P(f'Explained data % is {num*100:.2f}% data', style={'color':'coral'}),
                 html.Br(),
                 html.Br()
                 )

@my_app.callback(
    Output('pca-out', 'children'),
    [Input('button1', 'n_clicks')]
)

def update_svd(n_clicks):
    X = df_clean[features].values
    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components='mle', svd_solver='full')
    pca.fit(X)
    X_PCA = pca.transform(X)

    H = np.matmul(X.T, X)
    _, d, _ = np.linalg.svd(H)

    H_PCA = np.matmul(X_PCA.T, X_PCA)
    _, d_PCA, _ = np.linalg.svd(H_PCA)

    if n_clicks > 0:
        return (html.Br(),
                html.P(f'PCA suggested feature number after reduced dimension is: {X_PCA.shape[1]}', style={'fontSize': 20, 'font-weight': 'bold','color':'coral'}),
                html.Br(),
                html.P(f'Original Data: Singular Values {d}',style={'fontSize': 20, 'font-weight': 'bold','color':'coral'}),
                html.Br(),
                html.P(f'Original Data: condition number {LA.cond(X)}', style={'fontSize': 20, 'font-weight': 'bold', 'color':'coral'}),
                html.Br(),
                html.P(f'Transformed Data: Singular Values {d_PCA}', style={'fontSize': 20, 'font-weight': 'bold','color':'coral'}),
                html.Br(),
                html.P(f'Transformed Data: condition number {LA.cond(X_PCA)}', style={'fontSize': 20, 'font-weight': 'bold','color':'coral'}),
                html.Br(),
                html.Br(),
                )

my_app.run_server(
        port=8021,
        host='0.0.0.0')


#%%
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

my_app = dash.Dash('My App', external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css'])

#phase three
# Divioison is a section of app
my_app.layout = html.Div([
              
                html.Br(),
                html.Br(),
                html.P('Normality Test', style={'fontSize': 20, 'font-weight': 'bold', 'textAlign':'left'}),
                html.Div([
                    dcc.RadioItems(options=['Histogram','QQ Plot'],
                                   value='',
                                   id='radio-normal-plot',
                                   style={'fontSize': 15, 'font-weight': 'bold'}),
                    html.Br(),
                    dcc.Graph(id='graph-normality'),
                    html.Br(),
                    html.P(id='out3')
                ],
                    className='six columns'),
                
#                 html.Div([html.Div([
#                 html.P('Shapiro test', style={'fontSize': 20, 'font-weight': 'bold'}),
#                 html.Br(),
#                 html.Button('Calculate shapiro stats',
#                             id='button-s', n_clicks=0, style={'color':'white','background-color':'#3aaab2','fontSize': 20, 'font-weight': 'bold','height': '50px','width': '400px'}),
#                 html.Div(id='hy-out'),
#                 ],
#                 className='six columns')
                

# ]),
#                 html.Br(),
# html.Br(),
])
@my_app.callback(
     Output('graph-normality', 'figure'),
    Output('out3','children'),
    Input('radio-normal-plot', 'value'),
)
 
def update_nor_graph(a1):
    if a1 == 'Histogram':
        fig_normal_hist = make_subplots(rows=2, cols=2, y_title='Count')
        fig_normal_hist.add_trace(go.Histogram(x=df_clean['DEPARTURE_DELAY'], name='DEPARTURE_DELAY'),row=1, col=1)
        fig_normal_hist.add_trace(go.Histogram(x=df_clean['ARRIVAL_DELAY'], name='ARRIVAL_DELAY'), row=1, col=2)
        fig_normal_hist.add_trace(go.Histogram(x=df_clean['SCHEDULED_TIME'], name='SCHEDULED_TIME'), row=2, col=1)
        fig_normal_hist.add_trace(go.Histogram(x=df_clean['ELAPSED_TIME'], name='ELAPSED_TIM'), row=2, col=2)
        
        fig_normal_hist.update_layout(title_text='Subplots of Histogram for Numerical Features')

        return fig_normal_hist
    
    if a1 == 'QQ Plot':
        return plotly_fig
    
# @my_app.callback(
#     Output('hy-out', 'children'),
#     [Input('button-s', 'n_clicks')]
# )

# def update_test(n_clicks):
    
#     if n_clicks > 0:
#         for i in num_f:
            
#             return shapiro_test(df_clean[i], 'Data')

my_app.run_server(
        port=8030,
        host='0.0.0.0')      

#%%
df_clean.columns
# %%
fig_normal_hist = make_subplots(rows=2, cols=2, y_title='Count')
fig_normal_hist.add_trace(go.Histogram(x=df_clean['DEPARTURE_DELAY'], name='DEPARTURE_DELAY'),row=1, col=1)
fig_normal_hist.add_trace(go.Histogram(x=df_clean['ARRIVAL_DELAY'], name='ARRIVAL_DELAY'), row=1, col=2)
fig_normal_hist.add_trace(go.Histogram(x=df_clean['SCHEDULED_TIME'], name='SCHEDULED_TIME'), row=2, col=1)
fig_normal_hist.add_trace(go.Histogram(x=df_clean['ELAPSED_TIME'], name='ELAPSED_TIM'), row=2, col=2)
        
fig_normal_hist.update_layout(title_text='Subplots of Histogram for Numerical Features')
fig_normal_hist.write_html('first_figure.html', auto_open=True)



#%%
df_clean.columns
# %%
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

my_app = dash.Dash('My App', external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css'])

#phase three
# Divioison is a section of app
my_app.layout  = html.Div([
                html.Br(),
                html.Br(),
                html.P('Normality Test - Plot', style={'fontSize': 20, 'font-weight': 'bold', 'textAlign':'left'}),
                html.Div([
                    dcc.RadioItems(options=['Histogram','QQ Plot'],
                                   value='',
                                   id='radio-normal-plot',
                                   style={'fontSize': 15, 'font-weight': 'bold'}),
                    html.Br(),
                    dcc.Graph(id='graph-normality'),
                    html.Br(),
                    html.P(id='out3')
                ],
                 ),
])

@my_app.callback(
    Output('graph-normality', 'figure'),
    Output('out3','children'),
    [Input('radio-normal-plot', 'value'),]
)

def update_normal_plot(a1):
     if a1 == 'Histogram':
        fig_normal_hist = make_subplots(rows=2, cols=2, y_title='Count')
        fig_normal_hist.add_trace(go.Histogram(x=df_clean['DEPARTURE_DELAY'], name='DEPARTURE_DELAY'),row=1, col=1)
        fig_normal_hist.add_trace(go.Histogram(x=df_clean['ARRIVAL_DELAY'], name='ARRIVAL_DELAY'), row=1, col=2)
        fig_normal_hist.add_trace(go.Histogram(x=df_clean['SCHEDULED_TIME'], name='SCHEDULED_TIME'), row=2, col=1)
        fig_normal_hist.add_trace(go.Histogram(x=df_clean['ELAPSED_TIME'], name='ELAPSED_TIM'), row=2, col=2)
        
        fig_normal_hist.update_layout(title_text='Subplots of Histogram for Numerical Features')

        return fig_normal_hist
    
     elif a1 == 'QQ Plot':
        return plotly_fig
    
my_app.run_server(
        port=8002,
        host='0.0.0.0')  

    # if value == 'Histogram':
    #     fig_normal_hist = make_subplots(rows=2, cols=3, y_title='Count')
    #     fig_normal_hist.add_trace(go.Histogram(x=injuries_direct['injuries_direct'], name='injuries_direct'),row=1, col=1)
    #     fig_normal_hist.add_trace(go.Histogram(x=injuries_indirect['injuries_indirect'], name='injuries_indirect'), row=1, col=2)
    #     fig_normal_hist.add_trace(go.Histogram(x=deaths_direct['deaths_direct'], name='deaths_direct'), row=1, col=3)
    #     fig_normal_hist.add_trace(go.Histogram(x=deaths_indirect['deaths_indirect'], name='deaths_indirect'), row=2, col=1)
    #     fig_normal_hist.add_trace(go.Histogram(x=damage_property['damage_property'], name='damage_property'), row=2, col=2)
    #     fig_normal_hist.add_trace(go.Histogram(x=damage_crops['damage_crops'], name='damage_crops'), row=2, col=3)

    #     # Update xaxis properties
    #     fig_normal_hist.update_xaxes(title_text='People',rangeselector_font_size=2, row=1, col=1)
    #     fig_normal_hist.update_xaxes(title_text='People',rangeselector_font_size=2,  row=1, col=2)
    #     fig_normal_hist.update_xaxes(title_text='People',rangeselector_font_size=2,  row=1, col=3)
    #     fig_normal_hist.update_xaxes(title_text='People',rangeselector_font_size=2, row=2, col=1)
    #     fig_normal_hist.update_xaxes(title_text='USD($)',rangeselector_font_size=2, row=2, col=2)
    #     fig_normal_hist.update_xaxes(title_text='USD($)',rangeselector_font_size=2,  row=2, col=3)

    #     fig_normal_hist.update_layout(title_text='Subplots of Histogram for Numerical Features')

    #     return fig_normal_hist, 'You have chosen '+value

    # if value == 'QQ Plot':
    #     return plotly_fig, 'You have chosen '+value
# %%
#Datetime
print(df_clean.info())
# %%
