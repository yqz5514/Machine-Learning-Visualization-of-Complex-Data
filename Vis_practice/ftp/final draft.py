
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
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_daq as daq

#import datetime

df_clean = pd.read_csv('clean_data.csv')
df_final = pd.read_csv('Final_data.csv')

df_clean.columns
features = ['ARRIVAL_DELAY', 'SCHEDULED_TIME', 'ELAPSED_TIME']
num_f = ['ARRIVAL_DELAY', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'DEPARTURE_DELAY']
##***********************************************************

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
my_app = dash.Dash('my-app', external_stylesheets=external_stylesheets)

my_app.layout = html.Div([
    # header box
    html.Div([
    html.P(style={'fontSize':50, 'textAlign':'center'}),
    html.H1('Dashboard for 2015 Flight Delay Data'),
    html.H5('by Yaxin(Janet) Zhuang   |   Dec 17, 2022')],
        style = {'font-weight': 'bold','padding' : '50px', 'textAlign':'center','backgroundColor': '#88d8c0','color':'white'}),

    html.Div([
    html.H3('       Project Summary')], style = {'textAlign':'left'}),
    html.H6('       Flight delays not only cause inconvenience to passengers, but also cost the carriers billions of dollars.'
            'The Federal Aviation Administration (FAA) considers a flight to be delayed when it is 15 minutes later than its scheduled time, '
            'while a cancellation occurs when the airline does not operate the flight at all for a certain reason. Carriers attribute flight '
            'delays to several causes such as bad weather conditions, airport congestion, airspace congestion, and use of smaller aircraft by airlines. '
            'These delays and cancellations tarnish the airlines’ reputation, often resulting in loss of demand by passengers. Further, it may have an indirect '
            'impact as the inefficiency of the air transportation system calls for a larger number of employees and ground staff, increasing the cost of doing business.(according to trefis.com)',
            style = {'textAlign':'left'}),
    html.H6('       This dataset has over 60K observations and 8 features after preprocessing and'
            'cleaning. It covers flight delay information in the US from year 2015-01-01 ot 2015-02-28. '
            'It contains categorical data such as airline name and airport name'
            'Also includes 4 numerical data. Since we have two delay features(departure and arrival delay), according to some flight dealy'
            'analysis online, departure delay tend to become the main delay problem of delay analysis becasue of the huge amount of delay happening'
            'on departure, therefore, I will primary focus on departure delay, and how does other conditions espicially airline relate to it.   '
            , style = {'textAlign':'left'}),
    dcc.Tabs(id='main-tab',
             children=[
                 dcc.Tab(label='Data Description', value='data', style={'font-size': '20px',
    'font-weight': 'bold','backgroundColor':'purple'}),
                 dcc.Tab(label='Outlier Analysis', value='outlier', style={'font-size': '20px',
    'font-weight': 'bold','backgroundColor':'cyan'}),
                 dcc.Tab(label='PCA Analysis', value='pca', style={'font-size': '20px',
    'font-weight': 'bold', 'backgroundColor':'lightcoral'}),
                 dcc.Tab(label='Normality Test', value='normality', style={'font-size': '20px',
    'font-weight': 'bold', 'backgroundColor':'palegreen'}),
                 dcc.Tab(label='Corr Coef Matrix', value='pearson', style={'font-size': '20px',
    'font-weight': 'bold', 'backgroundColor':'yellow'}),
                 dcc.Tab(label='Airline Brief look ', value='brief', style={'font-size': '20px',
    'font-weight': 'bold', 'backgroundColor':'magenta'}),
                 dcc.Tab(label='Airline Delay Analysis', value='airline-delay', style={'font-size': '20px',
    'font-weight': 'bold', 'backgroundColor':'pink'}),
                 dcc.Tab(label='Summary and Reference', value='summary', style={'font-size': '20px',
    'font-weight': 'bold', 'backgroundColor':'turquoise'}),


                      ]),
    html.Div(id='main-layout'),
    html.Br(),
])

@my_app.callback(
    Output(component_id='main-layout', component_property='children'),
    [Input(component_id='main-tab', component_property='value')]
)

def update_layout(tabs):
    if tabs == 'data':
        return data_layout
    elif tabs == 'outlier':
        return outlier_layout
    elif tabs == 'pca':
        return pca_layout
    elif tabs == 'normality':
        return norm_layout
    elif tabs == 'pearson':
        return corr_layout
    elif tabs == 'brief':
        return brief_layout
    elif tabs == 'airline-delay':
        return count_layout
    elif tabs == 'summary':
        return summ_layout

##********************************Tab*1**************************************************
#tab1 data info
missing_df = df_clean.isnull().sum(axis=0).reset_index()
missing_df.columns = ['variable', 'missing values']
missing_df['filling factor (%)']=(df_clean.shape[0]-missing_df['missing values'])/df_clean.shape[0]*100
missing_df.sort_values('filling factor (%)').reset_index(drop = True)

df_decribe = df_clean.describe()
df_decribe.reset_index(inplace=True)
df_5 = df_clean[0:5]

data_layout = html.Div([
    html.H1('Data Description',
            style={'fontSize':40, 'textAlign':'center'}),
    html.Br(),
    html.Br(),
    html.H6('      Data that will be used in this project is originally from 2015 Flight Delays and Cancellations.'
           'This data includes the on-time performance of domestic flights operated by large air carriers, and summary'
           'information on the number of on-time, delayed, canceled, and diverted flights.                            '
           '      The original data has over 500K records and 31 features, because of the large amount of data, and   '
           'unnecessary features, after data preprocessing, I decide to use data from January to February with 7 main '
           'features, including two categorical features and four numerical features. The final data have the number of'
           'total records more than 60K. The dependent variable is departure delay and all others are independent variables.',
           style = {'textAlign':'left'}),
    html.Br(),
    html.Br(),
    html.Br(),
    html.P('The top 5 rows of cleaned dataset'),
    html.Br(),
    dash_table.DataTable(df_5.to_dict('records'),
                         [{"name": i, "id": i} for i in df_5.columns]),
    html.Br(),
    html.Br(),
    html.P('The missing value table of cleaned data'),
    html.Br(),
    dash_table.DataTable(missing_df.to_dict('records'),
                         [{"name": i, "id": i} for i in missing_df.columns]),
    html.Br(),
    html.Br(),
    html.P('The describe statistic of cleaned data'),
    html.Br(),
    dash_table.DataTable(df_decribe.to_dict('records'),
                         [{"name": i, "id": i} for i in df_decribe.columns]),
    html.Br(),
    html.Br(),


    ])



##********************************Tab*2**************************************************
# tab2 outlier of dependent variable
outlier_layout = html.Div([

    html.H1(f'Outlier detection of dependent variable',
            style={'fontSize': 40, 'textAlign': 'center'}),


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

    dcc.Graph(id='graph1'),
    html.Br(),
    html.Br(),
    html.P('Outlier_Boundary_Value'),
    html.Div(id='out1')

])


@my_app.callback(
    Output(component_id='graph1', component_property='figure'),
    Output(component_id='out1', component_property='children'),
    [Input(component_id='r1', component_property='value'), ]
)
def display(a1):
    if a1 == 'Before outlier removal':

        fig = px.box(df_final, y="DEPARTURE_DELAY",
                     width=800, height=400, )
        # w_clean ='The low limit of dependent variable is -28.5, the high limit is 31.5'
        return fig, (html.Br(),
                     html.P(f'The low limit of dependent variable is -28.5, the high limit is 31.5',
                            style={'fontSize': 20, 'font-weight': 'bold', 'color': 'coral'}))
    elif a1 == 'After outlier removal':

        fig = px.box(df_clean, y="DEPARTURE_DELAY",
                     width=800, height=400, )
        # clean = 'All outliers has been removed'
        return fig, (html.Br(),
                     html.P(f'Outlier lower than -28.5 and higher than 31.5 all removed',
                            style={'fontSize': 20, 'font-weight': 'bold', 'color': 'coral'}))

##*******************************************Tab*3*************************************************
#tab3 PCA analysis
pca_layout=html.Div([
                html.Br(),
                html.Br(),
                html.H1('Principal Component Analysis',
                        style={'fontSize': 40,'textAlign':'center'}),
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
                            id='button1', n_clicks=0, style={'color':'white','background-color':'#cced1d','fontSize': 20, 'font-weight': 'bold','height': '50px','width': '400px'}),
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

##*******************************************Tab*4*************************************************
#tab4 normality test
fig_qq = plt.figure()
for i in range(1, 5):
    ax = fig_qq.add_subplot(2,2,i)
    sm.graphics.qqplot(df_clean[num_f[i - 1]], line='s', ax=ax)
    plt.title(f'Q-Q Plot of {num_f[i - 1]}')
plotly_fig = mpl_to_plotly(fig_qq)

norm_layout = html.Div([
    html.Br(),
    html.Br(),
    html.H1('Normality Test Analysis',
            style={'fontSize': 40,'textAlign':'center'}),

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
    html.Br(),
    html.Br(),

])

@my_app.callback(
    Output(component_id='graph-normality', component_property='figure'),
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
                       style={'fontSize': 20, 'font-weight': 'bold','color':'green'}),
                html.Br(),
                html.P(f"p-value = {shapiro(df_clean['DEPARTURE_DELAY'])[1]}",
                       style={'fontSize': 20, 'font-weight': 'bold','color':'green'}),

                html.Br(),
                html.P(f"Shapiro test of ARRIVAL_DELAY: statistics = {shapiro(df_clean['ARRIVAL_DELAY'])[0]}",
                       style={'fontSize': 20, 'font-weight': 'bold', 'color': 'green'}),
                html.Br(),
                html.P(f"p-value = {shapiro(df_clean['ARRIVAL_DELAY'])[1]}",
                       style={'fontSize': 20, 'font-weight': 'bold', 'color': 'green'}),

                html.Br(),
                html.P(f"Shapiro test of SCHEDULED_TIME: statistics = {shapiro(df_clean['SCHEDULED_TIME'])[0]}",
                       style={'fontSize': 20, 'font-weight': 'bold', 'color': 'green'}),
                html.Br(),
                html.P(f"p-value = {shapiro(df_clean['SCHEDULED_TIME'])[1]}",
                       style={'fontSize': 20, 'font-weight': 'bold', 'color': 'green'}),

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

##*******************************************Tab*5*************************************************
# tab5 corr coef
fig_pearson = px.imshow(df_clean[num_f].corr(),
                        text_auto=True,
                        color_continuous_scale='Blues',
                        title='Heatmap of Pearson Correlation Coefficient Matrix of Numerical Features',
                        width=800,
                        height=800
                        )
#%%

corr_layout = html.Div([
                html.H1('Corr Coef Matrix',
                        style={'fontSize': 40,'textAlign':'center'}),
                html.Br(),
                html.Br(),
                html.Div([
                    dcc.Graph(id='graph-pearson', figure=fig_pearson),
                    html.Br(),
                ]),
                html.Br(),
                html.Div([
                    html.Button("Download Image", id="btn_image",
                                style={'backgroundColor' : '#3aaab2','color':'yellow','font-size': '20px'}),
                    dcc.Download(id="download-image"),

                        ]),
                html.Br(),
                html.Br(),
])
@my_app.callback(
    Output("download-image", "data"),
    Input("btn_image", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    return dcc.send_file(r'/Users/yaxin/Downloads/heatmap.png')
##*******************************************Tab*6*************************************************
# airline brief
airlines = ['AS', 'AA', 'US', 'DL', 'UA', 'NK', 'HA', 'B6', 'EV', 'OO', 'F9', 'WN', 'VX', 'MQ']

brief_layout = html.Div([

    html.H1('Airline Overall Brief Look - Line Plot',
            style={'fontSize': 40,'textAlign':'center'}),
    html.Br(),
    html.Br(),

    html.H3('Pick a airline(s)',style={'fontSize': 15, 'font-weight': 'bold'}),
    dcc.Dropdown(id='air',
                 options=[{'label': i, 'value': i} for i in airlines],
                 multi=True,
                 placeholder='Select Symbol...'
                 ),

    html.Br(),
    html.Br(),

    html.H3('Pick a feature',style={'fontSize': 15, 'font-weight': 'bold'}),
    dcc.Dropdown(id='feature',
                 options=[{'label': i, 'value': i} for i in num_f],
                 multi=False,
                 placeholder='Select Symbol...'
                 ),

    html.Br(),
    html.Br(),


    html.Br(),
    dcc.Graph(id='airline-line'),
    html.Br(),
    html.Br(),


], style={'width': '60%'})  # change the size of the app


@my_app.callback(
    Output(component_id='airline-line', component_property='figure'),
    [Input(component_id='air', component_property='value'),
     Input(component_id='feature', component_property='value'),

     ]
)
def update(a1, a2):

    df1 = pd.DataFrame()

    for i in a1:
        df = df_clean[df_clean['AIRLINE'] == i]
        df = df[a2]
        df1 = pd.concat([df1, df], axis=1)

    df1.columns = a1



    fig = px.line(df1,
                  #x='datetime',
                  y=a1,
                  width=800,
                  height=500,  # change the size of the figure
                  title=f'The {a2} of the {a1} airline(s)',
                  labels={'index': 'date', 'value': 'time'})

    return fig
##*******************************************Tab*7*************************************************
# airline count

df_delay = df_clean[df_clean['DEPARTURE_DELAY']>0]
df_d = df_final.copy()
# df_delay_detail = df_final[df_final['DEPARTURE_DELAY']>0]
delay_type = lambda x:((0,1)[x > 5],2)[x > 45]
df_d['DELAY_LEVEL'] = df_d['DEPARTURE_DELAY'].apply(delay_type)
label_map = {0:'On time',
             1:'Samll delay',
             2:'Significant delay'}
df_d['delay_level_l'] = df_d['DELAY_LEVEL'].map(label_map)
fig_air_counts = px.histogram(df_d,
                              y='AIRLINE',
                              color="delay_level_l",
                              barmode='group',
                              title="Airline Delay Counts",
                              width=1200,
                              height=600)
def get_stats(group):
    return {'min': group.min(), 'max': group.max(),
            'count': group.count(), 'mean': group.mean()}
air_stats = df_final['DEPARTURE_DELAY'].groupby(df_final['AIRLINE']).apply(get_stats).unstack()
air_stats = air_stats.sort_values('count')
air_stats.reset_index(inplace=True)
#--------------------------------------------
count_layout = html.Div([
    html.Br(),
    html.Br(),
    html.H1('Airline Delay Analysis',
            style={'fontSize': 40,'textAlign':'center'}),

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
        html.Br(),], className='six columns'),
    html.Br(),
    html.Br(),
    html.Br(),

        html.Div([html.Br(),
                  html.Br(),
    html.P('The describe stats between departure delay and airlines',
           style={'fontSize': 20, 'font-weight': 'bold', 'color': 'black'}),
    dash_table.DataTable(air_stats.to_dict('records'),
                         [{"name": i, "id": i} for i in air_stats.columns]),
                  html.Br(),
                  html.Br(),
        html.P(f'On time means (t < 5 mins)',
               style={'fontSize': 15, 'font-weight': 'bold', 'color': 'blue'}),
        html.P(f'Small delay means (5 < t < 45 mins)',
               style={'fontSize': 15, 'font-weight': 'bold', 'color': 'red'}),
        html.P(f'Significant delay mean (t > 45 mins)',
               style={'fontSize': 15, 'font-weight': 'bold', 'color': 'green'}),
                    dcc.Graph(id='air-count', figure=fig_air_counts),
                    html.Br(),
                ] ,className='six columns'),

                html.Br(),
                html.Br(),
                html.Div([
                    html.Button("Download Image", id="btn_image",
                                style={'backgroundColor' : '#88d8c0','color':'white','font-size': '20px'}),
                    dcc.Download(id="download-image1")
                        ],className='six columns'),
        html.Br(),
        html.Br(),
        html.Br(),
])

@my_app.callback(
    Output(component_id='pie-airline-plot', component_property='figure'),
    [Input(component_id='airline-pie', component_property='value')]
)
def update_pie_plot(a1):
    fig = px.pie(df_delay,
                 values=a1,
                 names='AIRLINE',
                 hole=.2,
                 title='Pie chart of Airlines analysis')
    return fig




@my_app.callback(
        Output("download-image1", "data"),
        Input("btn_image", "n_clicks"),
        prevent_initial_call=True,)

def func(n_clicks):
    if n_clicks>0:
        return dcc.send_file(r'/Users/yaxin/Downloads/well_labeled_plot.png')
##*******************************************Tab*8*************************************************
# summary and reference
folder = r'/Users/yaxin/Documents/input_com/data'
def text_ar_val(file='Textbox1.txt'):
    with open(folder + fr'/{file}', 'r') as f:
        return f.read()


summ_layout = html.Div([
    # header box
    html.Div([
    html.H1('Reference',
            style={'fontSize':40, 'textAlign':'center'}),
    html.Br(),
    html.Br(),
    html.P('https://www.kaggle.com/datasets/usdot/flight-delays?select=flights.csv',
        style={'fontSize': 12, 'font-weight': 'bold','color':'blue'}),
    html.P('https://plotly.com/python/legend/',
               style={'fontSize': 12, 'font-weight': 'bold', 'color': 'blue'}),
    html.P('https://plotly.com/python/plotly-express/',
               style={'fontSize': 12, 'font-weight': 'bold', 'color': 'blue'}),
    html.P('https://stackoverflow.com/questions/62045601/update-the-plotly-dash-dcc-textarea-value-from-what-user-inputs',
               style={'fontSize': 12, 'font-weight': 'bold', 'color': 'blue'}),
    html.P('https://plotly.com/python/plotly-express/',
               style={'fontSize': 12, 'font-weight': 'bold', 'color': 'blue'}),
    html.P('https://dash.plotly.com/dash-core-components/upload',
               style={'fontSize': 12, 'font-weight': 'bold', 'color': 'blue'}),
    html.P('https://dash.plotly.com/dash-core-components/textarea',
               style={'fontSize': 12, 'font-weight': 'bold', 'color': 'blue'}),
    html.P('https://dash.plotly.com/dash-core-components/radioitems',
               style={'fontSize': 12, 'font-weight': 'bold', 'color': 'blue'}),
    html.P('https://dash.plotly.com/datatable',
               style={'fontSize': 12, 'font-weight': 'bold', 'color': 'blue'}),
    html.P('https://dash.plotly.com/dash-html-components/button',
               style={'fontSize': 12, 'font-weight': 'bold', 'color': 'blue'}),
    html.P('https://dash.plotly.com/dash-core-components/download',
               style={'fontSize': 12, 'font-weight': 'bold', 'color': 'blue'}),


    ]),
        html.Br(),
        html.Br(),
    html.Div([
        html.H2('Thanks for using this app:-)',
                style={'color': 'green','fontSize':25, 'textAlign':'center','font-weight': 'bold'})
    ]),
        html.Br(),
        html.Br(),
        html.Br(),

        html.Div([
            html.P('Please leave any comments and suggestion using box below',
                   style={'color': 'black','fontSize':20}),
            html.Br(),
            dcc.Textarea(
            id='textarea1', className='textarea1',
            value=text_ar_val(),
            persistence=True, persistence_type='local'),
        html.Div(daq.StopButton(id='save1', className="button",
                                    buttonText='Save', n_clicks=0)),
        html.Br(),
        html.Br(),
        html.Br(),
        ]),
])

@my_app.callback(Output('textarea1', 'children'),
              [Input('save1', 'n_clicks'), Input('textarea1','value')])
def textbox1(n_clicks, value):
    if n_clicks>0:
        global folder
        with open(folder+r'/TextBox1.txt','w') as file:
            file.write(value)
            return value


my_app.run_server(
        port=8011,
        host='0.0.0.0')