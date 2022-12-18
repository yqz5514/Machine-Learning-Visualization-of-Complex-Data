import pandas as pd
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly
import numpy as np
# import seaborn as sns
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

# %%
df_clean = pd.read_csv('clean_data.csv')
df_final = pd.read_csv('Final_data.csv')

# %%
df_clean.columns
features = ['ARRIVAL_DELAY', 'SCHEDULED_TIME', 'ELAPSED_TIME']
num_f = ['ARRIVAL_DELAY', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'DEPARTURE_DELAY']
# %%
import numpy as np
from scipy.stats import shapiro
from scipy.stats import kstest
from scipy.stats import normaltest


def shapiro_test(x, title):
    stats, p = shapiro(x)
    print('=' * 50)
    print(f'Shapiro test : {title} dataset : statistics = {stats:.2f} p-vlaue of ={p:.2f}')
    alpha = 0.01
    if p > alpha:
        print(f'Shapiro test: {title} dataset is Normal')
    else:
        print(f'Shapiro test: {title} dataset is NOT Normal')
    print('=' * 50)


# %%

fig_qq = plt.figure()

for i in range(1, 5):
    ax = fig_qq.add_subplot(2, 2, i)
    sm.graphics.qqplot(df_clean[num_f[i - 1]], line='s', ax=ax)
    plt.title(f'Q-Q Plot of {num_f[i - 1]}')
plotly_fig = mpl_to_plotly(fig_qq)
########################################

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