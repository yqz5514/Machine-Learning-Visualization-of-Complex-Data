import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
steps = 0.1
my_app = dash.Dash('My App')

my_app.layout = html.Div([
    dcc.Graph(id = 'my-graph'),
    html.P('Mean'),
#===================
# mean
#===================
    dcc.Slider(id = 'mean',
               min=-3,
               max = 3,
               value = 0,
               step = steps,
               marks={i:f'{i}' for i in range(-3,3)}),
    html.Br(),
    html.Br(),
#===================
# stdandard deviation
#===================
    html.P('std'),
    dcc.Slider(id = 'std',
               min=1,
               max = 3,
               value = 1,
               step = steps,
               marks={i:f'{i}' for i in range(0,3)}),
    html.Br(),
    html.Br(),
#===================
# Number of samples
#===================
    html.P('Number of samples'),
    dcc.Slider(id='samples',
               min=1,
               max=10000,
               value=1000,

               marks={1:'1',100:'100',1000:'1000',10000:'10000'}),
    html.Br(),
    html.Br(),

    # ===================
    # Number of bins
    # ===================
    html.P('Number of bins'),
    dcc.Dropdown(id='bins',
               options=[
                   {'label': 20,'value': 20},
                   {'label': 30,'value': 30},
                   {'label': 40, 'value': 40},
                   {'label': 60, 'value': 60},
                   {'label': 80, 'value': 80},
                   {'label': 100, 'value': 100},
               ], value=20, clearable=False),
])

@my_app.callback(
    Output(component_id='my-graph',component_property='figure'),
    [Input(component_id = 'mean', component_property='value'),
    Input(component_id = 'std', component_property='value'),
    Input(component_id = 'samples', component_property='value'),
    Input(component_id = 'bins', component_property='value')]
)

def display(a1,a2, a3, a4):
    x = np.random.normal(a1, a2, size = a3)
    fig = px.histogram(x = x, nbins=a4, range_x=[-6,6])
    return fig




my_app.run_server(
        port=8076,
        host='0.0.0.0')


# df = px.data.iris()
# stocks = px.data.stocks()
# features = df.columns[:-2]
# print(features)
# X = df[features].values
# Y = df.species.values
# #===============
# # Standrization
# #===============
# X = StandardScaler().fit_transform(X)
# #===============
# # PCA Analsyis
# #===============
# pca = PCA(n_components=2, svd_solver='full')
# pca.fit(X)
# X_PCA = pca.transform(X)
# print(f'explained variance ratio {pca.explained_variance_ratio_}')
# print(f'singular values {pca.singular_values_}')
#
# plt.figure()
# plt.plot(np.arange(1,len(np.cumsum(pca.explained_variance_ratio_))+1,1),np.cumsum((pca.explained_variance_ratio_)))
# plt.grid()
# plt.xticks(np.arange(1,len(np.cumsum(pca.explained_variance_ratio_))+1,1))
# plt.xlabel('number of components')
# plt.ylabel('cumlative explained variance')
# plt.show()
# #========================
# # SVD and Condition Number
# #========================
# X = df[features].values
# H = X.T @ X
#
# _, d, _ = np.linalg.svd(H)
# print('Original singular values', d)
# print('The condition number is ofr orginal feature', np.linalg.cond(X))
# print(50*"*")
# print(50*"*")
#
# H = X_PCA.T @ X_PCA
#
# _, d, _ = np.linalg.svd(H)
# print('Tranfomred singular values', d)
# print('The condition number is ofr Tranfomred feature', np.linalg.cond(X_PCA))
#
# print(50*"*")
# print(50*"*")
#
