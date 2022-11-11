#%%
# import packages
import pandas_datareader as web 
import numpy as np
import pandas as pd
import plotly.express as px

#%%
#q1
#load data
stock = px.data.stocks()
stock.head()
#%%
stock.columns
# %%
#q2
fig = px.line(stock,
              x = stock.date,
              y = ['GOOG', 'AAPL', 'AMZN', 'FB', 'NFLX', 'MSFT'],
              width = 1400, height = 700,
              title = 'Stock Value-Major Companies',
              #labels = {'value': 'USD($)'}
              )

fig.update_layout(
    title_font_family = 'Times New Roman',
    legend_title_font_color = 'green',
    font_color = 'red',
    font_family = 'Courier New'
)
fig.write_html('first_figure.html', auto_open=True)
#fig.show(renderer = 'browser')


# %%
#q3
from plotly.subplots import make_subplots 
import plotly.graph_objects as go 

fig = make_subplots(rows=3, cols = 2)
fig.add_trace(go.Histogram(x = stock.GOOG,
                           nbinsx=50,
                           name = 'GOOGL'),
                            row = 1, col = 1)
fig.add_trace(go.Histogram(x = stock.AAPL,
                           nbinsx=50,
                           name = 'AAPL'),
                            row = 1, col = 2)
fig.add_trace(go.Histogram(x = stock.AMZN,
                           nbinsx=50,
                           name = 'AMZN'),
                            row = 2, col = 1)
fig.add_trace(go.Histogram(x = stock.FB,
                           nbinsx=50,
                           name = 'FB'),
                            row = 2, col = 2)
fig.add_trace(go.Histogram(x = stock.NFLX,
                           nbinsx=50,
                           name = 'NFLX'),
                            row = 3, col = 1)
fig.add_trace(go.Histogram(x = stock.MSFT,
                           nbinsx=50,
                           name = 'MSFT'),
                            row = 3, col = 2)
fig.update_layout(width = 800, height = 600,
                 )
fig.write_html('first_figure.html', auto_open=True)


# %%
# q4
#1
# only consider each company stocks as feature
stocks = stock.copy()

stocks.drop(columns = ['date'], inplace=True)
stocks
#%%
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# The StandardScaler
ss = StandardScaler()
# Standardize the training data
stock_sta = ss.fit_transform(stocks)
#%%
stock_sta = pd.DataFrame(stock_sta)
#%%
stock_sta
#%%
stock_sta.rename(columns={0: 'GOOG', 1:'AAPL', 2:'AMZN', 3:'FB', 4:'NFLX', 5:'MSFT'}, inplace=True)
stock_sta

# %%
features = stock_sta.columns
X = stock_sta[features].values
H = X.T @ X
_,d,_ = np.linalg.svd(H)
print('Original singular values', d)
print('The Condition number is original features', np.linalg.cond(X))

# #%%
# features
# #%%
# #2
# matrix = np.array(stocks)
# print(matrix)

# # %%
# #H = (matrix.T.dot(matrix))
# H = (stocks.T.dot(stocks))
# # %%
# [s,d,v] = np.linalg.svd(H) 
# print("SingularValues = ", d)
# # %%
# #np.linalg.cond(matrix)
# np.linalg.cond(stocks)
# #Weak Degree of Co-linearity(DOC)
# %%
#c

corr = stock_sta.corr()
#%%
import seaborn as sns
import matplotlib.pyplot as plt

#%%
sns.heatmap(corr, 
            linewidths=0.5, 
            vmin = -1,
            vmax = 1,
            annot= True)# annot= True will add value to each cube
plt.title('Correlation Coefficent between features-Original feature space')


# %%
# d
df = px.data.stocks()
# %%
features = df.columns[1:]
print(features)
# %%
X = df[features].values

# %%
# explained variance ratio of original feature space

X = StandardScaler().fit_transform(X)
# PCA analysis
# explained variance ratio
pca = PCA(n_components = 6, svd_solver = 'full') #n_components = useful feature number 
pca.fit(X)
#X_PCA = pca.transform(X)
print(f'explained variance ratio {pca.explained_variance_ratio_}') 
#print(f'singular values {pca.singular_values_}')

# %%
# explained variance ratio of reduced feature space

X = StandardScaler().fit_transform(X)
# PCA analysis
# explained variance ratio
pca_reduce = PCA(n_components = 'mle', svd_solver = 'full') #n_components = useful feature number 
pca_reduce.fit(X)
#X_PCA = pca.transform(X)
print(f'explained variance ratio of reduced feature space {pca_reduce.explained_variance_ratio_}') 
# %%
plt.figure()
plt.plot(np.arange(1,len(np.cumsum(pca_reduce.explained_variance_ratio_))+1,1),np.cumsum((pca_reduce.explained_variance_ratio_)))
plt.grid()
plt.xticks(np.arange(1,len(np.cumsum(pca_reduce.explained_variance_ratio_))+1,1))
plt.xlabel('number of components')
plt.ylabel('cumlative explained variance')
plt.title('cumulative explained variance versus the number of components for reduced value')
plt.show()
# %%
reduce_f = df.columns[1:5]
reduce_f
#%%
# SVD condition numebr 
X1 = df[reduce_f].values
N = X1.T @ X1
_,d,_ = np.linalg.svd(N)
print('Reduced singular values', d)
print('The Condition number of reduced features', np.linalg.cond(X1))
# %%
df_r = df.iloc[:,1:5]
df_r

#%%
corr_reduce = df_r.corr()
sns.heatmap(corr_reduce, 
            linewidths=0.5, 
            vmin = -1,
            vmax = 1,
            annot= True)# annot= True will add value to each cube
plt.title('Correlation Coefficent between reduced feature space')


# %%
X_PCA = pca.transform(X)
X_PCA
# %%
new_df = pd.DataFrame(X_PCA)
new_df
# %%
new_df = new_df.rename(columns={0:'Principal col 1', 
                        1:'Principal col 2', 
                        2:'Principal col 3',
                        3:'Principal col 4',
                        4:'Principal col 5',
                        5:'Principal col 6'
                        })

new_df.head(5)
#%%
pca_df = new_df.iloc[:,0:4]
pca_df.head(5)
#%%
new_df['date'] = stock['date']
new_df
#%%
# %%
fig = px.line(new_df,
              x = new_df.date,
              y = ['Principal col 1', 'Principal col 2', 'Principal col 3', 'Principal col 4'],
              width = 1400, height = 700,
              title = 'Stock Value-reduced dimension feature space',
              #labels = {'value': 'USD($)'}
              )

fig.update_layout(
    title_font_family = 'Times New Roman',
    legend_title_font_color = 'green',
    font_color = 'red',
    font_family = 'Courier New'
)
fig.write_html('first_figure.html', auto_open=True)
# %%
from plotly.subplots import make_subplots 
import plotly.graph_objects as go 

fig = make_subplots(rows=4, cols = 1)
fig.add_trace(go.Histogram(x = new_df['Principal col 1'],
                           nbinsx=50,
                           name = 'Principal col 1'),
                            row = 1, col = 1)
fig.add_trace(go.Histogram(x = new_df['Principal col 2'],
                           nbinsx=50,
                           name = 'Principal col 2'),
                            row = 2, col = 1)
fig.add_trace(go.Histogram(x = new_df['Principal col 3'],
                           nbinsx=50,
                           name = 'Principal col 3'),
                            row = 3, col = 1)
fig.add_trace(go.Histogram(x = new_df['Principal col 4'],                          nbinsx=50,
                           name = 'Principal col 4'),
                            row = 4, col = 1)

fig.update_layout(width = 400, height = 600,
                 )
fig.write_html('first_figure.html', auto_open=True)

# %%
import plotly.express as px
dff = px.data.stocks()
dff.drop(columns=['date'], inplace=True)

#%%
fig = px.scatter_matrix(dff,
    dimensions=dff.columns,
    title = 'Original feature space'
    )

fig.write_html('first_figure.html', auto_open=True)
# %%
fig = px.scatter_matrix(df_r,
    dimensions=df_r.columns,
    title='reduced feature space'
    )
fig.write_html('first_figure.html', auto_open=True)
# %%
