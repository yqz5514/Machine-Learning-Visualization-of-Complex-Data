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
              width = 1600, height = 700,
              title = 'Stock Value-Major Companies',
              #labels = {'value': 'USD($)'}
              )

fig.update_layout(
    #title_font_size = 20,
    #title_font_color = 'red',
    title_font_family = 'Times New Roman',
    #legrnd_title_font_size = 20,
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
#%%
from sklearn.preprocessing import StandardScaler

# The StandardScaler
ss = StandardScaler()
# Standardize the training data
stocks = ss.fit_transform(stocks)
#%%
stock1 = pd.DataFrame(stocks)
#%%
stock1
#%%
stock1.rename(columns={0: 'GOOG', 1:'AAPL', 2:'AMZN', 3:'FB', 4:'NFLX', 5:'MSFT'}, inplace=True)
stock1
# %%
#2
matrix = np.array(stocks)
print(matrix)

# %%
H = (matrix.T.dot(matrix))
# %%
s, d,v = np.linalg.svd(H) 
print("SingularValues = ",d)
# %%
np.linalg.cond(matrix)
#Weak Degree of Co-linearity(DOC)
# %%
#c
corr = stock1.corr()
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
