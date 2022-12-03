#%%
import pandas_datareader as web
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linalg as LA
import numpy as np
from tabulate import tabulate
from scipy.optimize import curve_fit
plt.style.use('seaborn-whitegrid')

#%%
aapl = web.DataReader('AAPL', data_source='yahoo', start = '2000-01-01', end='2022-09-01')
orcl = web.DataReader('ORCL', data_source='yahoo', start = '2000-01-01', end='2022-09-01')
tsla = web.DataReader('TSLA', data_source='yahoo', start = '2000-01-01', end='2022-09-01')
ibm  = web.DataReader('IBM', data_source='yahoo', start = '2000-01-01', end='2022-09-01')
yelp = web.DataReader('YELP', data_source='yahoo', start = '2000-01-01', end='2022-09-01')
msft = web.DataReader('MSFT', data_source='yahoo', start = '2000-01-01', end='2022-09-01')

#%%
#2
#2. The database contains 6 features: “High”, “Low”, “Open”, “Close”, “Volume”, “Adj Close” in USD($). 
# Using the matplotlib.pyplot package and subplot command, plot the “High” columns for all companies in one figure with 3 rows and 2 columns graph. 
# Make sure to add title, legend, x-label. y-label and grid to your plot.
# The plot should look like the following. Fig size = (16,8)

def get_subplot(col):
    fig, ax = plt.subplots(3,2,figsize=(16,8))

    ax[0,0].plot(aapl[col])
    ax[0,0].set_title(col+' price history of AAPL')
    ax[0,0].set_xlabel('Date')
    ax[0,0].set_ylabel(col+' price USD($)')
    
    ax[1,0].plot(tsla[col])
    ax[1,0].set_title(col+' price history of TSLA')
    ax[1,0].set_xlabel('Date')
    ax[1,0].set_ylabel(col+' price USD($)')

    ax[2,0].plot(yelp[col])
    ax[2,0].set_title(col+' price history of YELP')
    ax[2,0].set_xlabel('Date')
    ax[2,0].set_ylabel(col+' price USD($)')

    ax[0,1].plot(orcl[col])
    ax[0,1].set_title(col+' price history of ORCL')
    ax[0,1].set_xlabel('Date')
    ax[0,1].set_ylabel(col+' price USD($)')

    ax[1,1].plot(ibm[col])
    ax[1,1].set_title(col+' price history of IBM')
    ax[1,1].set_xlabel('Date')
    ax[1,1].set_ylabel(col+' price USD($)')

    ax[2,1].plot(msft[col])
    ax[2,1].set_title(col+' price history of MSFT')
    ax[2,1].set_xlabel('Date')
    ax[2,1].set_ylabel(col+' price USD($)')

#plt.grid(visible=True, which='both')
    plt.tight_layout()
    plt.show()
    #fig.savefig(col+' plots for all stocks', dpi = 500)
    #fig.savefig(fname=(col +' plots for all stocks.eps'), format = 'eps',dpi = 500)
#%%
get_subplot('High') 
get_subplot('Low')
get_subplot('Open')
get_subplot('Close')
get_subplot('Volume')
get_subplot('Adj Close')

# %%
#plt.savefig('Q9(3)',dpi = 500)
#%%
def get_subplot_hist(col):
    fig, ax = plt.subplots(3,2,figsize=(16,8))

    ax[0,0].hist(aapl[col], bins=50)
    ax[0,0].set_title(col+' price history of AAPL')
    ax[0,0].set_xlabel('Date')
    ax[0,0].set_ylabel(col+' price USD($)')
    
    ax[1,0].hist(tsla[col], bins=50)
    ax[1,0].set_title(col+' price history of TSLA')
    ax[1,0].set_xlabel('Date')
    ax[1,0].set_ylabel(col+' price USD($)')

    ax[2,0].hist(yelp[col], bins=50)
    ax[2,0].set_title(col+' price history of YELP')
    ax[2,0].set_xlabel('Date')
    ax[2,0].set_ylabel(col+' price USD($)')

    ax[0,1].hist(orcl[col], bins=50)
    ax[0,1].set_title(col+' price history of ORCL')
    ax[0,1].set_xlabel('Date')
    ax[0,1].set_ylabel(col+' price USD($)')

    ax[1,1].hist(ibm[col], bins=50)
    ax[1,1].set_title(col+' price history of IBM')
    ax[1,1].set_xlabel('Date')
    ax[1,1].set_ylabel(col+' price USD($)')

    ax[2,1].hist(msft[col], bins=50)
    ax[2,1].set_title(col+' price history of MSFT')
    ax[2,1].set_xlabel('Date')
    ax[2,1].set_ylabel(col+' price USD($)')

#plt.grid(visible=True, which='both')
    plt.tight_layout()
    plt.show()
    
    #fig.savefig(fname=('Histgoram of '+col+' for all stocks'), format = 'eps',dpi = 500)
    
# %%
get_subplot_hist('High')
get_subplot_hist('Low')
get_subplot_hist('Open')
get_subplot_hist('Close')
get_subplot_hist('Volume')
get_subplot_hist('Adj Close')


# %%
corr_table = aapl.corr()
corr_table
# #%%
# df1 = corr_table.reset_index(inplace=True)
# #%%
# print(df1)
#%%
names = ['High','Low','Open','Close','Volume','Adj Close']
# #%%
# for i in names:
    
#     print(corr_table.loc[[i]])
# #%%
# corr_table.loc[[0]]
# #%%
# t.tolist()
# #%%
# for i in names:
#     print(corr_table.loc[corr_table[i]])
# %%
from prettytable import PrettyTable

def get_table(com):
    
    df = com.corr()
    x = PrettyTable()
    x.field_names = ['name','High','Low','Open','Close','Volume','Adj Close']
    #x.add_row(['high',0.999894, 0.999933,0.999910,-0.421640,0.999676])
    names = ['High','Low','Open','Close','Volume','Adj Close']
    #list = []
    for i in names:
    #value = corr_table.loc[[i]]
    #list = []
    #for j in value:
    #    delim = ','
    #    s = ''
    #    s += (j + delim)
    #x.field_names(i)
        x.add_row([i, *df[i].tolist()])
        #x.title('Corr table of '+ com)
    print(x.get_string(title = 'Corr table'))
    return

#%%
get_table(aapl)
get_table(orcl)
get_table(tsla)
get_table(ibm)
get_table(msft)
get_table(yelp)

#%%
fig = plt.figure()
pd.plotting.scatter_matrix(aapl, hist_kwds= {'bins' : 50} , alpha = 0.5, s = 10, diagonal = 'kde')
#plt.savefig(fname=('scatter_matrix.eps''), format = 'eps',dpi = 500 )


    
# %%
def get_sca(com):
    #names = ['High','Low','Open','Close','Volume','Adj Close']
    
    fig, ax = plt.subplots(6,6,figsize=(16,16))
    

    ax[0,0].scatter(com['High'],com['High'] )
    ax[0,0].set_title(round(com['High'].corr(com['High']),2))
    ax[0,0].set_xlabel('High')
    ax[0,0].set_ylabel('High')
    
    ax[0,1].scatter(com['High'], com['Low'])
    ax[0,1].set_title(round(com['High'].corr(com['Low']),2))
    ax[0,1].set_xlabel('High')
    ax[0,1].set_ylabel('Low')

    ax[0,2].scatter(com['High'], com['Open'])
    ax[0,2].set_title(round(com['High'].corr(com['Open']),2))
    ax[0,2].set_xlabel('High')
    ax[0,2].set_ylabel('Open')
    
    ax[0,3].scatter(com['High'],com['Close'] )
    ax[0,3].set_title(round(com['High'].corr(com['Close']),2))
    ax[0,3].set_xlabel('High')
    ax[0,3].set_ylabel('Close')
    
    ax[0,4].scatter(com['High'],com['Volume'] )
    ax[0,4].set_title(round(com['High'].corr(com['Volume']),2))
    ax[0,4].set_xlabel('High')
    ax[0,4].set_ylabel('Volume')
    
    ax[0,5].scatter(com['High'],com['Adj Close'] )
    ax[0,5].set_title(round(com['High'].corr(com['Adj Close']),2))
    ax[0,5].set_xlabel('High')
    ax[0,5].set_ylabel('Adj Close')
    
    ax[1,0].scatter(com['Low'],com['High'] )
    ax[1,0].set_title(round(com['Low'].corr(com['High']),2))
    ax[1,0].set_xlabel('Low')
    ax[1,0].set_ylabel('High')

    ax[1,1].scatter(com['Low'],com['Low'] )
    ax[1,1].set_title(round(com['Low'].corr(com['Low']),2))
    ax[1,1].set_xlabel('Low')
    ax[1,1].set_ylabel('Low')
    
    ax[1,2].scatter(com['Low'],com['Open'] )
    ax[1,2].set_title(round(com['Low'].corr(com['Open']),2))
    ax[1,2].set_xlabel('Low')
    ax[1,2].set_ylabel('Open')
    
    ax[1,3].scatter(com['Low'],com['Close'] )
    ax[1,3].set_title(round(com['Low'].corr(com['Close']),2))
    ax[1,3].set_xlabel('Low')
    ax[1,3].set_ylabel('Close')
    
    ax[1,4].scatter(com['Low'],com['Volume'] )
    ax[1,4].set_title(round(com['Low'].corr(com['Volume']),2))
    ax[1,4].set_xlabel('Low')
    ax[1,4].set_ylabel('Volume')
    
    ax[1,5].scatter(com['Low'],com['Adj Close'] )
    ax[1,5].set_title(round(com['Low'].corr(com['Adj Close']),2))
    ax[1,5].set_xlabel('Low')
    ax[1,5].set_ylabel('Adj Close')
    
    ax[2,0].scatter(com['Open'],com['High'] )
    ax[2,0].set_title(round(com['Open'].corr(com['High']),2))
    ax[2,0].set_xlabel('Open')
    ax[2,0].set_ylabel('High')
    
    ax[2,1].scatter(com['Open'],com['Low'] )
    ax[2,1].set_title(round(com['Open'].corr(com['Low']),2))
    ax[2,1].set_xlabel('Open')
    ax[2,1].set_ylabel('Low')
    
    ax[2,2].scatter(com['Open'],com['Open'] )
    ax[2,2].set_title(round(com['Open'].corr(com['Open']),2))
    ax[2,2].set_xlabel('Open')
    ax[2,2].set_ylabel('Open')
    
    ax[2,3].scatter(com['Open'],com['Close'] )
    ax[2,3].set_title(round(com['Open'].corr(com['Close']),2))
    ax[2,3].set_xlabel('Open')
    ax[2,3].set_ylabel('Close')
    
    ax[2,4].scatter(com['Open'],com['Volume'] )
    ax[2,4].set_title(round(com['Open'].corr(com['Volume']),2))
    ax[2,4].set_xlabel('Open')
    ax[2,4].set_ylabel('Volume')
    
    ax[2,5].scatter(com['Open'],com['Adj Close'] )
    ax[2,5].set_title(round(com['Open'].corr(com['Adj Close']),2))
    ax[2,5].set_xlabel('Open')
    ax[2,5].set_ylabel('Adj Close')
    
    ax[3,0].scatter(com['Close'],com['High'] )
    ax[3,0].set_title(round(com['Close'].corr(com['High']),2))
    ax[3,0].set_xlabel('Close')
    ax[3,0].set_ylabel('High')
    
    ax[3,1].scatter(com['Close'],com['Low'] )
    ax[3,1].set_title(round(com['Close'].corr(com['Low']),2))
    ax[3,1].set_xlabel('Close')
    ax[3,1].set_ylabel('Low')
    
    ax[3,2].scatter(com['Close'],com['Open'] )
    ax[3,2].set_title(round(com['Close'].corr(com['Open']),2))
    ax[3,2].set_xlabel('Close')
    ax[3,2].set_ylabel('Open')
    
    ax[3,3].scatter(com['Close'],com['Close'] )
    ax[3,3].set_title(round(com['Close'].corr(com['Close']),2))
    ax[3,3].set_xlabel('Close')
    ax[3,3].set_ylabel('Close')
    
    ax[3,4].scatter(com['Close'],com['Volume'] )
    ax[3,4].set_title(round(com['Close'].corr(com['Volume']),2))
    ax[3,4].set_xlabel('Close')
    ax[3,4].set_ylabel('Volume')
    
    ax[3,5].scatter(com['Close'],com['Adj Close'] )
    ax[3,5].set_title(round(com['Close'].corr(com['Adj Close']),2))
    ax[3,5].set_xlabel('Close')
    ax[3,5].set_ylabel('Adj Close')
    
    ax[4,0].scatter(com['Volume'],com['High'] )
    ax[4,0].set_title(round(com['Volume'].corr(com['High']),2))
    ax[4,0].set_xlabel('Volume')
    ax[4,0].set_ylabel('High')
    
    ax[4,1].scatter(com['Volume'],com['Low'] )
    ax[4,1].set_title(round(com['Volume'].corr(com['Low']),2))
    ax[4,1].set_xlabel('Volume')
    ax[4,1].set_ylabel('Low')
    
    ax[4,2].scatter(com['Volume'],com['Open'] )
    ax[4,2].set_title(round(com['Volume'].corr(com['Open']),2))
    ax[4,2].set_xlabel('Volume')
    ax[4,2].set_ylabel('Open')
    
    ax[4,3].scatter(com['Volume'],com['Close'] )
    ax[4,3].set_title(round(com['Volume'].corr(com['Close']),2))
    ax[4,3].set_xlabel('Volume')
    ax[4,3].set_ylabel('Close')
    
    ax[4,4].scatter(com['Volume'],com['Volume'] )
    ax[4,4].set_title(round(com['Volume'].corr(com['Volume']),2))
    ax[4,4].set_xlabel('Volume')
    ax[4,4].set_ylabel('Volume')
    
    ax[4,5].scatter(com['Volume'],com['Adj Close'] )
    ax[4,5].set_title(round(com['Volume'].corr(com['Adj Close']),2))
    ax[4,5].set_xlabel('Volume')
    ax[4,5].set_ylabel('Adj Close')
    
    ax[5,0].scatter(com['Adj Close'],com['High'] )
    ax[5,0].set_title(round(com['Adj Close'].corr(com['High']),2))
    ax[5,0].set_xlabel('Adj Close')
    ax[5,0].set_ylabel('High')
    
    ax[5,1].scatter(com['Adj Close'],com['Low'] )
    ax[5,1].set_title(round(com['Adj Close'].corr(com['Low']),2))
    ax[5,1].set_xlabel('Adj Close')
    ax[5,1].set_ylabel('Low')
    
    ax[5,2].scatter(com['Adj Close'],com['Open'] )
    ax[5,2].set_title(round(com['Adj Close'].corr(com['Open']),2))
    ax[5,2].set_xlabel('Adj Close')
    ax[5,2].set_ylabel('Open')
    
    ax[5,3].scatter(com['Adj Close'],com['Close'] )
    ax[5,3].set_title(round(com['Adj Close'].corr(com['Close']),2))
    ax[5,3].set_xlabel('Adj Close')
    ax[5,3].set_ylabel('Close')
    
    ax[5,4].scatter(com['Adj Close'],com['Volume'] )
    ax[5,4].set_title(round(com['Adj Close'].corr(com['Volume']),2))
    ax[5,4].set_xlabel('Adj Close')
    ax[5,4].set_ylabel('Volume')
    
    ax[5,5].scatter(com['Adj Close'],com['Adj Close'] )
    ax[5,5].set_title(round(com['Adj Close'].corr(com['Adj Close']),2))
    ax[5,5].set_xlabel('Adj Close')
    ax[5,5].set_ylabel('Adj Close')


#plt.grid(visible=True, which='both')
    plt.tight_layout()
    plt.show()
    
    #fig.savefig(fname=('Q9'), format='eps', dpi = 500)
# %%
get_sca(aapl)
#%%
get_sca(orcl)
#%%
get_sca(ibm)
#%%
get_sca(msft)
#%%
get_sca(yelp)
#%%
get_sca(tsla)
# %%
