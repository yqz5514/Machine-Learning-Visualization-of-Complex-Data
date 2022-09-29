#%%
from prettytable import PrettyTable
# import packages
import pandas_datareader as web 
import numpy as np
import pandas as pd

#%%
#load data
aapl = web.DataReader('AAPL', data_source='yahoo', start = '2000-01-01', end='2022-09-01')
orcl = web.DataReader('ORCL', data_source='yahoo', start = '2000-01-01', end='2022-09-01')
tsla = web.DataReader('TSLA', data_source='yahoo', start = '2000-01-01', end='2022-09-01')
ibm  = web.DataReader('IBM', data_source='yahoo', start = '2000-01-01', end='2022-09-01')
yelp = web.DataReader('YELP', data_source='yahoo', start = '2000-01-01', end='2022-09-01')
msft = web.DataReader('MSFT', data_source='yahoo', start = '2000-01-01', end='2022-09-01')


#%%
# mean
x = PrettyTable()

x.field_names = ['Name','High','Low','Open','Close','Volume','Adj Close']

x.add_row( ['AAPL', round(aapl.High.mean(), 2), round(aapl.Low.mean(), 2), round(aapl.Open.mean(), 2), round(aapl.Close.mean(), 2), round(aapl.Volume.mean(), 2), round(aapl['Adj Close'].mean(), 2)])
x.add_row( ['ORCL', round(orcl.High.mean(), 2), round(orcl.Low.mean(), 2), round(orcl.Open.mean(), 2), round(orcl.Close.mean(), 2), round(orcl.Volume.mean(), 2), round(orcl['Adj Close'].mean(), 2)])
x.add_row( ['TSLA', round(tsla.High.mean(), 2), round(tsla.Low.mean(), 2), round(tsla.Open.mean(), 2), round(tsla.Close.mean(), 2), round(tsla.Volume.mean(), 2), round(tsla['Adj Close'].mean(), 2)])
x.add_row(['IBM', round(ibm.High.mean(), 2), round(ibm.Low.mean(), 2), round(ibm.Open.mean(), 2), round(ibm.Close.mean(), 2), round(ibm.Volume.mean(), 2), round(ibm['Adj Close'].mean(), 2)])
x.add_row( ['YELP', round(yelp.High.mean(), 2), round(yelp.Low.mean(), 2), round(yelp.Open.mean(), 2), round(yelp.Close.mean(), 2), round(yelp.Volume.mean(), 2), round(yelp['Adj Close'].mean(), 2)])
x.add_row( ['MSFT', round(msft.High.mean(), 2), round(msft.Low.mean(), 2), round(msft.Open.mean(), 2), round(msft.Close.mean(), 2), round(msft.Volume.mean(), 2), round(msft['Adj Close'].mean(), 2)])
x.add_row(['Maximum Value',125.17, 122.99,124.04, 124.1,419104965.2, 88.62])
x.add_row(['Minimum Value',28.58,27.95,28.26,28.28,2073301.85,27.12 ])
x.add_row(['Mainmum Company','IBM','IBM','IBM','IBM','AAPL','IBM' ])
x.add_row(['Minimum Company','AAPL', 'AAPL', 'AAPL', 'AAPL', 'YELP', 'AAPL' ])
print(x.get_string(title = 'Mean Value comparison'))
# %%
x = PrettyTable()

x.field_names = ['Name','High','Low','Open','Close','Volume','Adj Close']

x.add_row( ['AAPL', round(aapl.High.var(), 2), round(aapl.Low.var(), 2), round(aapl.Open.var(), 2), round(aapl.Close.var(), 2), round(aapl.Volume.var(), 2), round(aapl['Adj Close'].var(), 2)])
x.add_row(['ORCL', round(orcl.High.var(), 2), round(orcl.Low.var(), 2), round(orcl.Open.var(), 2), round(orcl.Close.var(), 2), round(orcl.Volume.var(), 2), round(orcl['Adj Close'].var(), 2)])
x.add_row(['TSLA', round(tsla.High.var(), 2), round(tsla.Low.var(), 2), round(tsla.Open.var(), 2), round(tsla.Close.var(), 2), round(tsla.Volume.var(), 2), round(tsla['Adj Close'].var(), 2)])
x.add_row(['IBM', round(ibm.High.var(), 2), round(ibm.Low.var(), 2), round(ibm.Open.var(), 2), round(ibm.Close.var(), 2), round(ibm.Volume.var(), 2), round(ibm['Adj Close'].var(), 2)])
x.add_row( ['YELP', round(yelp.High.var(), 2), round(yelp.Low.var(), 2), round(yelp.Open.var(), 2), round(yelp.Close.var(), 2), round(yelp.Volume.var(), 2), round(yelp['Adj Close'].var(), 2)])
x.add_row( ['MSFT', round(msft.High.var(), 2), round(msft.Low.var(), 2), round(msft.Open.var(), 2), round(msft.Close.var(), 2), round(msft.Volume.var(), 2), round(msft['Adj Close'].var(), 2)])
x.add_row(['Maximum Value',9055.73,8208.22,8651.4,8632.56,1.5044144690608e+17,8632.56])
x.add_row(['Minimum Value',235.0 , 216.4 , 226.74 ,225.0 ,6637023846016.25 ,225.0  ])
x.add_row(['Mainmum Company','TSLA','TSLA','TSLA','TSLA','AAPL','TSLA' ])
x.add_row(['Minimum Company','YELP', 'YELP', 'YELP', 'YELP', 'YELP', 'YELP' ])
print(x.get_string(title = 'Variance comparison'))
#%%
x = PrettyTable()

x.field_names = ['Name','High','Low','Open','Close','Volume','Adj Close']
x.add_row(['AAPL', round(aapl.High.std(), 2), round(aapl.Low.std(), 2), round(aapl.Open.std(), 2), round(aapl.Close.std(), 2), round(aapl.Volume.std(), 2), round(aapl['Adj Close'].std(), 2)])
x.add_row(['ORCL', round(orcl.High.std(), 2), round(orcl.Low.std(), 2), round(orcl.Open.std(), 2), round(orcl.Close.std(), 2), round(orcl.Volume.std(), 2), round(orcl['Adj Close'].std(), 2)])
x.add_row(['TSLA', round(tsla.High.std(), 2), round(tsla.Low.std(), 2), round(tsla.Open.std(), 2), round(tsla.Close.std(), 2), round(tsla.Volume.std(), 2), round(tsla['Adj Close'].std(), 2)])
x.add_row(['IBM', round(ibm.High.std(), 2), round(ibm.Low.std(), 2), round(ibm.Open.std(), 2), round(ibm.Close.std(), 2), round(ibm.Volume.std(), 2), round(ibm['Adj Close'].std(), 2)])
x.add_row(['YELP', round(yelp.High.std(), 2), round(yelp.Low.std(), 2), round(yelp.Open.std(), 2), round(yelp.Close.std(), 2), round(yelp.Volume.std(), 2), round(yelp['Adj Close'].std(), 2)])
x.add_row(['MSFT', round(msft.High.std(), 2), round(msft.Low.std(), 2), round(msft.Open.std(), 2), round(msft.Close.std(), 2), round(msft.Volume.std(), 2), round(msft['Adj Close'].std(), 2)])
x.add_row(['Maximum Value',95.16,90.6,93.01,92.91,387867821.44,92.91])
x.add_row(['Minimum Value',15.33,14.71,15.06,15.0,2576242.19,15.0])
x.add_row(['Mainmum Company','TSLA','TSLA','TSLA','TSLA','AAPL','TSLA' ])
x.add_row(['Minimum Company','YELP', 'YELP', 'YELP', 'YELP', 'YELP', 'YELP' ])
print(x.get_string(title = 'Standard Deviation Value comparison'))

# %%
x = PrettyTable()

x.field_names = ['Name','High','Low','Open','Close','Volume','Adj Close']
x.add_row(['AAPL', round(aapl.High.median(), 2), round(aapl.Low.median(), 2), round(aapl.Open.median(), 2), round(aapl.Close.median(), 2), round(aapl.Volume.median(), 2), round(aapl['Adj Close'].median(), 2)])
x.add_row(['ORCL', round(orcl.High.median(), 2), round(orcl.Low.median(), 2), round(orcl.Open.median(), 2), round(orcl.Close.median(), 2), round(orcl.Volume.median(), 2), round(orcl['Adj Close'].median(), 2)])
x.add_row(['TSLA', round(tsla.High.median(), 2), round(tsla.Low.median(), 2), round(tsla.Open.median(), 2), round(tsla.Close.median(), 2), round(tsla.Volume.median(), 2), round(tsla['Adj Close'].median(), 2)])
x.add_row(['IBM', round(ibm.High.median(), 2), round(ibm.Low.median(), 2), round(ibm.Open.median(), 2), round(ibm.Close.median(), 2), round(ibm.Volume.median(), 2), round(ibm['Adj Close'].median(), 2)])
x.add_row(['YELP', round(yelp.High.median(), 2), round(yelp.Low.median(), 2), round(yelp.Open.median(), 2), round(yelp.Close.median(), 2), round(yelp.Volume.median(), 2), round(yelp['Adj Close'].median(), 2)])
x.add_row(['MSFT', round(msft.High.median(), 2), round(msft.Low.median(), 2), round(msft.Open.median(), 2), round(msft.Close.median(), 2), round(msft.Volume.median(), 2), round(msft['Adj Close'].median(), 2)])
x.add_row(['Maximum Value',122.59,120.61,121.57,121.6,302513400.0,94.58])
x.add_row(['Minimum Value',12.48  ,12.31,12.37,12.38,1425450.0 ,10.57])
x.add_row(['Mainmum Company','IBM','IBM','IBM','IBM','AAPL','IBM' ])
x.add_row(['Minimum Company','AAPL', 'AAPL', 'AAPL', 'AAPL', 'YELP', 'AAPL'])

print(x.get_string(title = 'Median Value comparison'))

# %%
msft.corr()
#%%
print(orcl.corr())
#%%
print(tsla.corr())
#%%
print(ibm.corr())
print(yelp.corr())
print(msft.corr())
# %%
