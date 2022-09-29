#%%
# import packages
import pandas_datareader as web 
import numpy as np
import pandas as pd
from tabulate import tabulate

#%%
aapl = web.DataReader('AAPL', data_source='yahoo', start = '2000-01-01', end='2022-09-01')
orcl = web.DataReader('ORCL', data_source='yahoo', start = '2000-01-01', end='2022-09-01')
tsla = web.DataReader('TSLA', data_source='yahoo', start = '2000-01-01', end='2022-09-01')
ibm  = web.DataReader('IBM', data_source='yahoo', start = '2000-01-01', end='2022-09-01')
yelp = web.DataReader('YELP', data_source='yahoo', start = '2000-01-01', end='2022-09-01')
msft = web.DataReader('MSFT', data_source='yahoo', start = '2000-01-01', end='2022-09-01')

#%%
aapl.columns
# %%
# mean
header = ['Name\Features','High','Low','Open','Close','Volume','Adj Close']
title_m = ['Mean']
aapl_m = ['AAPL', round(aapl.High.mean(), 2), round(aapl.Low.mean(), 2), round(aapl.Open.mean(), 2), round(aapl.Close.mean(), 2), round(aapl.Volume.mean(), 2), round(aapl['Adj Close'].mean(), 2)]
orcl_m = ['ORCL', round(orcl.High.mean(), 2), round(orcl.Low.mean(), 2), round(orcl.Open.mean(), 2), round(orcl.Close.mean(), 2), round(orcl.Volume.mean(), 2), round(orcl['Adj Close'].mean(), 2)]
tsla_m = ['TSLA', round(tsla.High.mean(), 2), round(tsla.Low.mean(), 2), round(tsla.Open.mean(), 2), round(tsla.Close.mean(), 2), round(tsla.Volume.mean(), 2), round(tsla['Adj Close'].mean(), 2)]
ibm_m = ['IBM', round(ibm.High.mean(), 2), round(ibm.Low.mean(), 2), round(ibm.Open.mean(), 2), round(ibm.Close.mean(), 2), round(ibm.Volume.mean(), 2), round(ibm['Adj Close'].mean(), 2)]
yelp_m = ['YELP', round(yelp.High.mean(), 2), round(yelp.Low.mean(), 2), round(yelp.Open.mean(), 2), round(yelp.Close.mean(), 2), round(yelp.Volume.mean(), 2), round(yelp['Adj Close'].mean(), 2)]
msft_m = ['MSFT', round(msft.High.mean(), 2), round(msft.Low.mean(), 2), round(msft.Open.mean(), 2), round(msft.Close.mean(), 2), round(msft.Volume.mean(), 2), round(msft['Adj Close'].mean(), 2)]
max_m = ['Maximum Value', 'IBM','IBM','IBM','IBM','AAPL','IBM']
min_m = ['Minimum Value', 'AAPL', 'AAPL', 'AAPL', 'AAPL', 'IBM', 'AAPL']
# also has max_value
# min_value
table_m = [title_m,aapl_m, orcl_m, tsla_m, ibm_m, yelp_m, msft_m, max_m, min_m]
print(tabulate(table_m, headers=header, tablefmt='fancy_grid'))
# %%
#Variance
header = ['Name\Features','High','Low','Open','Close','Volume','Adj Close']
aapl_v = ['AAPL', round(aapl.High.var(), 2), round(aapl.Low.var(), 2), round(aapl.Open.var(), 2), round(aapl.Close.var(), 2), round(aapl.Volume.var(), 2), round(aapl['Adj Close'].var(), 2)]
orcl_v  = ['ORCL', round(orcl.High.var(), 2), round(orcl.Low.var(), 2), round(orcl.Open.var(), 2), round(orcl.Close.var(), 2), round(orcl.Volume.var(), 2), round(orcl['Adj Close'].var(), 2)]
tsla_v  = ['TSLA', round(tsla.High.var(), 2), round(tsla.Low.var(), 2), round(tsla.Open.var(), 2), round(tsla.Close.var(), 2), round(tsla.Volume.var(), 2), round(tsla['Adj Close'].var(), 2)]
ibm_v = ['IBM', round(ibm.High.var(), 2), round(ibm.Low.var(), 2), round(ibm.Open.var(), 2), round(ibm.Close.var(), 2), round(ibm.Volume.var(), 2), round(ibm['Adj Close'].var(), 2)]
yelp_v = ['YELP', round(yelp.High.var(), 2), round(yelp.Low.var(), 2), round(yelp.Open.var(), 2), round(yelp.Close.var(), 2), round(yelp.Volume.var(), 2), round(yelp['Adj Close'].var(), 2)]
msft_v = ['MSFT', round(msft.High.var(), 2), round(msft.Low.var(), 2), round(msft.Open.var(), 2), round(msft.Close.var(), 2), round(msft.Volume.var(), 2), round(msft['Adj Close'].var(), 2)]
max_v = ['Maximum Value', 'TSLA','TSLA','TSLA','TSLA','AAPL','TSLA']
min_v = ['Minimum Value', 'YELP', 'YELP', 'YELP', 'YELP', 'YELP', 'YELP']
table_v = [aapl_v, orcl_v, tsla_v, ibm_v, yelp_v, msft_v, max_v, min_v]
print(tabulate(table_v, headers=header, tablefmt='fancy_grid'))

# %%
#std
header = ['Name\Features','High','Low','Open','Close','Volume','Adj Close']
aapl_std = ['AAPL', round(aapl.High.std(), 2), round(aapl.Low.std(), 2), round(aapl.Open.std(), 2), round(aapl.Close.std(), 2), round(aapl.Volume.std(), 2), round(aapl['Adj Close'].std(), 2)]
orcl_std  = ['ORCL', round(orcl.High.std(), 2), round(orcl.Low.std(), 2), round(orcl.Open.std(), 2), round(orcl.Close.std(), 2), round(orcl.Volume.std(), 2), round(orcl['Adj Close'].std(), 2)]
tsla_std  = ['TSLA', round(tsla.High.std(), 2), round(tsla.Low.std(), 2), round(tsla.Open.std(), 2), round(tsla.Close.std(), 2), round(tsla.Volume.std(), 2), round(tsla['Adj Close'].std(), 2)]
ibm_std = ['IBM', round(ibm.High.std(), 2), round(ibm.Low.std(), 2), round(ibm.Open.std(), 2), round(ibm.Close.std(), 2), round(ibm.Volume.std(), 2), round(ibm['Adj Close'].std(), 2)]
yelp_std = ['YELP', round(yelp.High.std(), 2), round(yelp.Low.std(), 2), round(yelp.Open.std(), 2), round(yelp.Close.std(), 2), round(yelp.Volume.std(), 2), round(yelp['Adj Close'].std(), 2)]
msft_std = ['MSFT', round(msft.High.std(), 2), round(msft.Low.std(), 2), round(msft.Open.std(), 2), round(msft.Close.std(), 2), round(msft.Volume.std(), 2), round(msft['Adj Close'].std(), 2)]
max_std = ['Maximum Value', 'TSLA','TSLA','TSLA','TSLA','AAPL','TSLA']
min_std = ['Minimum Value', 'YELP', 'YELP', 'YELP', 'YELP', 'YELP', 'YELP']
table_std = [aapl_std, orcl_std, tsla_std, ibm_std, yelp_std, msft_std, max_std, min_std]
print(tabulate(table_std, headers=header, tablefmt='fancy_grid'))

# %%
#median
header = ['Name\Features','High','Low','Open','Close','Volume','Adj Close']
aapl_md = ['AAPL', round(aapl.High.median(), 2), round(aapl.Low.median(), 2), round(aapl.Open.median(), 2), round(aapl.Close.median(), 2), round(aapl.Volume.median(), 2), round(aapl['Adj Close'].median(), 2)]
orcl_md  = ['ORCL', round(orcl.High.median(), 2), round(orcl.Low.median(), 2), round(orcl.Open.median(), 2), round(orcl.Close.median(), 2), round(orcl.Volume.median(), 2), round(orcl['Adj Close'].median(), 2)]
tsla_md  = ['TSLA', round(tsla.High.median(), 2), round(tsla.Low.median(), 2), round(tsla.Open.median(), 2), round(tsla.Close.median(), 2), round(tsla.Volume.median(), 2), round(tsla['Adj Close'].median(), 2)]
ibm_md = ['IBM', round(ibm.High.median(), 2), round(ibm.Low.median(), 2), round(ibm.Open.median(), 2), round(ibm.Close.median(), 2), round(ibm.Volume.median(), 2), round(ibm['Adj Close'].median(), 2)]
yelp_md = ['YELP', round(yelp.High.median(), 2), round(yelp.Low.median(), 2), round(yelp.Open.median(), 2), round(yelp.Close.median(), 2), round(yelp.Volume.median(), 2), round(yelp['Adj Close'].median(), 2)]
msft_md = ['MSFT', round(msft.High.median(), 2), round(msft.Low.median(), 2), round(msft.Open.median(), 2), round(msft.Close.median(), 2), round(msft.Volume.median(), 2), round(msft['Adj Close'].median(), 2)]
max_md = ['Maximum Value', 'IBM','IBM','IBM','IBM','AAPL','IBM']
min_md = ['Minimum Value', 'AAPL', 'AAPL', 'AAPL', 'AAPL', 'IBM', 'AAPL']
table_md = [aapl_md, orcl_md, tsla_md, ibm_md, yelp_md, msft_md, max_md, min_md]
print(tabulate(table_md, headers=header, tablefmt='fancy_grid'))
# %%
table_m.corr()
# %%
