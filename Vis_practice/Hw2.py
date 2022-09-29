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


fig, ax = plt.subplots(3,2,figsize=(16,8))

ax[0,0].plot(aapl.High)
ax[0,0].set_title('High price history of AAPL')
ax[0,0].set_xlabel('Date')
ax[0,0].set_ylabel('High price USD($)')

ax[1,0].plot(tsla.High)
ax[1,0].set_title('High price history of TSLA')
ax[1,0].set_xlabel('Date')
ax[1,0].set_ylabel('High price USD($)')

ax[2,0].plot(yelp.High)
ax[2,0].set_title('High price history of YELP')
ax[2,0].set_xlabel('Date')
ax[2,0].set_ylabel('High price USD($)')

ax[0,1].plot(orcl.High)
ax[0,1].set_title('High price history of ORCL')
ax[0,1].set_xlabel('Date')
ax[0,1].set_ylabel('High price USD($)')

ax[1,1].plot(ibm.High)
ax[1,1].set_title('High price history of IBM')
ax[1,1].set_xlabel('Date')
ax[1,1].set_ylabel('High price USD($)')

ax[2,1].plot(msft.High)
ax[2,1].set_title('High price history of MSFT')
ax[2,1].set_xlabel('Date')
ax[2,1].set_ylabel('High price USD($)')

#plt.grid(visible=True, which='both')
plt.tight_layout()
plt.show()
# how to do the grid....
#%%
fig = plt.figure(figsize=(16,8))
plt.plot(aapl.High)
plt.xlabel('Date')
plt.ylabel('Amount')
plt.show()
# %%
