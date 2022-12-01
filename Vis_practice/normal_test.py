#%%
from scipy import signal
import numpy as np
from scipy.stats import shapiro
from scipy.stats import kstest
from scipy.stats import normaltest
from scipy.stats import rankdata
#%%
T = 5000
#x = np.random.randn(T) this is noemal distribution
#non normal distribution:
x = np.cumsum(np.random.randn(T))
y = 2*(rankdata(x)/(T+1))-1
y = np.arctanh(y)# complete the process of above 
#%%
shapiro(x, 'raw data')
shapiro(y, 'transformed data')
#%%
kstest(x,'raw data')
kstest(y,'raw data')

#%%
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

def ks_test(x, title):
    mean = np.mean(x)
    std = np.std(x)
    dist = np.random.normal(mean, std, len(x))
    stats, p = kstest(x, dist)
    print('='*50)
    print(f'K-S test: {title} dataset: statistics= {stats:.2f} p-value = {p:.2f}' )

    alpha = 0.01
    if p > alpha :
        print(f'K-S test:  {title} dataset is Normal')
    else:
        print(f'K-S test : {title} dataset is Not Normal')
    print('=' * 50)


def da_k_squared_test(x, title):
    stats, p = normaltest(x)
    print('='*50)
    print(f'da_k_squared test: {title} dataset: statistics= {stats:.2f} p-value = {p:.2f}' )

    alpha = 0.01
    if p > alpha :
        print(f'da_k_squaredtest:  {title} dataset is Normal')
    else:
        print(f'da_k_squared test : {title} dataset is Not Normal')
    print('=' * 50)

#%%
import pandas_datareader as web
#%%
df = web.DataReader('AAPL', data_source = 'yahoo', start='2000-01-01', end='2022-11-30')

# volume = df['volume']
#%%
df.head()
#%%
df[['Volume','Close']].plot()
plt.show()
# %%
# how to transform? use (each value-mean/std)
df1 = ((df[['Volume','Close']] - df[['Volume','Close']].mean())/df[['Volume','Close']].std())
df1[['Volume', 'Close']].plot()
plt.show()
# %%
