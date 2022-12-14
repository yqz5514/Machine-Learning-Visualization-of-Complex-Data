import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
from scipy.stats import rankdata
from statsmodels.graphics.gofplots import qqplot
import pandas_datareader as web
import sys
sys.path.append(r'C:\GW\Time series Analysis\toolbox')
from toolbox import shapiro_test, ks_test, da_k_squared_test

df = web.DataReader('AAPL',data_source = 'yahoo',start = '2000-01-01',
                                                         end = '2022-11-30')
df[['Volume','Close']].plot()
plt.show()

df1 = (df[['Volume','Close']]- df[['Volume','Close']].mean())/df[['Volume','Close']].std()

df1[['Volume','Close']].plot()
plt.show()

# T = 5000
# x = np.cumsum(np.random.randn(T))
# y = 2*(rankdata(x)/(T+1))-1
# y = np.arctanh(y)
#
# fig,ax = plt.subplots(2,2, figsize = (9,7))
#
# ax[0,0].plot(x)
# ax[0,0].set_title('Original Non-gaussian data')
# ax[0,0].grid()
#
# ax[0,1].plot(y)
# ax[0,1].set_title('Transformed data (Gaussian) ')
# ax[0,1].grid()
#
# ax[1,0].hist(x,bins=50)
# ax[1,0].set_title('Histogram of Original Non-gaussian data')
# ax[1,0].grid()
# ax[1,1].hist(y,bins=50)
# ax[1,1].set_title('Histogram of Transformed data (Gaussian)')
# ax[1,1].grid()
# plt.tight_layout()
# plt.show()
#
#
# plt.show()
#
#
# plt.figure()
# qqplot(x, line='s')
# plt.title('Raw data')
# plt.show()
#
# plt.figure()
# qqplot(y, line='s')
# plt.title('Transformed data')
# plt.show()
#========================
#= Normality test Shapiro
#========================
# print(100*'=')
# shapiro_test(x,'raw data')
# shapiro_test(y,'Transformed data')
# print(100*'=')
#========================
#= ks_test test Shapiro
#========================
# print(100*'=')
# ks_test(x,'raw data')
# ks_test(y,'Transformed data')
# print(100*'=')
#========================
#= da_k_squared_test
# #========================
# print(100*'=')
# da_k_squared_test(x,'raw data')
# da_k_squared_test(y,'Transformed data')
# print(100*'=')
# mean = 194
# std = 11.2
# L = 175
# U = 225
# print(f'the probability that '
#       f'the observation between '
#       f'{U} and {L} is '
#       f'{(st.norm(mean,std).cdf(U)-st.norm(mean,std).cdf(L))*100:.2f}%')