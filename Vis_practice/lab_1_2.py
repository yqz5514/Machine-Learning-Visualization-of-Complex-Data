#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
# 1
np.random.seed(123)

m = int(input("Enter 0 as mean of x: "))
v = int(input("Enter 1 as the variance for x: "))
n = int(input("Enter 1000 as the number of observation for x: "))

m_y = int(input("Enter 5 as the mean for y: "))
v_y = int(input("Enter 2 as the variance for y: "))
n_y = int(input("Enter 1000 as the number of observation for y: "))

x = np.random.normal(loc = m, scale = np.sqrt(v), size = n)
y = np.random.normal(loc = m_y, scale = np.sqrt(v_y), size = n_y)

# %%
# 2 
pearson_corr = (sum ((x - m) * (y - m_y)) ) / np.sqrt(sum(np.power((x - m), 2))* sum(np.power((y - m_y), 2)))
round(pearson_corr, 2)
# %%
# 3 
print(f"The sample mean of random variable x is: {m}")
print(f"The sample mean of random variable y is: {m_y}")
print(f"The sample variance of random variable x is: {v}")
print(f"The sample variance of random variable y is: {v_y}")
print(f"The sample Pearsons correlation coefficient between x & y is: {round(pearson_corr, 2)}")

#%%
# 4
plt.plot(x, label = 'Random Variable x')
plt.plot(y, label = 'Random Variable y')
plt.legend(loc = 1)
plt.title(loc = 'center', label = 'Line plot for random variable x and y')

#%%
# 5
plt.hist(x, label = 'Random Variable x')
plt.hist(y, label = 'Random Variable y')
plt.legend(loc = 1)
plt.title(loc = 'center', label = 'Histogram plot for random variable x and y')

#%%
#6

url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/tute1.csv'
df = pd.read_csv(url, index_col= 0)
print(df.head(5))

#df.isnull().sum()
#df.dtypes
#df.columns

#%%
# 7
def get_pearsoncoef(i, j):
    pearson_corr1 = round((sum ((i - i.mean()) * (j - j.mean())) ) / np.sqrt(sum(np.power((i - i.mean()), 2))* sum(np.power((j - j.mean()), 2))), 2)
    return pearson_corr1

# %%
sales_corr_adbudget = get_pearsoncoef(df['Sales'], df['AdBudget'])
sales_corr_adbudget
# %%
sales_corr_gdp = get_pearsoncoef(df['Sales'], df['GDP'])
sales_corr_gdp
#%%
gdp_corr_adbudget = get_pearsoncoef(df['GDP'], df['AdBudget'])
gdp_corr_adbudget
#%%
# 8 
print(f"The sample Pearson's correlation coefficient between Sales & AdBudget is: {sales_corr_adbudget}" )
print(f"The sample Pearson's correlation coefficient between Sales & GDP is: {sales_corr_gdp}" )
print(f"The sample Pearson's correlation coefficient between AdBudget & GDP is: {gdp_corr_adbudget}" )


#%%
# 9
plt.figure(figsize=(20,15))

plt.plot(df['Sales'], marker = 'o', label = 'Sales', color = 'red' )
plt.plot(df['AdBudget'], marker = 'o', label = 'AdBudget', color = 'green' )
plt.plot(df['GDP'], marker = 'o', label = 'GDP')
plt.legend(loc = 1)
plt.title('Sales, AdBudget and GDP versus time')
plt.xlabel('Date')
plt.ylabel('Amount')
# %%
# 10
plt.figure(figsize=(15,10))

plt.hist(df['Sales'], label = 'Sales', color = 'red' )
plt.hist(df['AdBudget'], label = 'AdBudget', color = 'green' )
plt.hist(df['GDP'], label = 'GDP')
plt.legend(loc = 1)
plt.title('Sales, AdBudget and GDP amount frequency distribution')
plt.xlabel('Amount')
plt.ylabel('Freq')
# %%
from scipy import stats
stats.pearsonr(df['GDP'], df['AdBudget'])
# %%
