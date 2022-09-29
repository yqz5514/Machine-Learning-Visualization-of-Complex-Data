
#%%
import pandas as pd
import numpy as np
from prettytable import PrettyTable



#%%
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/tute1.csv'
df = pd.read_csv(url, index_col= 0)
print(df.head(5))

#%%
#1- Load the “tute1.csv” dataset. Write a python program that calculate the correlation coefficient between Sales and AdBudget and display the following message on the console.
s_a_corr = df['Sales'].corr(df['AdBudget'])
print(f'Correlation Coefficient between Sales and AdBudget is:{round(s_a_corr, 2)}')
# %%
#2Write a python program that calculate the correlation coefficient between AdBudget and GDP and display the following message on the console:
a_g_corr = df['GDP'].corr(df['AdBudget'])
print(f'Correlation Coefficient between GDP and AdBudget is:{round(a_g_corr, 2)}')

# %%
#3Write a python program that calculate coefficient between Sales and GDP and display the following message on the console:
s_g_corr = df['GDP'].corr(df['Sales'])
print(f'Correlation Coefficient between GDP and Sales is:{round(s_g_corr, 2)}')

#%%
# Using the hypothesis test (t-test) show whether the correlation coefficients in step 2, 3, and 4 are statistically significant? 
# Assume the level of confident to be 95% with two tails (𝛼=0.05 ).

def ttest(r,n):
    t = (r*np.sqrt(len(n)-2))/np.sqrt(1-np.power(r,2))
    return t

#%%
print(f'The t-value of Sales and AdBudget is {ttest(s_a_corr, df):.2f}')
print(f'The t-value of GDP and AdBudget is {ttest(a_g_corr, df):.2f}')
print(f'The t-value of Sales and GDP is {ttest(s_g_corr, df):.2f}')

# %%
len(df)
# %%
import scipy.stats as stats
def sci_ttest(i,j):
    result = stats.ttest_ind(a=i, b=j, equal_var=True)
    return result

#%%
sci_ttest(df.GDP,df.AdBudget)
# %%
#5
def pcorr(ab,ac,bc):
    r = (ab-ac*bc)/(np.sqrt(1-np.power(ac,2))*np.sqrt(1-np.power(bc,2)))
    return r

def pcorr_t(p_corr):
    t0 = p_corr*(np.sqrt((len(df)-2-1)/(1-np.power(p_corr,2))))
    return t0
#%%
# partial correlation coef between sales and adbudget
pcorr_sa = pcorr(s_a_corr,s_g_corr,a_g_corr)
pcorr_sg = pcorr(s_g_corr,s_a_corr,a_g_corr)
pcorr_ag = pcorr(a_g_corr,s_a_corr,s_g_corr)
print(pcorr_sa,pcorr_sg,pcorr_ag)
# %%
pct_sa = pcorr_t(pcorr_sa)
pct_sg = pcorr_t(pcorr_sg)
pct_ag = pcorr_t(pcorr_ag)
print(pct_sa,pct_sg,pct_ag)

#%%
print(f'The partial correlation coef between Sales and Adbudget is {pcorr_sa:.2f}, and the t-value of that is {pct_sa:.2f}')
print(f'The partial correlation coef between Sales and GDP is {pcorr_sg:.2f}, and the t-value of that is {pct_sg:.2f}')
print(f'The partial correlation coef between GDP and Adbudget is {pcorr_ag:.2f}, and the t-value of that is {pct_ag:.2f}')

# %%
# table
x = PrettyTable()

x.field_names = ['Group','Correlation Coef','Correlation Coef T-Value','Partial Correlation Coef','Parcial Correlation Coef T-Value']

x.add_row( ['Sales AdBudget', round(s_a_corr, 2), round(ttest(s_a_corr, df), 2),round(pcorr_sa, 2),round(pct_sa, 2)])
x.add_row( ['Sales GDP', round(s_g_corr, 2), round(ttest(s_g_corr, df), 2),round(pcorr_sg, 2),round(pct_sg, 2)])
x.add_row( ['AdBudget GDP', round(a_g_corr, 2),round(ttest(a_g_corr, df),2), round(pcorr_ag, 2),round(pct_ag, 2)])

print(x.get_string(title = 'Comparision of Correlation Coef with t-value and Partial Correlation Coef with t-value'))

# %%
