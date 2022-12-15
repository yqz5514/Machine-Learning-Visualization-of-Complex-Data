#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%%%
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/weight-height.csv'
df = pd.read_csv(url)
print(df.head(5))
# %%
#1 
df_fel = df[df['Gender']=='Female']

# %%
df_fel.head()
# %%
df_fel = df_fel[0:100]
# %%
df_fel.reset_index(inplace=True)
# %%
df_fel[['Height','Weight']] = round(df_fel[['Height','Weight']], 2)
# %%
df_fel.drop(columns='index', inplace=True)
#%%
df_fel.head()
#%%
df_fel.plot(kind = 'line',
            figsize=(12,6),
            title='Height and weight vs number of observation',
            xlabel='number of observation',
            ylabel='value')
#%%
df_fel.plot(kind = 'hist',
            figsize=(12,6),
            title='Height and weight vs number of observation',
            
            )
# %%
# fig, axes = plt.subplots(nrows=1, ncols=2,sharex=True, sharey=True)
# df_fel.hist(ax=axes)
# plt.suptitle('Felmale Height and Wieght distribution',ha='center', fontsize='xx-large')
# fig.text(0.04, 0.5, 'Freq', va='center', rotation='vertical')
# fig.text(0.5, 0.04, 'Value under each category', ha='center')

# %%
#z-transform
df_z = ((df_fel[['Height','Weight']] - df_fel[['Height','Weight']].mean())/df_fel[['Height','Weight']].std())

#%%
df_z[['Height','Weight']] = round(df_z[['Height','Weight']], 2)

#%%
df_z.head()
# %%
df_z.plot(kind = 'line',
            figsize=(12,6),
            title='Height and weight vs number of observation for transformed data',
            xlabel='number of observation',
            ylabel='value')
#%%
df_z.plot(kind = 'hist',
            figsize=(12,6),
            title='Height and weight vs number of observation for transformed data',
            )
#%%
fig, axes = plt.subplots(nrows=2, ncols=2)
df_fel.plot(kind = 'line',
            figsize=(12,6),
            title='Height and weight vs number of observation',
            xlabel='number of observation',
            ylabel='value',
            ax = axes[0,0])
df_z.plot(kind = 'line',
            figsize=(12,6),
            title='Height and weight vs number of observation for transformed data',
            xlabel='number of observation',
            ylabel='value',
            ax=axes[0,1])
df_fel.plot(kind = 'hist',
            figsize=(12,6),
            title='Height and weight vs number of observation',
            ax=axes[1,0]
            )
df_z.plot(kind = 'hist',
            figsize=(12,6),
            title='Height and weight vs number of observation for transformed data',
            ax=axes[1,1])
plt.tight_layout()
plt.show()
#%%
more_170 = len(df_fel[df_fel['Weight']>170])/len(df_fel)*100
taller_66 = len(df_fel[df_fel['Height']>66])/len(df_fel)*100
#%%
print(f'Sample mean of the ladies weight is {df_fel.Weight.mean():.2f} lb')
print(f'Sample mean of the ladies height is {df_fel.Height.mean():.2f} lb')
print(f'Sample std of the ladies weight is {df_fel.Height.std():.2f} lb')
print(f'Sample std of the ladies weight is {df_fel.Weight.std():.2f} lb')
print(f'Sample median of the ladies weight is {df_fel.Height.median():.2f} lb')
print(f'Sample median of the ladies weight is {df_fel.Height.median():.2f} lb')
print(f'The probability that a lady weight more than 170lb is {more_170:.2f} %')
print(f'The probability that a lady Height more than 66 inches is {taller_66:.2f} %')
#%%
import statsmodels
from statsmodels.graphics.gofplots import qqplot

plt.figure()
qqplot(df_fel['Height'],line='s')
plt.title('Height of female')
plt.show()
#%%
plt.figure()
qqplot(df_fel['Weight'],line='s')
plt.title('Weight of female')
plt.show()

# %%
import numpy as np
from scipy.stats import shapiro
from scipy.stats import kstest
from scipy.stats import normaltest

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
   #========================
#= Normality test Shapiro
#========================

shapiro_test(df_fel['Height'],'Height felmale data')

shapiro_test(df_fel['Weight'],'Weight female data')

#%%
#========================
#= ks_test test Shapiro
#========================
ks_test(df_fel['Height'],'Height felmale data')

ks_test(df_fel['Weight'],'Weight female data')

#%%
#========================
#= da_k_squared_test
# #========================

da_k_squared_test(df_fel['Height'],'Height felmale data')
da_k_squared_test(df_fel['Weight'],'Weight female data')
#%%
import numpy as np
#outlier
q1_d, q3_d = np.percentile(df_fel['Height'], [25,75])
q1_a, q3_a = np.percentile(df_fel['Weight'], [25,75])


IQR_d = q3_d - q1_d
IQR_a = q3_a - q1_a



low_outlier_d = q1_d - 1.5*IQR_d
high_outlier_d = q3_d + 1.5*IQR_d

low_outlier_a = q1_a - 1.5*IQR_a
high_outlier_a = q3_a + 1.5*IQR_a

# %%
outlier_table = {'Outlier Boundary': ['low limit','high limit'],
                 'Height':[low_outlier_d, high_outlier_d],
                 'Weight':[low_outlier_a, high_outlier_a],}

df_outlier_table = pd.DataFrame(outlier_table)
print(df_outlier_table)
#%%
print(f'Q1 of height of female is {low_outlier_d:.2f}, Q3 is {high_outlier_d:.2f}')
print(f'IQR of height of female is {IQR_d:.2f}')
print(f'Any height lower than {low_outlier_d:.2f} inches and higher than {high_outlier_d:.2f} inches consider as outlier')
#%%
sns.boxplot(df_fel.Height)
#%%
df_rem_h = df_fel[(df_fel['Height']>=56.11) &(df_fel['Height']<=71.23)]

#%%
sns.boxplot(df_rem_h.Height)
#%%
df_fel.shape
#%%
df_rem_h.shape

#%%
print(f'Q1 of weight of female is {low_outlier_a:.2f}, Q3 is {high_outlier_a:.2f}')
print(f'IQR of weight of female is {IQR_a:.2f}')
print(f'Any weight lower than {low_outlier_a:.2f} and higher than {high_outlier_a:.2f} consider as outliers')

#%%
sns.boxplot(df_fel.Weight)

# %%
df_remove_out = df_fel[(df_fel['Weight']>=90.94) &(df_fel['Weight']<=178.07)]

#%%
sns.boxplot(df_remove_out.Weight
            )
plt.show()
    
