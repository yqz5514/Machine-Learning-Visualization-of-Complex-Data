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
df_fel.shape
# %%
df_fel = df_fel[0:100]
# %%
df_fel.columns
# %%
df_fel[['Height','Weight']] = round(df_fel[['Height','Weight']], 2)
# %%
df_fel.head()
# %%
fig, axes = plt.subplots(nrows=1, ncols=2,sharex=True, sharey=True)
df_fel.hist(ax=axes)
plt.suptitle('Felmale Height and Wieght distribution',ha='center', fontsize='xx-large')
fig.text(0.04, 0.5, 'Freq', va='center', rotation='vertical')
fig.text(0.5, 0.04, 'Value under each category', ha='center')

# %%
#z-transform
df_z = ((df_fel[['Height','Weight']] - df_fel[['Height','Weight']].mean())/df_fel[['Height','Weight']].std())
df_z[['Height','Weight']].plot()
plt.show()
#%%
df_z[['Height','Weight']] = round(df_z[['Height','Weight']], 2)

#%%
df_z.head()
# %%
df_z[['Height','Weight']].hist()
#%%
import numpy as np
#outlier
q1_d, q3_d = np.percentile(df_z['Height'], [25,75])
q1_a, q3_a = np.percentile(df_z['Weight'], [25,75])


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
print(f'IQR of height of female is {IQR_d}')
print(f'IQR of weight of female is {IQR_a}')
print(f'Any height lower than {low_outlier_d:.2f} inches and higher than {high_outlier_d:.2f} inches consider as outlier')
print(f'Any weight lower than {low_outlier_a:.2f} inches and higher than {high_outlier_a:.2f} inches consider as outliers')

# %%
sns.boxplot(data = df_z)
plt.show()
# %%
df_remove_out = df_z.copy()
# %%
df_remove_out['Height'] = df_remove_out[(df_remove_out['Height']>=-2.6175)&(df_remove_out['Height']<=2.6425)]
#%%
remove_out_weight = df_remove_out[(df_remove_out['Weight']>=-2.23)&(df_remove_out['Weight']<=2.27)]

#%%
sns.boxplot(data = remove_out_weight
            )
plt.show()
# %%
import statsmodels
from statsmodels.graphics.gofplots import qqplot
plt.figure()
qqplot(df_fel['Height'],line='s')
plt.title('Height of female')
plt.show()
# %%
plt.figure()
qqplot(df_fel['Weight'],line='s')
plt.title('Weight of female')
plt.show()
# %%
plt.figure()
qqplot(df_z['Weight'],line='s')
plt.title('Weight of female(z-transformed)')
plt.show()
plt.figure()
qqplot(df_z['Height'],line='s')
plt.title('Height of female(z-transformed)')
plt.show()
# %%
import numpy as np
from scipy.stats import shapiro
from scipy.stats import kstest
from scipy.stats import normaltest
from scipy.stats import rankdata
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

shapiro_test(df_fel['Height'],'raw data')
shapiro_test(df_z['Height'],'Transformed data')
shapiro_test(df_fel['Weight'],'raw data')
shapiro_test(df_z['Weight'],'Transformed data')
#%%
#========================
#= ks_test test Shapiro
#========================
ks_test(df_fel['Height'],'raw data')
ks_test(df_z['Height'],'Transformed data')
ks_test(df_fel['Weight'],'raw data')
ks_test(df_z['Weight'],'Transformed data')
#%%
#========================
#= da_k_squared_test
# #========================
print(100*'=')
da_k_squared_test(x,'raw data')
da_k_squared_test(y,'Transformed data')
print(100*'=')
mean = 194
std = 11.2
L = 175
U = 225
print(f'the probability that '
      f'the observation between '
      f'{U} and {L} is '
      f'{(st.norm(mean,std).cdf(U)-st.norm(mean,std).cdf(L))*100:.2f}%') 