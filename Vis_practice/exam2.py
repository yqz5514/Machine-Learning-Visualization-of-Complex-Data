#%%
import dash as dash
from dash import dcc 
from dash import html
from dash.dependencies import Input, Output 
import plotly.express as px 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly
import numpy as np
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import datetime
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy import linalg as LA
import statsmodels.api as sm
from scipy import stats
from scipy.stats import shapiro
from scipy.stats import normaltest
import dash as dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
pd.set_option("display.max_columns", 100)
np.set_printoptions(linewidth=100)

plt.style.use('seaborn-whitegrid')
#%%
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/Metro_Interstate_Traffic_Volume.csv'
df = pd.read_csv(url)
df.head(5)
#%%
df_test = df.copy()
# %%
df.describe()
# %%
df.isnull().sum()
# %%
df.duplicated(subset='date_time').sum()
# %%
df_test = df_test.drop_duplicates(subset='date_time', keep ='last')
# %%
df_test.duplicated(subset='date_time').sum()
# %%
df_test.isnull().sum()
# %%
df_test.columns
#%%
#PCA
feature_pca = ['temp', 'rain_1h', 'snow_1h', 'clouds_all']
df_pca_m = df_test[feature_pca]
df_pca_m.head()
# %%
X = df_pca_m.values
X = StandardScaler().fit_transform(X)

pca = PCA(n_components='mle', svd_solver='full')
pca.fit(X)
X_PCA = pca.transform(X)

print('Original Dim', X.shape)
print('Transformed Dim', X_PCA.shape)
print(f'Explained variance ratio {pca.explained_variance_ratio_ }')
# %%
plt.figure()
x = np.arange(1, len(np.cumsum(pca.explained_variance_ratio_))+1, 1)
plt.xticks(x)
plt.plot(x, np.cumsum(pca.explained_variance_ratio_))
plt.grid()
plt.title('PCA plot of 4 numerical features')
plt.xlabel('Features')
plt.ylabel('Explained Variance Ratio')
plt.show()
# %%
X = df_pca_m.values
X = StandardScaler().fit_transform(X)

pca = PCA(n_components=4, svd_solver='full')
pca.fit(X)
X_PCA = pca.transform(X)

print('Original Dim', X.shape)
print('Transformed Dim', X_PCA.shape)
print(f'Explained variance ratio {pca.explained_variance_ratio_ }')

# %%
plt.figure()
x = np.arange(1, len(np.cumsum(pca.explained_variance_ratio_))+1, 1)
plt.xticks(x)
plt.plot(x, np.cumsum(pca.explained_variance_ratio_))
plt.grid()
plt.title('PCA plot of 4 numerical features')
plt.xlabel('Features')
plt.ylabel('Explained Variance Ratio')
plt.show()
# %%
# SVD Analysis and condition number
X = df_pca_m.values
X = StandardScaler().fit_transform(X)

pca = PCA(n_components='mle', svd_solver='full')
pca.fit(X)
X_PCA = pca.transform(X)

H = np.matmul(X.T, X)
_, d, _ = np.linalg.svd(H)
print('*'*100)
print(f'Original Data: Singular Values {d}')
print(f'Original Data: condition number {LA.cond(X)}')
print('*'*100)
# %%
print('*'*100)
H_PCA = np.matmul(X_PCA.T, X_PCA)
_, d_PCA, _ = np.linalg.svd(H_PCA)
print(f'Transformed Data: Singular Values {d_PCA}')
print(f'Transformed Data: condition number {LA.cond(X_PCA)}')
print('*'*100)
# %%
q1_tv, q3_tv = np.percentile(df_test['traffic_volume'], [25,75])

IQR_tv = q3_tv - q1_tv

low_outlier_tv = q1_tv - 1.5*IQR_tv
high_outlier_tv = q3_tv + 1.5*IQR_tv
print(f'Q1 is {q1_tv:.2f}, Q3 is {q3_tv:.2f}')
print(f'Any traffic volume less than {low_outlier_tv:.2f} or more than {high_outlier_tv:.2f} consider as outlier')
print(f'IQR is {IQR_tv:.2f}')
#%%
df_clean = df_test[(df_test['traffic_volume']>=-4306.75)&(df_test['traffic_volume']<=10507.25)]
# %%
sns.boxplot(data = df_test,
            )
plt.show
#%%
sns.boxplot(df_clean.traffic_volume)
plt.show
# %%
df_test.shape
# %%
df_clean.shape
# %%
q1_t, q3_t = np.percentile(df_test['temp'], [25,75])

IQR_t = q3_t - q1_t

low_outlier_t = q1_t - 1.5*IQR_t
high_outlier_t= q3_t + 1.5*IQR_t
print(f'Q1 is {q1_t:.2f}, Q3 is {q3_t:.2f}')
print(f'Any temp less than {low_outlier_t:.2f} or more than {high_outlier_t:.2f} consider as outlier')
print(f'IQR is {IQR_t:.2f}')
# %%
df_clean = df_test[(df_test['temp']>=241.18)&(df_test['temp']<=322.94)]

# %%
sns.boxplot(df_clean.temp)
plt.show
#%%
plt.figure(figsize=(12,10))
sns.heatmap(df_clean.corr(), annot=True, cmap='Blues')
plt.title('Heatmap of the correlation coef between features and dependent variable', fontsize=20)
plt.tight_layout
plt.show()
# %%
df_clean.columns
# %%
weather_group = df_clean.groupby('weather_main').count()
#%%
sns.barplot(data = weather_group,
             x = weather_group.index,
             y = 'traffic_volume',
             orient= 'veritical'
             )

plt.xticks(rotation = 90)
plt.show
# %%
sns.displot(data=df_clean,
             x='traffic_volume',
             hue='weather_main',
             #binwidth=3,
             #bins=30,
             element='poly'
             
)
plt.show()
# %%
sns.displot(data=df_clean,
             x='traffic_volume',
             hue='weather_main',
             #binwidth=3,
            # bins=30,
             element='bars'
             
)
plt.show()
# %%
sns.displot(data=df_clean,
             x='traffic_volume',
             hue='weather_main',
            
             element='step'
             
)
plt.show()
# %%
sns.displot(data=df_clean,
             x='traffic_volume',
             hue='weather_main',
             multiple='stack',
            #  binwidth=3,
            #  bins=30              
)
plt.show()
# %%
sns.displot(data=df_clean,
             x='traffic_volume',
             hue='weather_main',
            
            kind='kde',
            fill = True
            )
# %%

    
#%%
df_clean.columns
#%%
df_num1 = df_clean.select_dtypes(include = ['float64', 'int64'])
df_num1.columns


#%%
from statsmodels.graphics.gofplots import qqplot
plt.figure()
qqplot(df_clean['traffic_volume'],line='s')
plt.title('Traffic volume')
plt.show()
# %%
plt.figure()
qqplot(df_clean['temp'],line='s')
plt.title('Temp')
plt.show()
# %%
