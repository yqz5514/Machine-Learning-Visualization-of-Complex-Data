#%%
from re import L
from tkinter import Scale
from turtle import left
from xml.etree.ElementInclude import include
import numpy as np # linear algebra
import pandas as pd 
from scipy.stats import gmean
from scipy.stats import hmean
#%%
a = [1,2,3,4,100]
b = [1,2,3,4,0.01]
x = np.array(a)
y = np.array(b)

print(f'AM is : {np.mean(x): .2f}')
print(f'GM is : {gmean(x): .2f}')
print(f'HM is : {hmean(x): .2f}')

#%%
from scipy.stats import skew
def get_skw(i, x):
    m_i = (1/len(x)) * np.sum((x - np.mean(x))**i)
    return m_i

m3 = get_skw(3,a)
m2 = get_skw(2,a)
g1 = m3/m2**(1.5)
g = skew(a)
print(g1)
print(g)
# %%
import pandas_datareader as web      
web.DataReader('AAPL', data_source='yahoo', start = '2000-01-01', end='2022-09-01') 
#%%
mean_x, mean_y, men_z = 0, 5, 10
# can assign multiple variables at same line 
#%%
import numpy as np
import pandas as pd
from tabulate import tabulate
np.random.seed(123)
x1 = np.random.normal(loc = 0, scale = 1, size = 1000)
y2 = np.random.normal(loc = 5, scale = 2, size = 1000)
z = np.random.normal(loc = 10, scale = 3, size = 1000)

#%%
X = np.vstack((x,y,z)).transpose()
X.shape

# %%
df = pd.DataFrame(data = X, columns=['x', 'y', 'z'])
print(df.head())
#%%

print(tabulate(df.corr(), headers='keys', tablefmt='fancy_grid'))
# becasue randomly generazed smaple,they should be independent,  so corr should be close to 0
# %%
from scipy import stats
x = [1,2,3,4,5]
y = [10,20,30,40,50]
#%%
spear_res = stats.spearmanr(x1,y2)
spear_res
# %%
pearson_res = stats.pearsonr(x1,y1)
pearson_res
# (corr_coef, p_value)
# %%
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/weight-height.csv'
df1 = pd.read_csv(url)
df1.head()
#%%
df1.columns
# %%
# convert gender into numerical
# replace
#.apply
#map

#%%
#map
gender_mapping = {
           'Male': 1,
           'Female': 2}

df1['Gender_num'] = df1['Gender'].map(gender_mapping)
df1

#%%
# apply method
def change(i):
    if i == 'Male':
        return 1
    elif i == 'Female':
        return 0
df1['Gender_num'] = df1['Gender'].apply(change)



# %%
heigh = df1['Height'].values
weight = df1['Weight'].values
gender = df1['Gender_num'].values

# this function will be in toolbox from professor 
r_height_weight = correlation_coefficent_cal(height, weight)

# remove gender 
# particial correlation between height and weight
r_height_weight_dot_gender = (r_height_weight - r_height_gender*r_gender_weight)/(np.sqrt(1-r_gender_weight**2)*np.sqrt(1-r_height_gender)**2)
#0.857
#dof = n-3
n = len(df1)
k = 1
t = r_height_weight_dot_gender*np.sqrt((n-2-k)/(1-r_height_weight_dot_gender**2))

#%%
# pretty table 
from prettytable import PrettyTable

x = PrettyTable()
name = ['A', 'B', 'C', 'D']
x.field_names = name 

df = pd.DataFrame(data = np.random.rand(6,4), columns = name)

for i in range(6):
    x.add_row(df.iloc[i,:])
    
print(x.get_string(title = 'Random Number'))
#%%
#############################################pandas############################################
# pandas
import numpy as np
import pandas as pd
import seaborn as sns 
name = sns.get_dataset_names()
#print(name)

df = sns.load_dataset('diamonds') #more than 55k data
#print(df[10:15])
#df.head()
#df.tail(), cant put negative inside


# %%
df.describe()
# if want to show all columns in  add .to_string()
# 
# %%
# only numeric features
df2 = df.select_dtypes(include = np.number)
df2
# %%
# df.columns
df['cut'].unique() # unique only use for sepecific column
# %%
df.values
# convert to n-d array
# %%
df.columns
# %%
# create / select columns in easy way
df3 = df[['cut', 'color','clarity']]
df3
# %%
# more specific selecting
# df.iloc() select by number
# df.loc() select by number / column name
df.iloc[10:15,-4:]
# %%
df.clarity.unique()
# %%
# conditional selecting
#df[df['cut']== 'Premium'|'Very Good']
df5 = df.loc[(df['carat'] >= 0.75) &( (df['cut'] == 'Premium')|(df['cut'] == 'Very Good')) & ((df['color']=='F') | (df['cut'] == 'G')) & ((df['clarity'] == 'VVS1')|(df['clarity']=='VVS2')|(df['clarity']=='S1'))]
df5.head()
# second methosd of conditional selecting : .isin()
#df6 = df[df['cut'].isin(['Premium','Very Good'])]

#%%
import matplotlib.pyplot as plt
plt.figure()
df5.price.hist()
plt.show()
# %%
dff = sns.load_dataset('tips')
dff
# %%
dff['new_bill'] = dff['total_bill'] - 2
dff
# %%
# use inplace = True or axis wont execuate
dff.drop(['new_bill'], axis = 1, inplace = True)
#dff.drop([['new_bills','sex'], axis = 1, inplace = True])
#%%
dff['bucket'] = np.where(dff['total_bill'] <= 10, 'low', 'high')
#dff['bucket'] = df['total_bill'].map(lambda x: 'high' if x>=10 else 'low')
dff['bb'] = dff['total_bill'].map(lambda x : )
#%%
# three cases
#np.where solution:
# np.where(dff['total_bill']<=10, 'low',
# (np.where(dff['total_bill'] <= 15, normal, high)))
# lambda method:
# dff['bucket'] = df['total_bill'].map(lambda x: 'low' if x<=10 else ('normal' if x>10 and x<=15 else 'high') )

def change(x):
    if i <= 10:
        return 'low'
    elif i > 15:
        return 'high'
    else:
        return 'normal'
dff['bucket1'] = dff['total_bill'].apply(change)

#%%
dff.sort_values(['sex','total_bill'], inplace = True)
# how to change the asecding, deascding 
#%%
# count_value
dff.value_counts()

#%%
# insert
#DataFrame.insert(loc, column, value, allow_duplicates=_NoDefault.no_default)
dff.insert(7,'ID',1)

#%%
#DataFrame.drop(labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise')
dff1 = dff.drop(index = 11)
# drop by index
index = dff[dff['sex'] == 'Male'].idex
dff3 = dff.drop(index)
#%%
dff2 = dff[dff['sex'] == 'Female']
dff2
# %%
# url = nba.csv at Github
# drop na
#df.dropna(how ='any', inplace = True) all missing valuw will be droped
# imputation
#fillna
df.college.fillna(df['college'].mode()[0]#[0]make sure only return 1 number or will return a tuple becasue mode can be more than 1
                  , inplace = True)
#df.bfill(axis ='rows') df.ffill(axis = 0)
#df.bfill(axis ='columns') df.ffill(axis = 1)
#%%
# covariance matrix
from numpy import linalg as LA 
import numpy as np
from prettytable import PrettyTable
import pandas as pd
from tabulate import tabulate
#%%
A = np.array([[3,2],[2,4]])
a,b = LA.eig(A)
b = pd.DataFrame(b, columns=['v1','v2'])
print(tabulate(b))
print(f'eigen value is{a}')
# %%
import matplotlib.pyplot as plt
#%%
np.random.seed(123)
x = np.random.randn(2, 10000)
#y = np.random.normal(loc = m_y, scale = np.sqrt(v_y), size = n_y)

#%%
plt.figure(figsize=(12,8))
#plt.subplot(1,2,1)

plt.scatter(x[0,:],x[1,:],color='blue',label='raw-data')
plt.title('raw-data')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.axis('equal')
plt.tight_layout()
plt.show()

#%%

y = np.matmul(A,x)
plt.subplot(1,2,2)
plt.figure()
plt.scatter(y[0,:],y[1,:],color='blue',label='transformed-data')
plt.title('transformed-data')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.axis('equal')
plt.tight_layout()
plt.show()
# %%
#b = b.values
plt.figure(figsize=(12,8))
plt.scatter(y[0,:],y[1,:],color='blue',label='transformed-data')
origin = np.array([[0,0],[0,0]])
plt.quiver(*origin,a[0]*b[:,0],a[0]*b[:,1], color = ['y','c'],scale=12)
plt.title('transformed-data')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.axis('equal')
plt.tight_layout()
plt.show()
# %%
from scipy.optimize import curve_fit

#%%
# regression analysis
# find the best line, mimium MES 
def pbjective(x,a,b):
    return a+x+b

x_f,y_f = y[0,:],y[1,:]
popt,_=curve_fit(pbjective,x_f,y_f)
slope,intercept = popt

x_p = np.linspace(-10,10,1000)
y_p = slope +x_p + intercept


print(f'intercept of the fitted line is {intercept:.3f}')
print(f'slope of the fitted line is {slope:.3f}')

# linear transformation 
# squent function tanh
# review!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# lambda....
# find a job .....

# %%
# exam 1
# dataframe table
# conditional selection
# be able to use three way to do the subplot
# cheat sheet with formula
#################################################10/05##############################################

#%%
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/mnist_test.csv'
df = pd.read_csv(url) #low_memory=False) 
#print(df.head(5))
#%%
#print(df.info)
#%%

df0 = df[df['label']==0]
print(df0.shape)

#%%
for i in range(10):
    df0 = df[df['label']==i]
    print(f'number of observations corresponding to {i} is {df0.shape[0]}')
# %%
plt.figure(figsize=(12,12))
for i in range(10):
    df0 = df[df['label']==i]
    pic = df0[0:1].values.reshape(785)[1:].reshape(28,28)
    plt.subplot(4,3,i+1)
    plt.imshow(pic)

plt.tight_layout()
plt.show()
    
#%%
plt.subplot(3, 4,i+1)

#%%
############pie chart#####################
data = [23,17,32,29,12]
label = ['C','C++','Java','Python','GO']
explode = (0,0,0,0,0)
plt.figure()
plt.pie(data,labels = label, explode=explode)
plt.show()

#%%
data = [23,17,32,29,12]
label = ['C','C++','Java','Python','GO']
explode = (0.03,0.03,0.2,0.03,0.03)
plt.figure()
plt.pie(data,labels = label, explode=explode,autopct='%1.2f%%')#display percentage
plt.legend(loc= (.85,.85))
plt.axis('square') # make sure the plot will be in circle shape
plt.show()
# %%
# bar plot
score = [23,17,32,29,12]
label = ['C','C++','Java','Python','GO']
#explode = (0,0,0,0,0)
plt.figure()
plt.bar(label, score)
plt.ylabel('score')
plt.xlabel('languages')
plt.title('simple bar plot')
plt.show()
# %%
# stack bar plot
score_men = [23,17,32,29,12]
score_men_norm = (score_men)/np.sqrt(np.sum(np.sqrt(score_men)))# L2 normalized
score_women = [35,25,10,15,18]
label = ['C','C++','Java','Python','GO']
#explode = (0,0,0,0,0)
plt.figure()
plt.bar(label, score_men, label = 'men')
plt.bar(label, score_women, bottom = score_men, label='women')

plt.ylabel('score')
plt.xlabel('languages')
plt.title('simple stack bar plot')
plt.show()
# %%
# group barplot
width = 0.4
score_men = [23,17,32,29,12]
#score_men_norm = (score_men)/np.sqrt(np.sum(np.sqrt(score_men)))# L2 normalized
score_women = [35,25,10,15,18]
label = ['C','C++','Java','Python','GO']
x = np.arange(len(label))
fig,ax = plt.subplots()
ax.bar(x - width/2, score_men, width, label='men')
ax.bar(x + width/2, score_women, width, label='women')
ax.set_ylabel('scores')
ax.set_xlabel('languages')
ax.set_xticks(x)
ax.set_xticklabels(label)
ax.legend()
plt.show()
# %%
######group bar plot
# create data
x = ['A', 'B', 'C', 'D']
y1 = np.array([10, 20, 10, 30])
y2 = np.array([20, 25, 15, 25])
y3 = np.array([12, 15, 19, 6])
y4 = np.array([10, 29, 13, 19])
 
# plot bars in stack manner
plt.bar(x, y1, color='r')
plt.bar(x, y2, bottom=y1, color='b')
plt.bar(x, y3, bottom=y1+y2, color='y')
plt.bar(x, y4, bottom=y1+y2+y3, color='g')
plt.xlabel("Teams")
plt.ylabel("Score")
plt.legend(["Round 1", "Round 2", "Round 3", "Round 4"], loc='upper/lower/center right/left')
plt.title("Scores by Teams in 4 Rounds", loc='left')
plt.show()

# %%
# barh()
# plot it in horizontally
score_men = [23,17,32,29,12]
#score_men_norm = (score_men)/np.sqrt(np.sum(np.sqrt(score_men)))# L2 normalized
score_women = [35,25,10,15,18]
label = ['C','C++','Java','Python','GO']
fig = plt.figure()
plt.barh(label, score_men, label='men')
plt.barh(label, score_women, left=score_men, label='men')
plt.ylabel('languages')
plt.xlabel('scores')
plt.title('simple stack bar plot')
plt.legend(loc = 1)
plt.show()
# %%
x = ['A', 'B', 'C', 'D']
y1 = np.array([10, 20, 10, 30])
y2 = np.array([20, 25, 15, 25])
y3 = np.array([12, 15, 19, 6])
y4 = np.array([10, 29, 13, 19])
 
# plot bars in stack manner
plt.barh(x, y1, color='r')
plt.barh(x, y2, left=y1, color='b')
plt.barh(x, y3, left=y1+y2, color='y')
plt.barh(x, y4, left=y1+y2+y3, color='g')
plt.ylabel("Teams")
plt.xlabel("Score")
plt.legend(["Round 1", "Round 2", "Round 3", "Round 4"])
plt.title("Scores by Teams in 4 Rounds")
plt.show()
#%%
width = 0.4
score_men = [23,17,32,29,12]
#score_men_norm = (score_men)/np.sqrt(np.sum(np.sqrt(score_men)))# L2 normalized
score_women = [35,25,10,15,18]
label = ['C','C++','Java','Python','GO']
x = np.arange(len(label))
fig,ax = plt.subplots()
ax.barh(x - width/2, score_men, width, label='men')
ax.barh(x + width/2, score_women, width, label='women')
ax.set_xlabel('scores')
ax.set_ylabel('languages')
ax.set_yticks(x)
ax.set_yticklabels(label)
ax.legend()
plt.show()
# %%
# boxplot
# simple box-plot
plt.figure()
plt.boxplot(data)
plt.xticks([1])
plt.ylabel('Average')
plt.xlabel('x')
plt.grid()
plt.show()

#multivariable
data=[data1, data2,...]
plt.figure()
plt.boxplot(data)
plt.xticks([1,2,3,4])
plt.ylabel('y')
plt.xlabel('x')
plt.title('')
plt.grid()
plt.show()

#%%
# simple area plot
x = range(1,20)
y = np.random.normal(10,2,len(x))
plt.fill_between(x,y, alpha = 0.3,label = 'area between x and y')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left')
plt.title('t')
plt.grid()
plt.show()
#%%
#Multivariate Line plot
# to set the plot size
plt.figure(figsize=(16, 8), dpi=150)
  
# using plot method to plot open prices.
# in plot method we set the label and color of the curve.
tesla['Open'].plot(label='Tesla', color='orange')
gm['Open'].plot(label='GM')
ford['Open'].plot(label='Ford')
  
# adding title to the plot
plt.title('Open Price Plot')
  
# adding Label to the x-axis
plt.xlabel('Years')
  
# adding legend to the curve
plt.legend()
#%%
#stem plot
x = np.linspace(0.1,2*np.pi,41)
y = np.exp(np.sin(x))
(markers, stemlines, baseline) = plt.stem(y, label = 'sin(x)')
plt.setp(markers, color='red',marker='o')
plt.setp(baseline, color='grey', linewidth=2, linestyle='-')
plt.title('')
plt.xlabel()
plt.ylabel()
plt.grid()
plt.show()
#%%%
import pandas as pd
import matplotlib.pyplot as plt
#%%
df = pd.read_excel('Sample - Superstore.xls')
#%%
df.info()
#%%
df.value_counts()
#%%
df.head()
#%%
df.columns
#%%
df['Product Name'].value_counts()
# %%
df1 = df.groupby('Region').sum()
plt.figure()
plt.bar(df1.index, df1['Profit'])
plt.show()
#%%
df3 = df.groupby(by = ['Product Name','Region']).count()
df3.head(10)
#%%
df11 = df[['Region','Product Name']]
df4 = df11[df11['Region'] =='Central'].groupby(['Product Name']).count()
df4.head(4)
#%%
df5=df11.groupby('Region')['Product Name'].apply(lambda x: (x=='Staples').sum()).reset_index(name='count')
df5.head()
#%%
df5.columns
#%%
df4.columns
#%%
plt.figure(figsize=(20,20))
plt.bar(df5['Region'], df5['count'])
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
#%%

df2 = df['Product Name'].groupby(['Region']).count()
#plt.figure()
#plt.bar(df2., df1['Profit'])
#plt.show()
#%%
df2.head()
# %%
# cat
# find categorical variables under different country 
# see class notes from today
income_gender = pd.DataFrame(world1['income00'].groupby(world1['gender']).mean())

#%%
# plt.subplot(row,col,curent ax)


# x = np.linspace(0, 2*np.pi, 1000)
# y1 = np.sin(x)
# y2 = np.cos(x)


# font1 = {'family':'serif','color':'blue', 'size':20}
# font2 = {'family':'serif','color':'darkred', 'size':15}
#
#
# c1 = np.random.rand(1,3)
# c2 = np.random.rand(1,3)
#
# plt.figure(figsize=(12,8))
# plt.subplot(2,1,1)
# plt.plot(x, y1, label = r'$\sin(x)$', lw = 4, color = c1, linestyle = ':')
# plt.plot(x, y2, label = r'$\cos(x)$', lw = 2.5, color = c2, linestyle = '-')
# plt.legend(loc = 'lower left')
# plt.title('sine and cosine graph', fontdict=font1)
# plt.xlabel('time', fontdict=font2)
# plt.ylabel('Mag', fontdict=font2)
# plt.grid(axis='x')
#
# plt.subplot(2,1,2)
# plt.plot(x, y1+y2, label = r'$\sin(x)+\cos(x)$', lw = 4, color = c1, linestyle = '-.')
# # plt.plot(x, y2, label = r'$\cos(x)$', lw = 2.5, color = c2, linestyle = '-')
# plt.legend(loc = 'lower left')
# plt.title(r'$\sin(x)+\cos(x)$', fontdict=font1)
# plt.xlabel('time', fontdict=font2)
# plt.ylabel('Mag', fontdict=font2)
# plt.grid(axis='x')
# plt.tight_layout()
#
# plt.show()

#%%
# plt.add_subplot
x = np.linspace(1,10)
y = [10 ** el for el in x]
z = [2 ** el for el in x]
fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(2,2,1)
ax1.plot(x,y, color = 'blue')
ax1.set_yscale('log')
ax1.set_title('Logarithmic plot of $10^{x}$')
ax1.set_ylabel('$y = 10^{x} $')
plt.grid(visible=True, which='both')
ax2 = fig.add_subplot(2,2,2)
ax2.plot(x,y, color = 'blue')
ax2.set_yscale('linear')
ax2.set_title('linear plot of $10^{x}$')
ax2.set_ylabel('$y = 10^{x} $')
plt.grid(visible=True, which='both')
ax3 = fig.add_subplot(2,2,3)
ax3.plot(x,z, color = 'green')
ax3.set_yscale('linear')
ax3.set_title('Logarithmic plot of $10^{x}$')
ax3.set_ylabel('$y = 2^{x} $')
plt.grid(visible=True, which='both')
ax4 = fig.add_subplot(2,2,4)
ax4.plot(x,z, color = 'magenta')
ax4.set_yscale('linear')
ax4.set_title('Linear plot of $2^{x}$')
ax4.set_ylabel('$y = 2^{x} $')
plt.grid(visible=True, which='both')
plt.tight_layout()
plt.show()

#%%
# hexbin plot is usefl to represent the relationship of two numerical variables whn you have many data points
#plt.hexbin(x,y,gridsize=(50,50))


################################################10####19####notes########################################################################################
#%%
##################seaborn###################

##final term project
######transformation dropdown meun###############
#normalization: x = (xi-minx)/(maxx-minx)    , x will be in [0,1]
#sandarlization: x = (x-meanx)/x.std     , mean =0 & variance(x)=1
#iqr: (perform bettern when have outlier), x = (x-x.median)/(3rdQ-4thQ)
#differencing: first differencing# is first derivitive actually, second differencing #not require in project

###interact dashboard
###look at other's app get insight. how to design the app
#movie review
#%%
##################seaborn###################
import seaborn as sns 

flight = sns.load_dataset('flights')

#%%
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
#%%
flight.head
# %%
sns.lineplot(data = flight,
             y = 'passengers',
             x = 'year',
             hue = 'month')
plt.show()
# %%
sns.boxplot(data = flight.passengers)
plt.show

# %%
tip = sns.load_dataset('tips')
diamonds = sns.load_dataset('diamonds')
penguins = sns.load_dataset('penguins')
iris = sns.load_dataset('iris')

#%%
sns.countplot(data = tip,
              x = 'sex')
plt.show
# %%
sns.countplot(data = tip,
              x = 'smoker',
              hue = 'sex')
plt.show

# %%
sns.countplot(data = tip,
              x = 'day',
              hue = 'sex')
plt.show
# %%
sns.countplot(data = diamonds,
              x = 'clarity', # change x to y make it horizental
              order = diamonds['clarity'].value_counts().index)
plt.show
# %%
sns.countplot(data = diamonds,
              x = 'clarity', # change x to y make it horizental
              order = diamonds['clarity'].value_counts().index[::-1])#desacding order
plt.show
# %%
sns.countplot(data = diamonds,
              x = 'clarity', # change x to y make it horizental
              order = reversed(diamonds['clarity'].value_counts().index[::-1]))#desacding order
plt.show
# %%
sns.barplot(data='diamonds',
            x = 'color')
plt.show()
# %%
#s = 'adigoiijjklkj'
s =[1,2,3,4]
s[::-1]
# %%
color_group = diamonds.groupby('color').count()
#%%
color_group
# %%
sns.barplot(data = color_group,
             x = color_group.index,
             y = 'cut')
plt.show
# %%
#scatter plot
sns.relplot(data=tip,
            x = 'total_bill',
            y = 'tip',
            hue = 'day')
plt.show()
# x and y axis can not swap bcs independent and dependent variable is not interchangeable
# %%
sns.relplot(data=flight,
            x = 'year',
            y = 'passengers',
            kind = 'line',
            hue = 'month')
plt.show()
# %%
sns.relplot(data=tip,
            x = 'total_bill',
            y = 'tip',
            kind = 'scatter',
            hue = 'day',
            col = 'time')
plt.show()
# %%
sns.relplot(data=tip,
            x = 'total_bill',
            y = 'tip',
            kind = 'scatter',
            hue = 'day',
            col = 'time',
            row = 'sex')
plt.show()
# %%
flights = flight.pivot('month','year','passengers')
#%%
ax = sns.heatmap(flights, cmap='YlGnBu', linewidths=0.5 )# annot= True will add value to each cube
plt.title('Heat Map')

# %%
sns.pairplot(data=penguins,
             hue = 'species')
plt.show()
# %%
sns.kdeplot(data=tip, # show the estimate of density function 
             x = 'total_bill',
             bw_adjust=0.2,#the higher this filter is the smmother the line will be
             cut = 0)
plt.show()
# %%
sns.kdeplot(data=tip,
             x = 'total_bill',
             hue = 'time',
             multiple='stack'#fill the area under curve
             )
plt.show()
# %%
sns.kdeplot(data=diamonds,
             x = 'price',
             hue='cut',
             log_scale= True,
             fill=True,
             alpha = .5,#transparency,
             palette='crest'
             )
plt.show()
# %%
#contour plot
sns.kdeplot(data=tip,
             x = 'total_bill',
             y = 'tip',
             #fill=True
             
             )
plt.show()
# %%
sns.lmplot(data=tip,
             x = 'total_bill',
             y = 'tip',)
plt.show()
# %%
sns.displot(data=tip,
             x = 'total_bill',
             kde = True,
             )
plt.show()
# %%
###################################1026######################################
## dashboard
# plot in  browser 
# google cloud setup
# search google cloud console create project 
# plotly
# based on dash 

# %%
import plotly.express as px
import numpy as np
import pandas as pd
# %%
x = np.linspace(-8,8,100)
y = x ** 2
z = x ** 3
h = np.vstack((x,y,z))
#%%
print(h)
# %%
df = pd.DataFrame(data = h.T, columns=['x','$x^{2}$','$x^{3}$']) # T is transpose
# %%
df
# %%
fig = px.line(df,
              x = 'x',
              y = ['$x^{2}$','$x^{3}$'],
              width = 800, height = 400,
              title = r'$\text{Graph of}  x^2 \&  x^3$',
              labels = {'value': 'USD($)',
                        'x':'first variable'}
              )
fig.show(renderer = 'browser')
# %%
fig.update_layout(
    title_font_size = 20,
    title_font_color = 'red',
    title_font_family = 'Times New Roman',
    legrnd_title_font_size = 20,
    legend_title_font_color = 'green',
    font_color = 'blue',
    font_family = 'Courier New'
)
#%%
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
aapl.drop('Volume', axis = 1, inplace = True)
# %%
fig = px.line(aapl,
              #x = aapl.index,
              #y = ['High','Low'],
              y = aapl.columns,
              width = 800, height = 800,
              title = 'AAPL Stock Price',
              labels = {'value': 'USD($)'}
              )
fig.show(renderer = 'browser')

# %%
iris = px.data.iris()#good classfifcation dataset
tip = px.data.tips()
gapminder = px.data.gapminder()

#%%
fig = px.line(iris,
              x = 'species',
              y = ['petal_length', 
                   'petal_width',
                   'sepal_length',
                   'sepal_width'],
              title = '',
              width = 800, height = 400,
              labels = {'value': 'dimension(cm)'},
              template = 'plotly_white'
              
              )

#%%
fig = px.scatter(gapminder,
                 x = 'gdpPercap',
                 y = 'lifeExp',
                 size = 'pop',
                 color = 'continent',
                 size_max = 55,
                 animation_frame = 'year',
                 animation_group = 'country',
                 range_x = [0, 6000],
                 range_y = [25, 90],
                 title = 'GDP'
                 )
fig.show()
#%%
fig = px.choropleth(gapminder,
                 location = 'iso_alpha',
                 projection = 'natural earth',
                 hover_name = 'country',
                 color = 'lifeExp',
                 animation_frame = 'year',
                 #color_continuous_scale = px.colors.sequential.Plasma,# show the color for each country
                 #animation_group = 'country',
                 #range_x = [0, 6000],
                 #range_y = [25, 90],
                 title = 'GDP'
                 )
fig.show()

#%%
# horizontal bar plot
fig = px.bar(tip,
             x = 'total_bill',
             y = 'day',
             color = 'sex',
             barmode = 'stack'
             )
fig.show() 

#%%
# diamond 
import seaborn as sns 
dia = sns.load_dataset('diamond')
# %%
dia1 = dia.groupby('color').sum()
dia1 = dia1.reser_index()
fig = px.bar(dia1,
             x = 'color',
             y = 'price'
             )

fig.show()

#%%
fig = px.bar(tip,
             x = 'total_bill',
             y = 'day',
             color = 'sex',
             barmode = 'group'
             )
fig.show() 

#%%
# experess is easiler
# graph and objects is more complex.
nbins = 50

#%%
import plotly.graph_objects as go
# %%
fig = go.Figure(data = [go.Histogram(x=iris['sepal_width'], # change x to y, will turn it to horizontal
                nbinsx = 40)]
    
)
#%%
fig = go.Figure()
fig.add_trace(go.Histogram(y=iris['sepal_width'],nbinsx=40))
fig.add_trace(go.Histogram(x = iris['sepal_length'], nbinsx=40))
fig.update_layout(barmode = 'stack')

#%%
# subplot only in graph objects
from plotly.subplots import make_subplots

fig = make_subplots(rows =1, cols =2)

fig.add_trace(
    go.Scatter(x = [1,2,3], y = [4,5,6]),
               row = 1, col = 1)


fig.add_trace(
    go.Scatter(x = [1,2,3], y = [4,5,6]),
               row = 1, col = 2)

fig.update_layout(width = 800, height = 600,
                 title = 'side by side fig')

#fig['layout']['xaxis']['title'] = 'label x - axis 1'
#fig['layout']['xaxis2']['title'] = 'label x - axis 1'
#fig['layout']['xaxis']['title'] = 'label x - axis 1'

# %%
#######################################11/02################################################
# %%
# # dash 
import dash as dash
from dash import dcc 
from dash import html

# %%
# app.callback() connect to th input and output 
# phase two
my_app = dash.Dash('My App')

#phase three
# Divioison is a section of app
my_app.layout=html.Div([html.H1('Hellow World! with html.H1',style={'textAlign':'right'}),
                       html.H2('Hellow World! with html.H2',style={'textAlign':'right'}),
                       html.H3('Hellow World! with html.H3',style={'textAlign':'right'}),
                       html.H4('Hellow World! with html.H4',style={'textAlign':'right'}),
                       html.H5('Hellow World! with html.H5',style={'textAlign':'right'}),
                       html.H6('Hellow World! with html.H6',style={'textAlign':'right'}),
                       
])

my_app.run_server(
    port = 8002,
    host = '0.0.0.0'
)
# %%
my_app = dash.Dash('My App')

#phase three
# Divioison is a section of app
my_app.layout=html.Div([html.H1('Assignment 1'),
                       html.Button('submit assignment 1', id='a1'),
                       
                       html.H1('Assignment 2'),
                       html.Button('submit assignment 2', id='a2'),
                       
                       html.H1('Assignment 2'),
                       html.Button('submit assignment 2', id='a3'),
                       
                       html.H1('Assignment 2'),
                       html.Button('submit assignment 2', id='a4'),
                       
                       
])

my_app.run_server(
    port = 8003,
    host = '0.0.0.0'
)
# %%
my_app = dash.Dash('My App')
my_app.layout = html.Div([html.H1('Complex Data Vis'),
                          dcc.Dropdown(id='my_drop',options=[
                              {'label': 'Introduction','value':'introduction'},
                              {'label': 'Pandas','value':'Pandas'},
                              {'label': 'seaborn','value':'seaborn'},
                              {'label': 'plotly','value':'plotly'}'
                              ])
                    ]
                    )
my_app.run_server(
    port = 8004,
    host = '0.0.0.0'
)


#%%
#multiple select9 multiple features to be select
my_app = dash.Dash('My App')
my_app.layout = html.Div([html.H1('Complex Data Vis'),
                          dcc.Dropdown(id='my_drop',options=[
                              {'label': 'Introduction','value':'introduction'},
                              {'label': 'Pandas','value':'Pandas'},
                              {'label': 'seaborn','value':'seaborn'},
                              {'label': 'plotly','value':'plotly'}'
                              ],multi=True) # how tolimite the bumber of multi selection
                    ])
my_app.run_server(
    port = 8004,
    host = '0.0.0.0'
)

#%%
#phase4 callback
from dash.dependencies import Input, Output
#multiple select9 multiple features to be select
my_app = dash.Dash('My App')
my_app.layout = html.Div([html.H1('Complex Data Vis'),
                          dcc.Dropdown(id='my_drop',options=[
                              {'label': 'Introduction','value':'introduction'},
                              {'label': 'Pandas','value':'Pandas'},
                              {'label': 'seaborn','value':'seaborn'},
                              {'label': 'plotly','value':'plotly'},
                              ],multi=True), # how tolimite the bumber of multi selection
                          
                          html.Br(),
                          html.Br(),# line of break
                          html.Div(id='my_out')
                    
                    
                    ])
                    
@my_app.callback(# whnerver have a callbakck need follow by a update function or will return error
    Output(component_is='my_out',component_property='children'),
    [Input(compnent_id='my_drop', component_property='value')]
)

def update_Reza(input):
    return f'The select item is {input}'


my_app.run_server(
    port = 8004,
    host = '0.0.0.0'
)

#%%
# slider
my_app = dash.Dash('My App')
my_app.layout = html.Div([
                          dcc.Slider(id='my_input',
                          min = 10,
                          max = 90,
                          step = 5,
                          value = 70
                              ), # how tolimite the bumber of multi selection
                          
                          html.Div(id='my-out')
                    
                    
                    ])
                    
@my_app.callback(# whnerver have a callbakck need follow by a update function or will return error
    Output(component_is='my-out',component_property='children'),
    [Input(compnent_id='my_input', component_property='value')] 
)

def update_Reza(Input):
    return f'The select item is {(Intput-32)/1.8} celese'


my_app.run_server(
    port = 8004,
    host = '0.0.0.0'
)

#%%
# out put as slider
my_app = dash.Dash('My App')
my_app.layout = html.Div([
                          dcc.Slider(id='my-input',
                          min = 10,
                          max = 90,
                          step = 5,
                          value = 70
                              ), # how tolimite the bumber of multi selection
                          
                          html.Div(id='my-out')
                    
                    
                    ])
                    
@my_app.callback(# whnerver have a callbakck need follow by a update function or will return error
    Output(component_is='my-out',component_property='children'),
    [Input(compnent_id='my_input', component_property='value')] 
)

def update_Reza(Input):
    return f'The select item is {(Input-32)/1.8:.2f} celese'


my_app.run_server(
    port = 8004,
    host = '0.0.0.0'
    
    
)

#%%
# tabs
my_app = dash.Dash('My App')
my_app.layout=html.Div({
    html.H1('HW9', style={'textAlign':'Center'}),
    html.Br(),
    dcc.Tabs(id='hw-questions',
             children=[
                 dcc.Tab(label='Question 1', value='q1'),
                 dcc.Tab(label='Question 2', value='q2')
             ]),
    html.Div(id='layout')
    
    
    
})

queation1_layout =html.Div([
    html.H1(),
    html.H5(),
    html.P('Input')
])

