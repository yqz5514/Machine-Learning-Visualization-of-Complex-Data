#%%
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
# matplotlib
x = np.
y1 = np.sin(x)
y2 = np.cos(x)

font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':20}

c1 = np.random.rand(1,3)
c2 = np.random.rand(1,3)

plt.plot(x, y1,color=c1, linestyle ='-' )
plt.plot(x, y2,color=c2)
plt.xlabel('time',fontdict=font1)
plt.ylabel('time',fontdict=font2)
plt.grid(axis='x') # only x axis grid
plt.show()
#plt.tight_layout() do what? 
# %%


x = np.linspace(-10,10,1000)
#x = np.array(x, dtype=np.complex)
y1 = x ** 2
y2 = x ** 3
y3 = x ** (1/2)
y4 = x ** (1/3)

fig,ax = plt.subplots(2,2)
ax[0,0].plot(x,y1)
ax[0,1].plot(x,y2)
ax[1,0].plot(x,y3)
#x[1,0].set_title('$f(x)=\sqrt{x}$')
ax[1,1].plot(x,y4)

plt.show()
# RuntimeWarning: invalid value encountered in sqrt
# %%
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
plt.legend(["Round 1", "Round 2", "Round 3", "Round 4"])
plt.title("Scores by Teams in 4 Rounds")
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
# area line

#%%
df = pd.read_excel('Sample - Superstore.xls')
# %%
df1 = df.groupby('Region').sum()
plt.figure()
plt.bar(df1.index, df1['Profit'])
plt.show()
# %%
# cat
# find categorical variables under different country 
# see class notes from today