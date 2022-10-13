#%%
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

#%%
# Q1
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/mnist_test.csv'
df = pd.read_csv(url, low_memory=False) 
#%%
#a. Plot the digital numbers [0-9] for the first 100 observations in one figure with 10 rows and 10 columns.
#Figure size = (12,12)
for i in range(10):
    df0 = df[df['label']==i]
    print(f'number of observations corresponding to {i} is {df0.shape[0]}')
    #%%
df0[1:2]
#%%
df.head
#%%
df[df['label']==1 or 2 or 3]
#%%
df_new = df[(df['label']== 1) & (df['label']== 2)&(df['label']== 3)&(df['label']== 4)&(df['label']== 5)&(df['label']== 6)&(df['label']== 7)&(df['label']== 8)&(df['label']== 9)]
#%%
df_new.head()
#%%
for i in range(10):
    df0 = df[df['label']==i]
    
    #%%
    df0.label.unique()
#%%
for j in range(100):
        #df0 = df[df['label']==i]
        pic = df_new.loc[j:j].values.reshape(785)[1:].reshape(28,28)
    
        plt.subplot(10,10,j+1)
        plt.imshow(pic)

plt.tight_layout()
plt.show() 
#%%
plt.figure(figsize=(12,12))
for i in range(10):
    df0 = df[df['label']==i]
    for j in range(100):
        #df0 = df[df['label']==i]
        pic = df0.loc[j:j].values.reshape(785)[1:].reshape(28,28)
    
    plt.subplot(10,10,j+1)
    plt.imshow(pic)

plt.tight_layout()
plt.show() 
# %%

#%%
#Q 2
import numpy as np
import pandas as pd
import seaborn as sns 
plt.style.use('seaborn-whitegrid')

#from seaborn import seaborn-whitegrid
name = sns.get_dataset_names()
#print(name)

df = sns.load_dataset('diamonds')
#%%
#Check if the dataset contains missing values or ‘nan’s. 
# If it does, then remove [no need for replacement] all the missing observations. All the questions bellow must be answered using the ‘clean’ dataset. 
# Set the matplotlib style as ‘seaborn-whitegrid. [7.5pts
df.isnull().sum()

#%%
df.cut.unique()
# %%
from prettytable import PrettyTable

x = PrettyTable()
x.field_names = ['Index','Cut Name']
    
x.add_row([1, 'Ideal'])
x.add_row([2, 'Premium'])
x.add_row([3, 'Good'])
x.add_row([4, 'Very Good'])
x.add_row([5, 'Fair'])
        #x.title('Corr table of '+ com)
print(x.get_string(title = 'Diamond Dataset – Various cuts'))

# %%
df.color.unique()
# %%
x = PrettyTable()
x.field_names = ['Index','Color Name']
    
x.add_row([1, 'E'])
x.add_row([2, 'I'])
x.add_row([3, 'J'])
x.add_row([4, 'H'])
x.add_row([5, 'F'])
x.add_row([6, 'G'])
x.add_row([7, 'D'])

        #x.title('Corr table of '+ com)
print(x.get_string(title = 'Diamond Dataset – Various colors'))
# %%
df.clarity.unique()
#%%
x = PrettyTable()
x.field_names = ['Index','Clarity Name']
    
x.add_row([1, 'IF'])
x.add_row([2, 'VVS1'])
x.add_row([3, 'VVS2'])
x.add_row([4, 'VS1'])
x.add_row([5, 'VS2'])
x.add_row([6, 'SI1'])
x.add_row([7, 'SI2'])
x.add_row([8, 'I1'])


        #x.title('Corr table of '+ com)
print(x.get_string(title = 'Diamond Dataset – Various clarities'))
#%%
df.head()
#%%
df_cut =pd.DataFrame(df['cut'].value_counts())
#%%
df_cut.columns
#%%
df_cut.head()
# %%
#df1 = df.groupby('cut').sum()
plt.figure()
plt.barh(df_cut.index, df_cut['cut'])
plt.ylabel('Cut Type')
plt.xlabel('Count of Sales')
plt.title('Sales count per cut')
plt.show()
# %%
print(f'The diamond with Ideal cut has the maximum sales per count.')
print(f'The diamond with Fair cut has the minimum sales per count.')
# %%
df_co =pd.DataFrame(df['color'].value_counts())
#%%
df_co.head()
# %%
plt.figure()
plt.barh(df_co.index, df_co['color'])
plt.ylabel('Color Type')
plt.xlabel('Count of Sales')
plt.title('Sales count per color')
plt.show()
# %%
print(f'The diamond with G color has the maximum sales per count.')
print(f'The diamond with J color has the minimum sales per count.')
# %%
df_cl =pd.DataFrame(df['clarity'].value_counts())

# %%
df_cl.head()
# %%
plt.figure()
plt.barh(df_cl.index, df_cl['clarity'])
plt.ylabel('Clarity Type')
plt.xlabel('Count of Sales')
plt.title('Sales count per clarity')
plt.show()
# %%
print(f'The diamond with SI1 has the maximum sales per count.')
print(f'The diamond with I1 has the minimum sales per count.')
#%%
df.cut.unique()
# %%
lb = ['Ideal', 'Premium', 'Good', 'Very Good', 'Fair']
explode = (0.03,0.03,0.03,0.03,0.03)
plt.figure()
plt.pie(df_cut['cut'],labels = df_cut.index, explode=explode,autopct='%1.2f%%')#display percentage
plt.legend(loc= (.85,.85))
plt.axis('square') # make sure the plot will be in circle shape
plt.title('Sales count per cut in %', loc='center')

plt.show()
# %%
print(f'The diamond with Ideal cut has the maximum sales per count with 39.95% sales count')
print(f'The diamond with Fair cut has the minimum sales per count with 2.98% sales count')
# %%
explode = (0.03,0.03,0.03,0.03,0.03,0.03,0.03)
plt.figure()
plt.pie(df_co['color'],labels = df_co.index, explode=explode,autopct='%1.2f%%')#display percentage
plt.legend(loc= (.85,.85))
plt.axis('square') # make sure the plot will be in circle shape
plt.title('Sales count per color in %', loc='center')
#%%
print(f'The diamond with G color has the maximum sales per count with 20.93% sales count')
print(f'The diamond with J color has the minimum sales per count with 5.21% sales count')
# %%
explode = (0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03)
plt.figure()
plt.pie(df_cl['clarity'],labels = df_cl.index, explode=explode,autopct='%1.2f%%')#display percentage
plt.legend(loc= (.85,.85))
plt.axis('square') # make sure the plot will be in circle shape
plt.title('Sales count per clarity in %', loc='center')
# %%
print(f'The diamond with SI1 has the maximum sales per count with 24.22% sales count')
print(f'The diamond with I1 has the minimum sales per count with 1.37% sales count')
# %%
df1 = df[df['clarity'] == 'VS1'].groupby(['cut','color']).count()
df1.head()
# %%
df_se = df[['cut','color','clarity','price']]
df_se.head()
# %%
df_1= df_se[df_se['clarity'] == 'VS1']
df_1


# %%
df_11 = df_1.groupby(['cut','color']).mean()
#%%
round(df_11,2)
#%%
df_11.groupby(['cut']=='Ideal').max()
#%%
df_22 = round(df_11,2)
#%%
print(df_22.price.values.tolist())
#%%
df_22
# %%
x = PrettyTable()
x.field_names = ['Type','D', 'E', 'F', 'G', 'H', 'I', 'J','Max','Min']
    
x.add_row(['Ideal',2576.04, 2175.8, 3504.0, 4116.92, 3613.33, 3944.42, 4734.43,'J','E'])
x.add_row(['Premium', 4178.05, 3721.7, 4758.04, 4435.82, 3949.34, 5339.37, 5817.26,'J','E'])
x.add_row(['Very Good', 2955.48, 3089.36, 3880.8, 3770.15, 3750.2, 5276.97, 4339.59,'I','D'])
x.add_row(['Good', 3556.58, 3712.78, 2787.51, 4302.43, 3819.12, 4597.17, 3662.83,'I','F'])
x.add_row(['Fair', 2921.2, 3307.93, 4103.06, 3497.62, 4604.75, 4500.48, 5906.19, 'J','D'])
x.add_row(['Max', 'Premium','Premium','Premium','Premium','Fair','Premium','Fair','',''])
x.add_row(['Min', 'Ideal','Ideal','Good','Fair','Ideal','Ideal','Good','',''])


        #x.title('Corr table of '+ com)
print(x.get_string(title = 'Bonus Question'))
# %%
df['cut'].value_counts()
# %%
