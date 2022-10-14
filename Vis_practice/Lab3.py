#Lab3
#%%
#library
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


#%%
#1. Load the dataset using pandas package. Clean the dataset by removing the ‘nan’ and missing data.
ucl = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/CONVENIENT_global_confirmed_cases.csv'
df = pd.read_csv(ucl)
#%%
df.head()
#%%
df.isnull().sum().to_string()
#%%
df.dropna(how ='any', inplace = True)
#%%
df.head()
#%%
df.isnull().sum()
#%%
missing_df = df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['variable', 'missing values']
missing_df['filling factor (%)']=(df.shape[0]-missing_df['missing values'])/df.shape[0]*100
missing_df.sort_values('filling factor (%)').reset_index(drop = True)

#%%


#%%
#2. The country “China” has multiple columns (“ China.1”, “China.2”, …) . 
# Create a new column name it “China_sum” which contains the sum of “China.1” + “China.2”, … column wise. 
# You can use the following command to perform the task:
print(df.columns.values.tolist())
#%%
df.iloc[0:,58:90]
#%%
df['China_sum'] = df.iloc[0:,58:90].astype(float).sum(axis=1)
#%%
df['China_sum']
#%%
##3. Repeat step 2 for the “United Kingdom”.

df.iloc[0:,250:261]
#%%
df['UK_sum'] = df.iloc[0:,250:260].astype(float).sum(axis=1)
#%%
df['UK_sum']
#%%
df['China']


#%%
#4. Plot the COVID confirmed cases for the following US versus the time. The final plot should look like bellow.

df.dtypes
#%%
df.columns
#%%
df['Country/Region'] = pd.to_datetime(df['Country/Region'])
#%%
df_us = df[['Country/Region','US']]
#%%
df.head()
#%%
# resampling
#resampled_month =df_us.resample('M', on='Country/Region').mean().reset_index(drop=False)
#%%
#resampled_month
#%%
y1=['US','United Kingdom','China','Germany','Brazil','India','Italy']
plt.figure(figsize=(16, 8))
df.plot(x = 'Country/Region', y = 'US', label = "US" )
#df.plot(x = 'Country/Region', y = 'United Kingdom', label = "United Kingdom")
#df.plot(x = 'Country/Region', y = 'China', label = "China")
#df.plot(x = 'Country/Region', y = 'Germany', label = "Germany")
#df.plot(x = 'Country/Region', y = 'Brazil', label = "Brazil")
#df.plot(x = 'Country/Region', y = 'India', label = "India")
#df.plot(x = 'Country/Region', y = 'Italy', label = "Italy")

plt.title('US confirmed cases')
# adding Label to the x-axis
plt.ylabel('Confirmed Covid-19 cases')
plt.xlabel('Year')
#plt.xticks(['Feb2020','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov'])
plt.tight_layout()
#plt.grid(10)
# adding legend to the curve
plt.legend()
#%%
#5. Repeat step 4 for the “United Kingdom”, “China”, ”Germany”, ”Brazil”, “India” and “Italy”. 
# The final plot should look like bellow.
df_11 = df[['Country/Region','US','UK_sum','China_sum','Germany','Brazil','India','Italy']]
#%%
ax = df_11.plot(x = 'Country/Region')

# Additional customizations
ax.set_xlabel('Date')
ax.set_ylabel('Confirmed Covid-19 cases')
ax.set_title('GLobal confirmed cases')

ax.legend(fontsize=12)
#%%
ax = df_11.plot(x = 'Country/Region')

# Additional customizations
ax.set_xlabel('Date')
ax.set_ylabel('Confirmed Covid-19 cases')
ax.set_title('GLobal confirmed cases')

ax.legend(fontsize=12)

#%%
#6. Plot the histogram plot of the graph in Question 4.
plt.figure(figsize=(10,8))

plt.hist(df_us['US'], label = "US",bins=50 )
plt.legend(loc = 1)
plt.title('distribution of cases')
plt.xlabel('number of cases')
plt.ylabel('freq')
plt.tight_layout()
plt.legend()
plt.show()
#%%
#y1=['UK_sum','China_sum','Germany','Brazil','India','Italy']

plt.figure(figsize=(12,12))
plt.subplot(3,2,1)
plt.hist(df['UK_sum'], label = "UK_sum",bins=50 )
plt.legend(loc = 1)
plt.title('distribution of cases')
plt.xlabel('number of cases')
plt.ylabel('freq')

plt.subplot(3,2,2)
plt.hist(df['China_sum'], label = "China_sum",bins=50 )
plt.legend(loc = 1)
plt.title('distribution of cases')
plt.xlabel('number of cases')
plt.ylabel('freq')

plt.subplot(3,2,3)
plt.hist(df['Germany'], label = "Germany",bins=50 )
plt.legend(loc = 1)
plt.title('distribution of cases')
plt.xlabel('number of cases')
plt.ylabel('freq')

plt.subplot(3,2,4)
plt.hist(df['Brazil'], label = "Brazil",bins=50 )
plt.legend(loc = 1)
plt.title('distribution of cases')
plt.xlabel('number of cases')
plt.ylabel('freq')

plt.subplot(3,2,5)
plt.hist(df['India'], label = "India",bins=50 )
plt.legend(loc = 1)
plt.title('distribution of cases')
plt.xlabel('number of cases')
plt.ylabel('freq')

plt.subplot(3,2,6)
plt.hist(df['Italy'], label = "Italy",bins=50 )
plt.legend(loc = 1)
plt.title('distribution of cases')
plt.xlabel('number of cases')
plt.ylabel('freq')

plt.tight_layout()

plt.show()

#%%
y1=['UK_sum','China_sum','Germany','Brazil','India','Italy']
plt.figure(figsize=(12,12))
for j in y1:
    for i in range(6):
       plt.hist(df[j], label = j ,bins=50 )
       plt.subplot(3,2,i+1)
       plt.legend(loc = 1)
       plt.title('distribution of cases')
       plt.xlabel('number of cases')
       plt.ylabel('freq')
    plt.tight_layout()
plt.show()
#%%
df.head()
#7. Plot the histogram plot of the graph in Question 5. Use subplot 3x2. Not shared axis.
#8. Which country (from the list above) has the highest mean, variance and median of # of COVID confirmed cases?
#%%
y1=['US','UK_sum','China_sum','Germany','Brazil','India','Italy']

for i in y1:
    print(i, round(df[i].mean(),2), round(df[i].var(),2),round(df[i].median(),2))
    
#%%
df['US'].mean()
#For the second part of this LAB you will learn how to use pandas package for data filtering and data selection 
# and use the matplotlib package for visualization of pie chart. The dataset for this section of the LAB will is called ‘titanic’.
# The titanic dataset can be uploaded using the following: 
#%%
import matplotlib.pyplot as plt 
import seaborn as sns 
df = sns.load_dataset('titanic')
#1- The titanic dataset needs to be cleaned due to nan entries. Remove all the nan in the dataset using “”dropna()” method. 
# Display the first 5 row of the dataset.
#%%
df.isnull().sum()
#%%
df.info
#%%
df.dropna(how ='any', inplace = True)
df.isnull().sum()
#2- Write a python program that plot the pie chart and shows the number of male and female on the titanic dataset. 
# The final answer should look like bellow.
#%%
df.head(5)
#%%
import pandas as pd
#%%
df_s = pd.DataFrame(df.sex.value_counts())
#%%
df_s
#%%
#lb =['94','88']
explode = (0.03,0.03)
value = df_s['sex'].values
plt.figure()
plt.pie(df_s['sex'],labels = df_s.index, explode=explode,autopct = lambda x: '{:.0f}'.format(x*value.sum()/100))#display num
plt.axis('square') # make sure the plot will be in circle shape
plt.legend(loc= (.85,.85))

plt.title('Numbers of male and female', loc='center')
plt.show()
    
#3- Write a python program that plot the pie chart and shows the percentage of male and female on the titanic dataset. The final answer should look like bellow.
#%%
plt.figure()
explode = (0.03,0.03)

plt.pie(df_s['sex'],labels = df_s.index, explode=explode, autopct='%1.2f%%')#display percentage
plt.legend(loc= (.85,.85))
plt.axis('square') # make sure the plot will be in circle shape
plt.title('Numbers of male and female in %', loc='center')
plt.show()
#4- the percentage of males who survived versus the percentage of males who did not survive. 
#%%
df.columns

#%%
df_1 = df[df['sex']=='male']
#%%
df_ms = pd.DataFrame(df_1.survived.value_counts())
df_ms
#%%
plt.figure()
explode = (0.03,0.03)
lb =['Male not survived', 'Male survived']
plt.pie(df_ms['survived'], labels=lb, explode=explode, autopct='%1.2f%%')#display percentage
plt.legend(loc= (.85,.85))
plt.axis('square') # make sure the plot will be in circle shape
plt.title('Pie chart of male survivale in titanic', loc='center')
plt.show()
#5- the percentage of females who survived versus the percentage of females who did not survive. 
#%%
#%%
df_2 = df[df['sex']=='female']
#%%
df_ws = pd.DataFrame(df_2.survived.value_counts())
df_ws
#%%
plt.figure()
explode = (0.03,0.03)
lb =['Female survived', 'Female not survived']
plt.pie(df_ws['survived'], labels=lb, explode=explode, autopct='%1.2f%%')#display percentage
plt.legend(loc= (.85,.85))
plt.axis('square') # make sure the plot will be in circle shape
plt.title('Pie chart of female survivale in titanic', loc='center')
plt.show()
#6- the percentage passengers with first class, second class and third-class tickets. 
#%%
df.columns
#%%
df['pclass'].value_counts()

#7- the survival percentage rate based on the ticket class. 
#%%
#df_ps0 = df[df['survived']==0].groupby(df['pclass']).count()
df_ps1 = df[df['survived']==1].groupby(df['pclass']).count()
df_ps1
#%%
df_ps1.head()
#%%
q7 = df_ps1['survived']/59
#%%
#%%
plt.figure()
explode = (0.3,0.08,0.08)
lb =['ticket class 1', 'ticket class 2','ticket class 3']
plt.pie(q7, labels=lb, explode=explode, autopct='%1.2f%%')#display percentage
plt.legend(loc= (.85,.85))
plt.axis('square') # make sure the plot will be in circle shape
plt.title('Survival rate base on level', loc='center')
plt.show()
#%%
#8- the percentage passengers who survived versus the percentage of passengers who did not survive with the first-class ticket category. The final answer should look like bellow.
#%%
#%%
plt.figure()
explode = (0.02,0.02)
data = [157-106, 106]
lb =['Death rate', 'Survival rate']
plt.pie(data, labels=lb, explode=explode, autopct='%1.2f%%')#display percentage
plt.legend(loc= (.85,.85))
plt.axis('square') # make sure the plot will be in circle shape
plt.title('Survival&Death rate: ticker class 1', loc='center')
plt.show()
#9- the percentage passengers who survived versus the percentage of passengers who did not survive with the second-class ticket category. The final answer should look like bellow.
#%%
plt.figure()
explode = (0.02,0.02)
data = [15-12, 12]
lb =['Death rate', 'Survival rate']
plt.pie(data, labels=lb, explode=explode, autopct='%1.2f%%')#display percentage
plt.legend(loc= (.85,.85))
plt.axis('square') # make sure the plot will be in circle shape
plt.title('Survival&Death rate: ticker class 2', loc='center')
plt.show()
#10- Write a python program that plot the pie chart showing the percentage passengers who survived versus the percentage of passengers who did not survive in the third-class ticket category.
plt.figure()
explode = (0.02,0.02)
data = [10-5, 5]
lb =['Death rate', 'Survival rate']
plt.pie(data, labels=lb, explode=explode, autopct='%1.2f%%')#display percentage
plt.legend(loc= (.85,.85))
plt.axis('square') # make sure the plot will be in circle shape
plt.title('Survival&Death rate: ticker class 3', loc='center')
plt.show()
#11- Using the matplotlib and plt.subplots create a dashboard which includes all the pie charts above. Note: Use the figure size = (16,8). The final answer should look like the following.
#All the figure should have the appropriate title and legend with a correct label.
#Write an solution report and upload the .pdf file of the report and the .py through BB before the deadline.
# function that extract statistical parameters from a grouby objet:

# %%
#%%
fig,axes = plt.subplots(3,3,figsize=(16,8))

axes[0].pie(new_df[new_df.HeartDisease=="No"][name_feature].value_counts(), labels=labels,autopct='%1.1f%%',textprops =textprops)
axes[0].set_title('No Heart Disease',fontsize=15)
axes[1].pie(new_df[new_df.HeartDisease=="Yes"][name_feature].value_counts(), labels=labels,autopct='%1.1f%%',textprops =textprops)
axes[1].set_title('Yes Heart Disease',fontsize=15)

plt.legend(title = name_feature, fontsize=15, title_fontsize=15)
plt.show()