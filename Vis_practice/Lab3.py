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
df.isnull().sum()
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
##3. Repeat step 2 for the “United Kingdom”.

df.iloc[0:,250:260]
#%%
df['UK_sum'] = df.iloc[0:,250:260].astype(float).sum(axis=1)
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
# resampling
resampled_month =df_us.resample('M', on='Country/Region').mean().reset_index(drop=False)
#%%
resampled_month
#%%
y1=['US','United Kingdom','China','Germany','Brazil','India','Italy']
plt.figure(figsize=(16, 8))
df.plot(x = 'Country/Region', y = y1, label = "US" )
#df.plot(x = 'Country/Region', y = 'United Kingdom', label = "United Kingdom")
#df.plot(x = 'Country/Region', y = 'China', label = "China")
#df.plot(x = 'Country/Region', y = 'Germany', label = "Germany")
#df.plot(x = 'Country/Region', y = 'Brazil', label = "Brazil")
#df.plot(x = 'Country/Region', y = 'India', label = "India")
#df.plot(x = 'Country/Region', y = 'Italy', label = "Italy")

plt.title('GLobal confirmed cases')
# adding Label to the x-axis
plt.ylabel('Confirmed Covid-10 cases')
plt.xlabel('Year')
#plt.xticks(['Feb2020','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov'])
plt.tight_layout()
#plt.grid(10)
# adding legend to the curve
plt.legend()
#%%
df_11 = df[['Country/Region','US','UK_sum','China_sum','Germany','Brazil','India','Italy']]
#%%
ax = df_11.plot(x = 'Country/Region')

# Additional customizations
ax.set_xlabel('Date')
ax.legend(fontsize=12)
#%%
#5. Repeat step 4 for the “United Kingdom”, “China”, ”Germany”, ”Brazil”, “India” and “Italy”. 
# The final plot should look like bellow.
#6. Plot the histogram plot of the graph in Question 4.
#7. Plot the histogram plot of the graph in Question 5. Use subplot 3x2. Not shared axis.
#8. Which country (from the list above) has the highest mean, variance and median of # of COVID confirmed cases?
#For the second part of this LAB you will learn how to use pandas package for data filtering and data selection 
# and use the matplotlib package for visualization of pie chart. The dataset for this section of the LAB will is called ‘titanic’.
# The titanic dataset can be uploaded using the following: 
import matplotlib.pyplot as plt 
import seaborn as sns 
df = sns.load_dataset('titanic')
#1- The titanic dataset needs to be cleaned due to nan entries. Remove all the nan in the dataset using “”dropna()” method. Display the first 5 row of the dataset.
#2- Write a python program that plot the pie chart and shows the number of male and female on the titanic dataset. The final answer should look like bellow.
#3- Write a python program that plot the pie chart and shows the percentage of male and female on the titanic dataset. The final answer should look like bellow.
#4- Write a python program that plot the pie chart showing the percentage of males who survived versus the percentage of males who did not survive. The final answer should look like bellow.
#5- Write a python program that plot the pie chart showing the percentage of females who survived versus the percentage of females who did not survive. The final answer should look like bellow.
#6- Write a python program that plot the pie chart showing the percentage passengers with first class, second class and third-class tickets. The final answer should look like bellow.
#7- Write a python program that plot the pie chart showing the survival percentage rate based on the ticket class. The final answer should look like bellow.
#8- Write a python program that plot the pie chart showing the percentage passengers who survived versus the percentage of passengers who did not survive with the first-class ticket category. The final answer should look like bellow.
#9- Write a python program that plot the pie chart showing the percentage passengers who survived versus the percentage of passengers who did not survive with the second-class ticket category. The final answer should look like bellow.
#10- Write a python program that plot the pie chart showing the percentage passengers who survived versus the percentage of passengers who did not survive in the third-class ticket category.
#11- Using the matplotlib and plt.subplots create a dashboard which includes all the pie charts above. Note: Use the figure size = (16,8). The final answer should look like the following.
#All the figure should have the appropriate title and legend with a correct label.
#Write an solution report and upload the .pdf file of the report and the .py through BB before the deadline.
# function that extract statistical parameters from a grouby objet:
def get_stats(group):
    return {'min': group.min(), 'max': group.max(),
            'count': group.count(), 'mean': group.mean()}
#_______________________________________________________________
# Creation of a dataframe with statitical infos on each airline:
global_stats = df['DEPARTURE_DELAY'].groupby(df['AIRLINE']).apply(get_stats).unstack()
global_stats = global_stats.sort_values('count')
global_stats
# %%