#%%
# library
import seaborn as sns 
import matplotlib.pyplot as plt

#%%
# Q1
penguins = sns.load_dataset('penguins')

# %%
penguins.tail(5)
# %%
penguins.describe()
# %%

#Q2
penguins.isnull().sum()
# %%
penguins.dropna(how='any',inplace = True)
# %%
# check
penguins.isnull().sum()

# %%
plt.style.use('seaborn-darkgrid')


# %%
sns.histplot(data=penguins,
             x='flipper_length_mm',
             #binwidth = 3
    
)
plt.show()
# %%
# q4
sns.histplot(data=penguins,
             x='flipper_length_mm',
             binwidth = 3
    
)
plt.show()

#%%
#q5 bins
sns.histplot(data=penguins,
             x='flipper_length_mm',
             binwidth=3,
             bins=30 
)
plt.show()
# %%
#q6
sns.displot(data=penguins,
             x='flipper_length_mm',
             hue='species',
             binwidth=3,
             bins=30
             
)
plt.show()
# %%
# 6 element=’step’
sns.displot(data=penguins,
             x='flipper_length_mm',
             hue='species',
             binwidth=3,
             bins=30,
             element='step'
             
)
plt.show()
# %%
sns.histplot(data=penguins,
             x='flipper_length_mm',
             hue='species',
             multiple='stack',
             binwidth=3,
             bins=30 
)
plt.show()
# %%
# 9
sns.displot(data=penguins,
             x='flipper_length_mm',
             hue='sex',
             binwidth=3,
             bins=30,
             #element='step'
             multiple='dodge'
             
)
plt.show()
# %%
sex_df = penguins[['flipper_length_mm','sex']]
#%%
male = sex_df[sex_df['sex']=='Male']
#male
female = sex_df[sex_df['sex']=='Female']
#%%
#10
fig, axs =plt.subplots(1,2,figsize=(15, 5), sharey=True)
sns.histplot(x='flipper_length_mm',
            data=male,
            hue='sex',
            ax=axs[0])
sns.histplot(x='flipper_length_mm',
            data=female,
            hue='sex',
            ax=axs[1])
fig.show()
#`displot` is a figure-level function and does not accept the ax= parameter. You may wish to try histplot.
# %%

