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
sns.displot(data=penguins,
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
# sex_df = penguins[['flipper_length_mm','sex']]
# #%%
# male = sex_df[sex_df['sex']=='Male']
# #male
# female = sex_df[sex_df['sex']=='Female']
# #%%
#10
# fig, axs =plt.subplots(1,2,figsize=(15, 5), sharey=True)
# sns.histplot(x='flipper_length_mm',
#             data=male,
#             hue='sex',
#             ax=axs[0])
# sns.histplot(x='flipper_length_mm',
#             data=female,
#             hue='sex',
#             ax=axs[1])
# fig.show()
#%%
sns.displot(data=penguins, x="flipper_length_mm", col="sex")
#`displot` is a figure-level function and does not accept the ax= parameter. You may wish to try histplot.
# %%
#11
sns.displot(data=penguins, 
            x="flipper_length_mm", 
            hue="species", 
            stat = "density")

# %%
#12
sns.displot(data=penguins, 
            x="flipper_length_mm", 
            hue="sex",
            stat = "density")

# %%
#13
sns.displot(data=penguins, 
            x="flipper_length_mm", 
            hue="species", 
            stat = "probability")

# %%
#14??

sns.displot(data=penguins, x="flipper_length_mm", hue="species", kind='kde')

# %%
#15
sns.displot(data=penguins, x="flipper_length_mm", hue="sex", kind='kde')

# %%
#16
sns.displot(data=penguins, x="flipper_length_mm", hue="species", kind='kde',multiple='stack')

# %%
#17
sns.displot(data=penguins, 
            x="flipper_length_mm", 
            hue="sex", 
            
            multiple='stack',
            kind='kde'
            #fill = True
            )

# %%
#18
sns.displot(data=penguins, 
            x="flipper_length_mm", 
            hue="species", 
            kind='kde',
            fill = True
            )
# %%
#19
sns.displot(data=penguins, 
            x="flipper_length_mm", 
            hue="sex", 
            kind='kde',
            fill = True
            )
# %%
#20
sns.lmplot(data=penguins,
            x = 'bill_length_mm',
            y = 'bill_depth_mm',
            
            )
plt.show()
# %%
sns.relplot(data=penguins,
            x = 'bill_length_mm',
            y = 'bill_depth_mm',
            kind = 'scatter',
            )
plt.show()
# %%
#21
sns.countplot(data =penguins,
              x = 'island',
              hue = 'species')
plt.show
# %%
sns.countplot(data =penguins,
              x = 'sex',
              hue = 'species')
plt.show
# %%
#23
sns.kdeplot(data=penguins,
             x = 'bill_length_mm',
             y = 'bill_depth_mm',
             hue = 'sex',
             fill = True
             )
plt.show()
#%%
sns.kdeplot(data=penguins,
             x = 'bill_length_mm',
             y = 'flipper_length_mm',
             hue = 'sex',
             fill = True
             )
plt.show()

# %%
sns.kdeplot(data=penguins,
             x = 'flipper_length_mm',
             y = 'bill_depth_mm',
             hue = 'sex',
             fill = True,
             #palette='crest'
             #cbar = True
             )
plt.show()
# %%
#26
fig, axs =plt.subplots(3,1,figsize=(8,16))
sns.kdeplot(data=penguins,
             x = 'bill_length_mm',
             y = 'bill_depth_mm',
             hue = 'sex',
             fill = True,         
             ax=axs[0]
             )

sns.kdeplot(data=penguins,
             x = 'bill_length_mm',
             y = 'flipper_length_mm',
             hue = 'sex',
             fill = True,
             #log_scale= True,
             ax=axs[1]
             )

sns.kdeplot(data=penguins,
             x = 'flipper_length_mm',
             y = 'bill_depth_mm',
             hue = 'sex',
             fill = True,
             #log_scale= True,
             ax=axs[2]
             )

# axs[0].set_xlim(25, 65)
# axs[0].set_ylim(10, 24)

# axs[1].set_xlim(25, 65)
# axs[1].set_ylim(160, 260)

# axs[2].set_xlim(160, 260)
# axs[2].set_ylim(10, 24)

# fig.savefig('Q26',dpi = 500)
fig.show()
# %%
#27
sns.displot(penguins,
            x='bill_length_mm',  
            y='bill_depth_mm',
            hue='sex')  
plt.show()

# %%
#flipper_length_mm
sns.displot(penguins,
            x='bill_length_mm',  
            y='flipper_length_mm',
            hue='sex')  
plt.show()

# %%
#29
sns.displot(penguins,
            x='flipper_length_mm',  
            y='bill_depth_mm',
            hue='sex')  
plt.show()
# %%
df = penguins[['sex','flipper_length_mm']]
# %%
# df[df['sex']=='Male']['flipper_length_mm'].mean()
# # %%
# df[df['sex']=='Female']['flipper_length_mm'].mean()

# # %%
