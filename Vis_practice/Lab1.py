#%%
#from cProfile import label
#from http.client import CONTINUE
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd



#%%
# 1
np.random.seed(123)

m = int(input("Enter the mean for x: "))
v = int(input("Enter the variance for x: "))
n = int(input("Enter the number of observation for x: "))

m_y = int(input("Enter the mean for y: "))
v_y = int(input("Enter the variance for y: "))
n_y = int(input("Enter the number of observation for y: "))

#500:500
#sample_size = n + n_y
#sample_size
#%%

while n != 1000:
   n = int(input("Enter the number of observation for x: "))
   n_y = int(input("Enter the number of observation for y: "))
   if sample_size == 1000:
       continue

if sample_size == 1000:
      print("yes")
   


#%%
while sample_size != 1000: 
    n = int(input("Enter the number of observation for x: "))
    n_y = int(input("Enter the number of observation for y: "))
    
    if sample_size == 1000: 
      print(" Vaild Input")

#%%
if sample_size == 1000:
    print(" Vaild Input, Sum of sample should be 1000")
    
else :
    print("not vaild imput, the sum pf sample size shoube be 1000")
 

#%%     
    
x = np.random.normal(loc = m, scale = math.sqrt(v), size = n)
y = np.random.normal(loc = m_y, scale = math.sqrt(v_y), size = n_y)



  
# %%
# 2 
#my_rho = np.corrcoef(x_simple, y_simple)

#print(my_rho)

#pearson_corr = (sum (x - m) * (y - m_y) ) / math.sqrt(sum(math.pow((x - m), 2))* sum(math.pow((y - m_y), 2)))
p2 = np.cov(x, y) / (np.std(x) *np.std(y))
p2
#array([[0.7119568 , 0.0096081 ],
   #    [0.0096081 , 1.41021482]])
# %%
pearson_corr = (sum ((x - m) * (y - m_y)) ) / np.sqrt(sum(np.power((x - m), 2))* sum(np.power((y - m_y), 2)))
pearson_corr
#0.01
# %%
my_rho = np.corrcoef(x, y)
print(my_rho)
#[[1.         0.00958889]
# [0.00958889 1.        ]]
# %%
# 3 
print("The sample mean of random variable x is :")
print("The sample mean of random variable y is :")
print("The sample variance of random variable x is :")
print("The sample variance of random variable y is :")
print("The sample Pearsons correlation coefficient between x & y is:")


#%%
# 4
plt.plot(x, label = 'Random Variable x')
plt.plot(y, label = 'Random Variable y')
plt.legend(loc = 1)
plt.title(loc = 'center', label = 'Line plot for x and y')

#%%
plt.hist(x, label = 'Random Variable x')
plt.hist(y, label = 'Random Variable y')
plt.legend(loc = 1)
plt.title(loc = 'center', label = 'Histogram plot for x and y')
# %%
df = pd.read_csv("tute1.csv")
df.head()
# %%
df.isnull().sum()
#%%
df.dtypes
# %%
def get_pearsoncoef(i, j):
    pearson_corr = (sum ((i - i.mean()) * (j - j.mean())) ) / np.sqrt(sum(np.power((i - i.mean()), 2))* sum(np.power((j - j.mean()), 2)))
    return pearson_corr


# %%
get_pearsoncoef(df['Sales'], df['AdBudget'])
# %%
get_pearsoncoef(df['Sales'], df['GDP'])

#%%
get_pearsoncoef(df['GDP'], df['AdBudget'])

# %%
print(f"The sample Pearson’s correlation coefficient between Sales & AdBudget is: " )
print(f"The sample Pearson’s correlation coefficient between Sales & GDP is: " )
print(f"The sample Pearson’s correlation coefficient between AdBudget & GDP is: {}" )

#%%
df.rename(columns={"Unnamed: 0": "Date"}, inplace = True)
#%%
df.columns
#%%
df["Date"] = pd.to_datetime(df["Date"])

#%%
plt.figure(figsize=(20,15))

plt.plot(df['Date'], df['Sales'], marker = 'o', label = 'Sales', color = 'red' )
plt.plot(df['Date'], df['AdBudget'], marker = 'o', label = 'AdBudget', color = 'green' )
plt.plot(df['Date'], df['GDP'], marker = 'o', label = 'GDP')
plt.legend(loc = 1)
plt.title('Sales, AdBudget and GDP versus time')
plt.xlabel('Date')
plt.ylabel('Amount')
# %%
plt.figure(figsize=(15,10))

plt.hist(df['Sales'], label = 'Sales', color = 'red' )
plt.hist(df['AdBudget'], label = 'AdBudget', color = 'green' )
plt.hist(df['GDP'], label = 'GDP')
plt.legend(loc = 1)
plt.title('Sales, AdBudget and GDP')
plt.xlabel('Amount')
plt.ylabel('Freq')
# %%
