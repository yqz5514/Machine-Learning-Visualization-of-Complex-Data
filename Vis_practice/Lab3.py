#Lab3

#1. Load the dataset using pandas package. Clean the dataset by removing the ‘nan’ and missing data.
#2. The country “China” has multiple columns (“ China.1”, “China.2”, …) . 
# Create a new column name it “China_sum” which contains the sum of “China.1” + “China.2”, … column wise. 
# You can use the following command to perform the task:
#3. Repeat step 2 for the “United Kingdom”.
#4. Plot the COVID confirmed cases for the following US versus the time. The final plot should look like bellow.
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
