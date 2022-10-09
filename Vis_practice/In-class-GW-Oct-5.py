import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel('Sample - Superstore.xls')
df1 = df.groupby('Region').sum()
plt.figure()
plt.bar(df1.index, df1['Profit'])
plt.show()


# width = 0.4
# score_men = [23,17,35,29,20]
# score_women = [35,25,10,15,30]
# label = ['C', 'C++', 'Java', 'Python', 'PHP']
# x = np.arange(len(label))
# fig, ax = plt.subplots()
# ax.barh(x - width/2, score_men, width, label='men')
# ax.barh(x + width/2, score_women, width, label='women')
# ax.set_xlabel('scores')
# ax.set_ylabel('Computer languages')
# ax.set_yticks(x)
# ax.set_yticklabels(label)
# ax.legend()
# plt.show()
#




# score_men = np.array([23,17,35,29,20])
# score_women = np.array([35,25,10,15,30])
# label = ['C', 'C++', 'Java', 'Python', 'PHP']
#
# fig = plt.figure()
# plt.barh(label, score_men, label ='men')
# plt.barh(label, score_women, left = score_men,label='women')
# plt.xlabel('score')
# plt.title('Simple stack Bar Plot')
# plt.ylabel('Computer Languages')
# plt.legend()
# plt.show()


# width = 0.4
# score_men = [23,17,35,29,20]
# score_women = [35,25,10,15,30]
# label = ['C', 'C++', 'Java', 'Python', 'PHP']
# x = np.arange(len(label))
# fig, ax = plt.subplots()
# ax.bar(x - width/2, score_men, width, label='men')
# ax.bar(x + width/2, score_women, width, label='women')
# ax.set_ylabel('scores')
# ax.set_xlabel('Computer languages')
# ax.set_xticks(x)
# ax.set_xticklabels(label)
# ax.legend()
# plt.show()



# score_men = np.array([23,17,35,29,20])
# score_men_norm = score_men/np.sqrt(np.sum(np.square(score_men)))
# score_women = np.array([35,25,10,15,30])
# score_women_norm = score_women/np.sqrt(np.sum(np.square(score_women)))
# label = ['C', 'C++', 'Java', 'Python', 'PHP']
# explode = (0.03,0.03,0.1,0.03,0.03)
# plt.figure()
# plt.bar(label, score_men_norm, label='men')
# plt.bar(label, score_women_norm, bottom=score_men_norm,label='women')
# plt.ylabel('score')
# plt.title('Simple stack Bar Plot')
# plt.xlabel('Computer Languages')
# plt.legend()
# plt.show()



# data = [23,17,32,29,12]
# label = ['C', 'C++', 'Java', 'Python', 'PHP']
# explode = (0.03,0.03,0.1,0.03,0.03)
# plt.figure()
# plt.pie(data, labels=label, explode=explode, autopct='%1.2f%%')
# plt.legend(loc = (.8,.8))
# plt.axis('square')
# plt.show()





# url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/mnist_test.csv'
# df = pd.read_csv(url)

# for i in range(10):
#     df0 = df[df['label']==i]
#     print(f'Number of observations corresponding to {i} is {df0.shape[0]}')
#     pic = df0[0:1].values.reshape(785)[1:].reshape(28, 28)
#     plt.figure()
#     plt.imshow(pic)
#     plt.show()
# plt.figure(figsize=(12,12))
# for i in range(10):
#     df0 = df[df['label']==i]
#     # print(f'Number of observations corresponding to {i} is {df0.shape[0]}')
#     pic = df0[0:1].values.reshape(785)[1:].reshape(28, 28)
#     plt.subplot(4,3,i+1)
#     plt.imshow(pic,vmin=0)
# plt.tight_layout()
# plt.show()

