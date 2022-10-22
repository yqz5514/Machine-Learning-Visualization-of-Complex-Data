import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
style = ['whitegrid']
sns.set_style(style=style[0])
flights = sns.load_dataset('flights')
tips = sns.load_dataset('tips')
diamonds = sns.load_dataset('diamonds')
penguins = sns.load_dataset('penguins')
iris = sns.load_dataset('iris')

sns.displot(data = tips,
            x = 'total_bill',
            kde = True,
           )
plt.show()

sns.kdeplot(data = diamonds,
            x = 'price',
            hue = 'clarity',
            fill = True,
            alpha = 0.5,
            )
plt.show()

# sns.lmplot(data = tips,
#             x = 'total_bill',
#             y = 'tip',
#            )
# plt.show()

# sns.kdeplot(data = tips,
#             x = 'total_bill',
#             y = 'tip',
#             fill = True)
# plt.show()

# sns.kdeplot(data = diamonds,
#             x = 'price',
#             hue = 'cut',
#             log_scale = True,
#             fill = True,
#             alpha = 0.5,
#             palette = 'crest')
# plt.show()

# sns.kdeplot(data = tips,
#             x = 'total_bill',
#             bw_adjust = 5, cut=0)
# plt.show()

# sns.pairplot(data = penguins,
#              hue = 'species')
# plt.show()
# flights = flights.pivot('month', 'year', 'passengers')
# ax = sns.heatmap(flights,cmap = 'YlGnBu', linewidth = 0.5)
# plt.title('Heat Map')
# plt.show()

# data = np.random.normal(size=(100,6)) + np.arange(6)/2
# sns.boxplot(data=data)
# plt.show()
#
# sns.barplot(data = diamonds,
#             x = 'color',
#             y = 'price')
# plt.show()

# sns.relplot(data = flights,
#             x = 'year',
#             y = 'passengers',
#             kind = 'line',
#             hue = 'month')
# plt.show()


# sns.relplot(data = tips,
#             x = 'total_bill',
#             y = 'tip',
#             kind = 'scatter',
#             hue = 'day',
#             col = 'time')
# plt.show()

# sns.relplot(data = tips,
#             x = 'total_bill',
#             y = 'tip',
#             kind = 'scatter',
#             hue = 'sex',
#             col = 'time',
#             row = 'smoker')
# plt.show()

# color_group = diamonds.groupby('color').count()
# sns.barplot(data = color_group,
#               x = color_group.index,
#               y = 'cut')
# plt.show()
# sns.lineplot(data = flights,
#              y = 'passengers',
#              x = 'year',
#              hue = 'month')
# plt.show()
# sns.boxplot(data = flights.passengers)
# plt.show()

# sns.countplot(data = tips,
#               x = 'sex')
# plt.show()
#
# sns.countplot(data = tips,
#               x = 'smoker')
# plt.show()
#
# sns.countplot(data = tips,
#               x = 'day')
# plt.show()
# sns.countplot(data = tips,
#               x = 'day',
#               hue = 'sex')
# plt.show()
#
# # Decsending order
# sns.countplot(data = diamonds,
#               y = 'clarity',
#               order = diamonds['clarity'].value_counts().index)
# plt.show()
#
# # Decsending order
# sns.countplot(data = diamonds,
#               x = 'clarity',
#               order = diamonds['clarity'].value_counts().index[::-1])
# plt.show()
#
# sns.countplot(data = diamonds,
#             x = 'color')
# plt.show()
