import pandas as pd
import plotly.express as px
import seaborn as sns
import numpy as np
import pandas_datareader as web
import plotly.graph_objects as go
import plotly.subplots
from plotly.subplots import make_subplots

#%%
fig = make_subplots(rows=1, cols= 2)

fig.add_trace(
    go.Scatter(x = [1,2,3], y = [4,5,6]),
               row=1, col=1)



fig.add_trace(
    go.Scatter(x = [1,2,3], y = [4,5,6]),
               row=1, col=2)

fig.update_layout(width = 800, height = 600,
                  title = 'Side by side fig')
fig['layout']['xaxis']['title'] = 'label x -axis 1'
fig['layout']['xaxis2']['title'] = 'label x -axis 2'
fig['layout']['yaxis']['title'] = 'label y -axis 1'
fig['layout']['yaxis2']['title'] = 'label y -axis 2'
fig.write_html('first_figure.html', auto_open=True)

#%%
# iris = px.data.iris()
# tip = px.data.tips()
# gapminder = px.data.gapminder()
# diamonds = sns.load_dataset('diamonds')
# diamonds_color = diamonds.groupby('color').sum()
# diamonds_color = diamonds_color.reset_index()

# fig = go.Figure()
#
# fig.add_trace(go.Histogram(x = iris['sepal_width'], nbinsx=40))
# fig.add_trace(go.Histogram(x = iris['sepal_length'], nbinsx=40))
# fig.update_layout(barmode = 'stack')

# fig = px.histogram(iris,
#                    x = 'sepal_length',
#                    width=800, height=400,
#                    color='species',
#                    nbins=50,
#                    marginal='box',
#                    )
# df1 = iris.groupby('species').mean()

# fig = px.bar(tip,
#              x = 'total_bill',
#              y = 'day',
#              color = 'sex',
#              barmode='group')

# fig = px.bar(diamonds_color,
#              y = 'price',
#              x = 'color',
#            )


# fig = px.choropleth(gapminder,
#                  locations = 'iso_alpha',
#                     projection='natural earth',
#                     hover_name = 'country',
#                     color='lifeExp',
#                  animation_frame = 'year',
#                   color_continuous_scale=px.colors.sequential.Plasma,
#                  title = 'life exp per country',
#
#                  )



# fig = px.scatter(gapminder,
#                  x = 'gdpPercap',
#                  y = 'lifeExp',
#                  size = 'pop',
#                  color='continent',
#                  size_max= 55,
#                  animation_frame = 'year',
#                  animation_group = 'country',
#                  range_x = [0,60000],
#                  range_y = [25,90],
#                  title = 'GDP per capital and life Exc',
#
#                  )

# fig = px.bar(iris,
#               x = 'species',
#               y = ['petal_length',
#                    'petal_width',
#                    'sepal_length',
#                    'sepal_width'],
#              title='iris bar plot',
#               width=800,height=400,
#               labels={'value':'dimension (cm)'},
#              template='plotly_white')


# stock = 'FRD'
# df = web.DataReader(stock, data_source = 'yahoo',
#                     start = '2000-01-01',
#                     end = '2022-10-26')
# df.drop('Volume', axis = 1, inplace = True)
#
# fig = px.line(df,
#               y = df.columns,
#               width=800, height=400,
#               title='Ford Stock price',
#               labels={'value': 'USD($)'})


# x = np.linspace(-8,8,100)
# y = x ** 2
# z = x ** 3
# h = np.vstack((x,y,z))
#
# df = pd.DataFrame(data= h.T, columns=['x','$x^{2}$', '$x^{3}$'])
# fig = px.line(df ,
#               x = 'x',
#               y = ['$x^{2}$','$x^{3}$'],
#               width=800, height=400,
#               title=r'$\text{Graph of } x^{ 2 } \& x^{ 3}$',
#               labels= {'value':'USD($)',
#                        'x':'first variable'})
# fig.update_layout(
#     title_font_size = 20,
#     title_font_color = 'red',
#     title_font_family = 'Times New Roman',
#     legend_title_font_size = 20,
#     legend_title_font_color = 'green',
#     font_color = 'blue',
#     font_size = 20,
#     font_family = 'Courier New'
#
# )


fig.show(renderer = 'browser')


