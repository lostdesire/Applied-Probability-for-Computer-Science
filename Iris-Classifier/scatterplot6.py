import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Iris : petal_length & petal_width scatterplot

iris = pd.read_csv("E:/iris.data", names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

iris.groupby('species').size()

sns.relplot(x='petal_length', y='petal_width', data=iris, hue='species')

plt.show()
