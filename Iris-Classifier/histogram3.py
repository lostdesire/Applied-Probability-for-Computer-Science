import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Iris : petal_length histogram

iris = pd.read_csv("E:/iris.data", names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

iris.groupby('species').size()

sns.distplot(iris[iris.species == 'Iris-setosa']['petal_length'],
             kde=False, color='red', label='setosa', bins=np.arange(1, 7, 0.2))
sns.distplot(iris[iris.species == 'Iris-versicolor']['petal_length'],
             kde=False, color='green', label='versicolor', bins=np.arange(1, 7, 0.2))
sns.distplot(iris[iris.species == 'Iris-virginica']['petal_length'],
             kde=False, color='blue', label='virginica', bins=np.arange(1, 7, 0.2))

plt.show()
