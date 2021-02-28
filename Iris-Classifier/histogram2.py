import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Iris : sepal_width histogram

iris = pd.read_csv("E:/iris.data", names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

iris.groupby('species').size()

sns.distplot(iris[iris.species == 'Iris-setosa']['sepal_width'],
             kde=False, color='red', label='setosa', bins=np.arange(2, 4.5, 0.2))
sns.distplot(iris[iris.species == 'Iris-versicolor']['sepal_width'],
             kde=False, color='green', label='versicolor', bins=np.arange(2, 4.5, 0.2))
sns.distplot(iris[iris.species == 'Iris-virginica']['sepal_width'],
             kde=False, color='blue', label='virginica', bins=np.arange(2, 4.5, 0.2))

plt.show()
