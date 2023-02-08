import sys
import pandas as pds
import mglearn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import IPython
import sklearn

from sklearn.datasets import load_iris
iris_dataset = load_iris();
# Объект iris, возвращаемый load_iris, является объектом Bunch, который очень похож на словарь.
print("Ключи iris_dataset: \n{}".format(iris_dataset.keys()))
# Значение ключа DESCR–это краткое описание набора данных.
print(iris_dataset['DESCR'][:193] + "\n...")

print("Названия ответов: {}".format(iris_dataset['target_names']))

print("Названия признаков: \n{}".format(iris_dataset['feature_names']))

print("Тип массива data: {}".format(type(iris_dataset['data'])))

print("Форма массива data: {}".format(iris_dataset['data'].shape))
print("Первые пять строк массива data:\n{}".format(iris_dataset['data'][:5]))
# Значение ключа target_names–это массив строк, содержащий сорта цветов, которые мы хотим предсказать
print("Тип массива target: {}".format(type(iris_dataset['target'])))
print("Форма массива target: {}".format(iris_dataset['target'].shape))

print("Ответы:\n{}".format(iris_dataset['target']))

print("0 –setosa;\n1 –versicolor;\n2 –virginica;")
print("Ответы:\n{}".format(iris_dataset['target']))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
iris_dataset['data'], iris_dataset['target'], random_state=0)

print("Форма массива X_train: {}".format(X_train.shape))
print("Форма массива y_train: {}".format(y_train.shape))

print("Форма массива X_test: {}".format(X_test.shape))
print("Форма массива y_test: {}".format(y_test.shape))

# создаем dataframe изданных в массиве X_train
# маркируем столбцы, используя строкив iris_dataset.feature_names
iris_dataframe = pds.DataFrame(X_train, columns=iris_dataset.feature_names)
# создаем матрицу рассеяния из dataframe, цвет точек задаем спомощью y_train
from pandas.plotting import scatter_matrix

grr = scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='0',
                     hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

plt.show()
