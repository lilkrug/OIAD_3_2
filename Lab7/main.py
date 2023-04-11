
import mglearn
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

# Классификатор BernoulliNB подсчитывает ненулевые частоты признаков по каждому классу.

X = np.array([[0, 1, 0, 1], [1, 0, 1, 1], [0, 0, 0, 1], [1, 0, 1, 0]])
y = np.array([0, 1, 0, 1])

counts = {}
for label in np.unique(y):
 counts[label] = X[y == label].sum(axis=0)
print("Частоты признаков:\n{}".format(counts))

clf = BernoulliNB()
clf.fit(X, y)
print("1 clf.predict:\n{}".format(clf.predict(X[0:4])))
# Классификатор MultinomialNB  работает лучше чем Классификатор BernoulliNB когда набор данных имеет большое кол-во признаков

rng = np.random.RandomState(1)
X = rng.randint(25, size=(6, 10))
y = np.array([1, 2, 3, 4, 5, 6])

clf = MultinomialNB()
clf.fit(X, y)

print("2 clf.predict:\n{}".format(clf.predict(X[0:6])))
print(X[0:5])
# Классификатор GaussianNB испльзуется для данных с очень высокой размерность, тогда как остальные
# наивные байевсовские модели широко исп для дискретных данных

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
clf = GaussianNB()
clf.fit(X, Y)

print("3 clf.predict:\n{}".format(clf.predict([[1, -1]])))


clf_pf = GaussianNB()
clf_pf.partial_fit(X, Y, np.unique(Y))

print("4 clf.predict:\n{}".format(clf_pf.predict([[-0.8, 3]])))

# Преимущества: быстро обучаются, быстро прогнозируются, легкий процесс обучения.
# Недостатки: обучение занимают слишком много времени