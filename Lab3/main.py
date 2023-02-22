import mglearn
import matplotlib.pyplot as plt
import numpy as np

X, y = mglearn.datasets.make_forge()
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(['Class 0', 'Classs 1'], loc=4)
plt.xlabel('First feature')
plt.ylabel('Second feature')
print(f'Size of array X: {X.shape}')


X, y = mglearn.datasets.make_wave()
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel('Feature')
plt.ylabel('Label variable')


from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
print(f'Keys cancer: {cancer.keys()}')
print(f'Size: {cancer["data"].shape}')
print(cancer["feature_names"])
print(cancer["data"][:1])


print(cancer['DESCR'])


from sklearn.datasets import load_boston

boston = load_boston()
print(f'Size data {boston.data.shape}')
print(f"Targets {boston.feature_names}")


mglearn.plots.plot_knn_classification(n_neighbors=1)


mglearn.plots.plot_knn_classification(n_neighbors=3)


from mglearn.datasets import make_forge
from sklearn.model_selection import train_test_split

X, y = make_forge()
X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=0)

from sklearn.neighbors import KNeighborsClassifier

knc = KNeighborsClassifier(n_neighbors=3)

knc.fit(X_train, Y_train)

prediction = knc.predict(X_test)
score = knc.score(X_test, Y_test)

print(f"Prediction array: {prediction}")
print(f"Accurancy score: {score * 100:.2f}%")


fig, axes = plt.subplots(1, 3, figsize=(10, 3))

for n_neighbors, ax in zip([1, 3, 9], axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, Y_train)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=0.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title(f"Amount of neighbors {n_neighbors}")
    ax.set_xlabel(f"Feature 0")
    ax.set_xlabel(f"Feature 1")
axes[0].legend(loc=3)


X_train, X_test, Y_train, Y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=777
)

training_accurancy = []
test_accurancy = []

neighbors_settings = [i for i in range(1, 11)]

for n_neg in neighbors_settings:
    knc = KNeighborsClassifier(n_neighbors=n_neg)
    knc.fit(X_train, Y_train)
    training_accurancy.append(knc.score(X_train, Y_train))
    test_accurancy.append(knc.score(X_test, Y_test))

plt.plot(neighbors_settings, training_accurancy, label="Properly of train set")
plt.plot(neighbors_settings, test_accurancy, label="Properly of test set")
plt.ylabel("Properly")
plt.xlabel("N neighbors")
plt.legend()


mglearn.plots.plot_knn_regression(n_neighbors=1)


mglearn.plots.plot_knn_regression(n_neighbors=3)


from mglearn.datasets import make_wave
from sklearn.neighbors import KNeighborsRegressor

X, y = make_wave()
X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=0)

knr = KNeighborsRegressor(n_neighbors=1)
knr.fit(X_train, Y_train)

print(f"Prediction results:\n{knr.predict(X_test)}")
print(f"Accurancy score(R^2, determination factor): {knr.score(X_test, Y_test) * 100:.2f}%")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
line = np.linspace(-3, 3, 1000).reshape(-1, 1)

for n_neig, ax in zip([1, 3, 9], axes):
    knr = KNeighborsRegressor(n_neighbors=n_neig)
    knr.fit(X_train, Y_train)
    ax.plot(line, knr.predict(line))
    ax.plot(X_train, Y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, Y_test, 'v', c=mglearn.cm2(1), markersize=8)
    score = knr.score(X_test, Y_test)
    ax.set_title(f"{n_neig} neighbor(s)\n accurancy score: {score:.2f}%")
    ax.set_label('Feature')
    ax.set_ylabel('Proparly answer')
axes[0].legend(["Prediction model", "Traning dataset", 'Tested dataset'], loc='best')
plt.show()