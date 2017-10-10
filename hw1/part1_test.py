import numpy as np
from sklearn import datasets
from part1 import MyNearestNeighborClassifier
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test  = iris_X[indices[-10:]]
iris_y_test  = iris_y[indices[-10:]]

knn = MyNearestNeighborClassifier()
knn.fit(iris_X_train, iris_y_train) 

knn.predict(iris_X_test, iris_y_test)

knn.accuracy(iris_y_test)
print("accuracy=", knn.acc)