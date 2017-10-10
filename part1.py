
# coding: utf-8

# In[1]:

import math
import numpy as np
import operator
class MyNearestNeighborClassifier:
    def __init__(self, n_neighbor=3):
        self.n_neighbor = n_neighbor
        self.presion = 0
        self.recall = 0
        self.F = 0
    
    def fit(self, X_train, y_train):
        if self.n_neighbor < 0:
            print("K should be larger than 0")
            return None
        self.X_train = X_train
        self.y_train = y_train
        
    def distance(self, x, y):
        length = len(x)
        dis = 0
        for i in range(length):
            dis = (x[i] - y[i])**2 + dis
        return math.sqrt(dis)
    
    def accuracy(self, y_test):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        count = [0, 0]
        for i in range(len(self.predictions)):
            if self.predictions[i] == y_test[i]:
                count[0] = count[0] + 1
                if y_test[i] == 0:
                    tp = tp + 1
                else:
                    tn = tn + 1
            else:
                count[1] = count[1] + 1
                if y_test[i] == 0:
                    fn = fn + 1
                else:
                    fp = fp + 1
        self.precision = tp / (tp + fp)
        self.recall = tp / (tp + fn)
        self.F = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        self.acc = count[0]/(count[0] + count[1])
        
    def predict(self, X_test, y_test):
        self.predictions = []
        for i in range(len(X_test)):
            dis = []
            for j in range(len(self.X_train)):
                dis.append(self.distance(self.X_train[j], X_test[i]))
            distances = np.array(dis)
            sorteddistances = distances.argsort()
            vote = {}
            for n in range(self.n_neighbor):
                nth_label = self.y_train[sorteddistances[n]]
                vote[nth_label] = vote.get(nth_label, 0)+1
            sortedvote = sorted(vote.items(),key = operator.itemgetter(1), reverse = True)  
            self.predictions.append(sortedvote[0][0])
            
        return np.array(self.predictions)

