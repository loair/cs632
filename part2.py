
# coding: utf-8

# In[6]:

from os import listdir
import re
import json
import numpy as np
from part1 import MyNearestNeighborClassifier
PATH = "misc-master/spam_data"
files = listdir(PATH)
file_list = []
for f in files:
    if "txt" in f:
        file_list.append(f)
        
with open("bagofwords.txt") as f:
    string = f.read()
    string = string.replace("\n", ",")
    bagofwords = re.split(",", string)


BAG_OF_WORDS = np.array(bagofwords)
file = file_list[50]
y_data = np.loadtxt(PATH + "/" + file).astype(np.int64)


# In[7]:

def count_words(file):
    words = {}
    for k in BAG_OF_WORDS:
        words[k] = 0
    text_string = open(PATH + "/" + file, encoding = "ISO-8859-1").read().lower()
    for word in re.findall(r"[\w']+", text_string):
        if word in BAG_OF_WORDS:
            words[word] = words.get(word) + 1
    return list(words.values())


# In[8]:

X_data = np.array(count_words(file_list[0]))

for i in range(1,50):
    words = count_words(file_list[i])
    X_data = np.row_stack((X_data, words))


# In[9]:

np.random.seed(0)
indices = np.random.permutation(len(X_data))
X_train = X_data[indices[:-30]]
y_train = y_data[indices[:-30]]
X_test = X_data[indices[-10:]]
y_test = y_data[indices[-10:]]

knn = MyNearestNeighborClassifier()
knn.fit(X_train, y_train)
knn.predict(X_test, y_test)


# In[10]:

knn.accuracy(y_test)
print(knn.precision)
print(knn.recall)
print(knn.F)
print(knn.acc)

