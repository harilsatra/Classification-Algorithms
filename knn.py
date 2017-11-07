#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 02:00:28 2017

@author: harilsatra
"""

import numpy as np
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances

#http://pythoncentral.io/how-to-check-if-a-string-is-a-number-in-python-including-unicode/
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    
    return False

#filename = input("Enter the filename with extension: ")
with open('project3_dataset2.txt') as textFile:
    lines = [line.split('\t') for line in textFile]

no_samples = len(lines)
no_attrs = len(lines[0])-1
              
data = np.zeros((no_samples,no_attrs),dtype=float)
class_labels = [int(row[-1].rstrip("\n")) for row in lines]

# Handle the nominal attributes
for j in range(no_attrs):
    nominal_attr = {}
    nominal_count = 0
    for i in range(no_samples):
        if is_number(lines[i][j]):
            data[i][j] = float(lines[i][j])
        elif lines[i][j] in nominal_attr:
            data[i][j] = nominal_attr[lines[i][j]]
        else:
            nominal_attr[lines[i][j]] = nominal_count
            nominal_count += 1

#http://scikit-learn.org/stable/modules/preprocessing.html
min_max_scaler = preprocessing.MinMaxScaler()
normalized_data = min_max_scaler.fit_transform(data)

#print(normalized_data)
#print(len(class_labels))

k = 10
nn= 4

test_len = int(no_samples/k)
train_len = no_samples - test_len
start = 0
end = test_len

test_data = normalized_data[start:end][:]
test_true_labels = class_labels[start:end]
train_data = np.delete(normalized_data,np.s_[start:end],0)
train_labels = np.delete(class_labels,np.s_[start:end])

distances = euclidean_distances(test_data,train_data)
test_new_labels = []
for dist in distances:
    classes = {}
    idx = np.argpartition(dist,nn)
    print([train_labels[temp] for temp in idx[0:nn]])
    for x in range(0,nn):
        classes[train_labels[idx[x]]] = classes.get(train_labels[idx[x]],0)+1

    print(str(classes))
    election = 0
    label = -1
    for key in classes.keys():
        if classes[key] > election:
            election = classes[key]
            label = key
    test_new_labels.append(label)

print("TRUE LABELS: ")
print(test_true_labels)
print("NEW LABELS: ")
print(test_new_labels)
