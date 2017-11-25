#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 02:00:28 2017

@author: harilsatra
"""

import numpy as np
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
import math

#http://pythoncentral.io/how-to-check-if-a-string-is-a-number-in-python-including-unicode/
# Function to check if a string is a number or not
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

# Function to calculate the distance between two samples
def euclidean_dist(mat,cat_var):
    no_samples = len(mat)
    dist_mat = np.zeros((no_samples,no_samples))
    for i in range(0,no_samples):
        for j in range(i,no_samples):
            sum = 0
            if i==j:
                dist_mat[i][j] = 0
            else:
                for k in range(len(mat[0])):
                    if k not in cat_var:
                        sum += (mat[i][k] - mat[j][k])**2
                    else:
                        if mat[i][k] == mat[j][k]:
                            sum += 1
                dist_mat[i][j] = dist_mat[j][i] = math.sqrt(sum)
    return dist_mat

# Prompt the user to enter the filename
filename = input("Enter the filename with extension: ")

# Read the file specified by the user
with open(filename) as textFile:
    lines = [line.split('\t') for line in textFile]

no_samples = len(lines)
no_attrs = len(lines[0])-1
              
data = np.zeros((no_samples,no_attrs),dtype=float)
class_labels = [int(row[-1].rstrip("\n")) for row in lines]

cat_attrs = set()

# Encode the nominal attributes
for j in range(no_attrs):
    nominal_attr = {}
    nominal_count = 0
    if not is_number(lines[0][j]):
        cat_attrs.add(j)
    for i in range(no_samples):
        if is_number(lines[i][j]):
            data[i][j] = float(lines[i][j])
        elif lines[i][j] in nominal_attr:
            data[i][j] = nominal_attr[lines[i][j]]
        else:
            nominal_attr[lines[i][j]] = nominal_count
            nominal_count += 1

#http://scikit-learn.org/stable/modules/preprocessing.html
# Normalize the data so that certain attributes do not dictate the distance
min_max_scaler = preprocessing.MinMaxScaler()
normalized_data = min_max_scaler.fit_transform(data)

# Obtain the distance matrix
dist_mat = euclidean_dist(normalized_data,cat_attrs)

k = 10 # K-fold value
nn= 4 # Number of nearest neighbours

# Initialize the evaluation metrics
accuracy = 0
precision = 0
recall = 0
f1 = 0

test_len = int(no_samples/k)
train_len = no_samples - test_len
start = 0
end = test_len

# 10-fold Cross Validation
for fold in range(k):
    
    # Extract the test data and training data
    test_data = normalized_data[start:end][:]
    test_true_labels = class_labels[start:end]
    train_data = np.delete(normalized_data,np.s_[start:end],0)
    train_labels = np.delete(class_labels,np.s_[start:end])
    
    distances = dist_mat[start:end][:]
    distances = np.delete(distances,np.s_[start:end],axis=1)
    test_new_labels = []
    for dist in distances:
        classes = {}
        idx = np.argpartition(dist,nn)
        #print([train_labels[temp] for temp in idx[0:nn]])
        for x in range(0,nn):
            classes[train_labels[idx[x]]] = classes.get(train_labels[idx[x]],0)+1
    
        #print(str(classes))
        election = 0
        label = -1
        for key in classes.keys():
            if classes[key] > election:
                election = classes[key]
                label = key
        test_new_labels.append(label)
    
    # Populate the Confusion matrix in order to calculate the evaluation metrics.
    confusion_matrix = np.zeros((2,2))
    for i in range(len(test_true_labels)):
        if test_true_labels[i]==test_new_labels[i]:
            if test_true_labels[i]==1:
                confusion_matrix[0][0] += 1
            else:
                confusion_matrix[1][1] += 1
        else:
            if test_true_labels[i]==1:
                confusion_matrix[0][1] += 1
            else:
                confusion_matrix[1][0] += 1
    
    # Calculate the evaluation metric for kth fold using the confusion matrix.
    accuracy += (confusion_matrix[0][0]+confusion_matrix[1][1])/(confusion_matrix[1][0]+confusion_matrix[0][1]+confusion_matrix[0][0]+confusion_matrix[1][1])
    precision += confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[1][0])
    recall += confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[0][1])
    f1 += (2*confusion_matrix[0][0])/((2*confusion_matrix[0][0])+confusion_matrix[0][1]+confusion_matrix[1][0])
    
    start += test_len
    end += test_len


    #print("TRUE LABELS: ")
    #print(test_true_labels)
    #print("NEW LABELS: ")
    #print(test_new_labels)

print("Accuracy: " ,accuracy/k)
print("Precision: " ,precision/k)
print("Recall: " ,recall/k)
print("F1-measure: ",f1/k)