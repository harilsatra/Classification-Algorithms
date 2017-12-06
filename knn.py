#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 02:00:28 2017

@author: harilsatra
"""

import numpy as np
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

# Function to calculate the distance between a test sample and all the training samples
def euclidean_dist(test,train,cat_var):
    dist = []
    for i in range(len(train)):
        temp = 0
        for j in range(len(test)):
            if j not in cat_var:
                temp += (test[j]-train[i][j])**2
            else:
                if test[j] != train[i][j]:
                    temp += 1
        dist.append(math.sqrt(temp))
    return dist

#Prompt the user to enter the filename
filename = input("Enter the filename with extension: ")

# Read the file specified by the user
with open(filename) as textFile:
    lines = [line.split('\t') for line in textFile]

no_samples = len(lines)
no_attrs = len(lines[0])-1

#Extract the class labels from the data              
data = np.zeros((no_samples,no_attrs),dtype=float)
class_labels = [int(row[-1].rstrip("\n")) for row in lines]

cat_attrs = set()

# Encode the nominal attributes
for j in range(no_attrs):
    nominal_attr = {}
    nominal_count = 0.0
    if not is_number(lines[0][j]):
        cat_attrs.add(j)
    for i in range(no_samples):
        if is_number(lines[i][j]):
            data[i][j] = float(lines[i][j])
        elif lines[i][j] in nominal_attr:
            data[i][j] = nominal_attr[lines[i][j]]
        else:
            nominal_attr[lines[i][j]] = nominal_count
            data[i][j] = nominal_attr[lines[i][j]]
            nominal_count += 1

k = 10 # K-fold value
nn= 5 # Number of nearest neighbours

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
    test_data = data[start:end][:]
    test_true_labels = class_labels[start:end]
    train_data = np.delete(data,np.s_[start:end],0)
    train_labels = np.delete(class_labels,np.s_[start:end])
    
    #Calculate the mean and standard devitation required to normalize the data
    mean = train_data.mean(0)
    sd = train_data.std(0)
    
    #Normalize the training data based on the mean and sd calculated for each attribute
    normalized_train_data = []
    for i in range(len(train_data)):
        temp_list = []
        for j in range(no_attrs):
            temp_list.append((train_data[i][j]-mean[j])/sd[j])
        normalized_train_data.append(temp_list)

    
    test_new_labels = []
    for test_sample in test_data:
        classes = {}
        normalized_test_sample = []
        for i in range(len(test_sample)):
            normalized_test_sample.append((test_sample[i]-mean[i])/sd[i])
        idx = np.argpartition(euclidean_dist(normalized_test_sample,normalized_train_data,cat_attrs),nn)
        for x in range(nn):
            classes[train_labels[idx[x]]] = classes.get(train_labels[idx[x]],0)+1
        
        # Assign class label to the test sample by counting the class which is most common in the neighbors          
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
    if confusion_matrix[0][0]+confusion_matrix[1][0] != 0:
        precision += confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[1][0])
    if confusion_matrix[0][0]+confusion_matrix[0][1] != 0:
        recall += confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[0][1])
    if (2*confusion_matrix[0][0])+confusion_matrix[0][1]+confusion_matrix[1][0] != 0:
        f1 += (2*confusion_matrix[0][0])/((2*confusion_matrix[0][0])+confusion_matrix[0][1]+confusion_matrix[1][0])
    
    if fold==8:
        start += test_len
        end += test_len+no_samples-(10*test_len)
    else:
        start += test_len
        end += test_len

# Print the evaluation metrics
print("Average Metrics(Cross Validation): ")
print("Accuracy: " ,accuracy/k)
print("Precision: " ,precision/k)
print("Recall: " ,recall/k)
print("F1-measure: ",f1/k)