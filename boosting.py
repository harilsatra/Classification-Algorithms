#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 16:49:15 2017

@author: harilsatra
"""

# Import libraries and inbuilt functions
import numpy as np
from collections import Counter
import sys
import random
import math

# Node class
class Node:
    def __init__(self, split_index = None, split_value = None, splits = None, label = None):
        self.left = None
        self.right = None
        self.split_index = split_index
        self.split_value = split_value
        self.splits = splits
        self.label = label

# Function to predict the label for a test sample.
def classify(node, test_sample, cat_attrs):
    if node.split_index in cat_attrs:
        # Decide the direction in which to move
        if test_sample[node.split_index] == node.split_value:
            # If a leaf node is reached, return the label of the leaf node else keep traversing the tree recursively.
            if node.left.label is not None:
                return node.left.label
            else:
                return classify(node.left, test_sample, cat_attrs)
            
        else:
            if node.right.label is not None:
                return node.right.label
            else:
                return classify(node.right, test_sample, cat_attrs)
    else:
        # Decide the direction in which to move
        if test_sample[node.split_index] < node.split_value:
            # If a leaf node is reached, return the label of the leaf node else keep traversing the tree recursively.
            if node.left.label is not None:
                return node.left.label
            else:
                return classify(node.left, test_sample, cat_attrs)
            
        else:
            if node.right.label is not None:
                return node.right.label
            else:
                return classify(node.right, test_sample, cat_attrs)

# Split the node and keep building the tree recursively or stop if it's a leaf node.
def splitNode(node, max_depth, min_size, current_depth, cat_attrs):
    left, right = node.splits
    if len(left) == 0:
        node.left = node.right = leafNode(right)
        return
    if len(right) == 0:
        node.left = node.right = leafNode(left)
        return
    if current_depth >= max_depth:
        node.left = leafNode(left)
        node.right = leafNode(right)
        return
    if len(left) <= min_size:
        node.left = leafNode(left)
    else:
       node.left = bestSplit(left, cat_attrs)
       splitNode(node.left, max_depth, min_size, current_depth + 1, cat_attrs)
       
    if len(right) <= min_size:
        node.right = leafNode(right)
    else:
       node.right = bestSplit(right, cat_attrs)
       splitNode(node.right, max_depth, min_size, current_depth + 1, cat_attrs)
    
# Create a new Node which indicates that it's a leaf node along with the associated label.
def leafNode(data):
    most_common = Counter([i[-1] for i in data]).most_common(1)
    node = Node(label = most_common[0][0])
    return node

# Function to start building the tree
def decisionTree(data, max_depth, min_size, cat_attrs):
    node = bestSplit(data, cat_attrs)
    splitNode(node, max_depth, min_size, 1, cat_attrs)
    return node


# Function to find where would the best split occur in the data provided
def bestSplit(data, cat_attrs):
    no_attrs = len(data[0])
    no_samples = len(data)
    min_gini = sys.maxsize
    split_index = sys.maxsize
    split_value = sys.maxsize
    
    #Iterate each and every value and calculate the gini index for each one and choose the one with minimum gini index
    for j in range(no_attrs-1):
        if j not in cat_attrs:
        
            visited_values = set()
            for i in range(no_samples):
                if data[i][j] not in visited_values:
                    visited_values.add(data[i][j])
                    temp_gini = calGini(j,data[i][j],data,False)
                    if temp_gini <= min_gini:
                        min_gini = temp_gini
                        split_index = j
                        split_value = data[i][j]
                        #print(min_gini, split_value, split_index)

        else:
            visited_values = set()
            for i in range(no_samples):
                if data[i][j] not in visited_values:
                    visited_values.add(data[i][j])
                    temp_gini = calGini(j,data[i][j],data,True)
                    if temp_gini <= min_gini:
                        min_gini = temp_gini
                        split_index = j
                        split_value = data[i][j]
    
    # Split the data based on the selected attribute        
    if split_index in cat_attrs:
        splits = splitData(split_index, split_value, data, True)
    else: 
        splits = splitData(split_index, split_value, data, False)
    #print(split_value)
    #print(split_index)
    node = Node(split_index, split_value, splits)
    #print(min_gini, split_value, split_index)
    return node
    
    
# Split the data based on the split index and value provided    
def splitData(split_index, split_value, data, isCat):
    split1 = []
    split2 = []
    if isCat:
        for i in range(len(data)):
            if data[i][split_index] == split_value:
                split1.append(data[i])
            else:
                split2.append(data[i])
    else:
        for i in range(len(data)):
            if data[i][split_index] < split_value:
                split1.append(data[i])
            else:
                split2.append(data[i])
    return split1, split2
           
            
# Function to calculate the gini index
def calGini(col,split_value,data, isCat):
    split1_counts = [0,0]
    split2_counts = [0,0]
    gini_val = 0.0
    score1 = 0.0
    score2 = 0.0
    if isCat:
        for i in range(len(data)):
            if data[i][col] == split_value:
                split1_counts[int(data[i][-1])] += 1
            else:
                split2_counts[int(data[i][-1])] += 1
    else:
        for i in range(len(data)):
            if data[i][col] < split_value:
                split1_counts[int(data[i][-1])] += 1
            else:
                split2_counts[int(data[i][-1])] += 1
    
    if (split1_counts[0] + split1_counts[1]) != 0:
        for i in range(len(split1_counts)):
            score1 += (split1_counts[i] / (split1_counts[0] + split1_counts[1]))**2
        gini_val += (1.0 - score1) * ((split1_counts[0] + split1_counts[1]) / len(data))    
    if (split2_counts[0] + split2_counts[1]) != 0:
        for i in range(len(split2_counts)):
            score2 += (split2_counts[i] / (split2_counts[0] + split2_counts[1]))**2     
        gini_val += (1.0 - score2) * ((split2_counts[0] + split2_counts[1]) / len(data))    
    #print(gini_val, col, split_value)
    
    return gini_val


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

# Prompt the user to enter the filename
filename = input("Enter the filename with extension: ")

# Read the file specified by the user
with open(filename) as textFile:
    lines = [line.split('\t') for line in textFile]


classes=np.array
no_samples = len(lines)
no_attrs = len(lines[0])
              
data = np.zeros((no_samples,no_attrs),dtype=float)
class_labels = [int(row[-1].rstrip("\n")) for row in lines]
classes=np.unique(class_labels)
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
            data[i][j] = nominal_attr[lines[i][j]]
            nominal_count += 1

k = 10 # K-fold value
max_depth = 100
min_size = -1
T = 5

# Initialize the evaluation metrics
accuracy = 0
precision = 0
recall = 0
f1 = 0

test_len = int(no_samples/10)
train_len = no_samples - test_len
start = 0
end = test_len

# 10-fold Cross Validation
for fold in range(k):
    
    print("FOLD: ", fold+1)
    
    # Extract the test data and training data
    test_data = data[start:end][:]
    test_true_labels = class_labels[start:end]
    train_data = np.delete(data,np.s_[start:end],0)
    train_labels = np.delete(class_labels,np.s_[start:end])
    
    # Initialize the weights
    weights = [1/len(train_data)]*len(train_data)
    test_new_labels = []
    roots = []
    stages = []
    indexes = [i for i in range(0,len(train_data))]

    while len(roots)<T:
        sample_data = []
        weights = [float(i)/sum(weights) for i in weights]
        for index in range(len(train_data)):
            sample_data.append(train_data[random.choices(indexes,weights)].tolist()[0])
        root = decisionTree(sample_data, max_depth, min_size, cat_attrs)
        temp_labels = []
        for train_sample in train_data:
            temp_labels.append(classify(root,train_sample,cat_attrs))
        numerator = 0
        denominator = 0
        misclassified = []
        for j in range(len(train_data)):
            if train_labels[j] != temp_labels[j]:
                misclassified.append(j)
                numerator += weights[j]
            denominator += weights[j]
        error = numerator/denominator
        if error <= 0.5:
            stage = math.log((1-error)/error)*0.5
            stages.append(stage)
            for j in range(len(train_data)):
                if j in misclassified:
                    weights[j] = weights[j]*math.exp(stage)
                else:
                    weights[j] = weights[j]*math.exp(0)
            weights = [float(i)/sum(weights) for i in weights]
            roots.append(root)

        
    # Classify all the test samples        
    for test_sample in test_data:
        labels = [0,0]
        counts = [0,0]
        # Aggregrate the weights for each class based on what is predicted by the decision trees
        for i in range(len(roots)):
            label = int(classify(roots[i],test_sample,cat_attrs))
            labels[label] += stages[i]
            counts[label] += 1
        if counts[0] == 0:
            test_new_labels.append(1.0)
        elif counts[1] == 0:
            test_new_labels.append(0.0)
        elif labels[0] > labels[1]:
            test_new_labels.append(0.0)
        else:
            test_new_labels.append(1.0)


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
    accuracy_k = (confusion_matrix[0][0]+confusion_matrix[1][1])/(confusion_matrix[1][0]+confusion_matrix[0][1]+confusion_matrix[0][0]+confusion_matrix[1][1])
    accuracy += accuracy_k 
    if confusion_matrix[0][0]+confusion_matrix[1][0] != 0:
        precision_k = confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[1][0])
        precision += precision_k
    if confusion_matrix[0][0]+confusion_matrix[0][1] != 0:
        recall_k = confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[0][1])
        recall += recall_k
    if (2*confusion_matrix[0][0])+confusion_matrix[0][1]+confusion_matrix[1][0] != 0:
        f1_k = (2*confusion_matrix[0][0])/((2*confusion_matrix[0][0])+confusion_matrix[0][1]+confusion_matrix[1][0])
        f1 += f1_k
    
    if fold==8:
        start += test_len
        end += test_len+no_samples-(10*test_len)
    else:
        start += test_len
        end += test_len
    
    print("Accuracy: " ,accuracy_k)
    print("Precision: " ,precision_k)
    print("Recall: " ,recall_k)
    print("F1-measure: ",f1_k)
    print()

# Print the evaluation metrics
print("Average Metrics: ")
print("Accuracy: " ,accuracy/k)
print("Precision: " ,precision/k)
print("Recall: " ,recall/k)
print("F1-measure: ",f1/k)