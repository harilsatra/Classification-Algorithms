#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 19:42:02 2017

@author: harilsatra
"""
import numpy as np
from scipy.stats import norm
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
#filename = input("Enter the filename with extension: ")

# Read the file specified by the user
with open('project3_dataset2.txt') as textFile:
    lines = [line.split('\t') for line in textFile]

no_samples = len(lines)
no_attrs = len(lines[0])
              
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

k = 10 # K-fold value

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
    
    # Extract the test data and training data
    test_data = data[start:end][:]
    test_true_labels = class_labels[start:end]
    train_data = np.delete(data,np.s_[start:end],0)
    train_labels = np.delete(class_labels,np.s_[start:end])
    
    temp_0 = train_data[train_data[:,-1]==0,:]
    temp_1 = train_data[train_data[:,-1]==1,:]
    
    prior_0 = len(temp_0)/len(train_data)
    prior_1 = len(temp_1)/len(train_data)
    
    mean_0 = temp_0.mean(0)
    mean_1 = temp_1.mean(0)
    
    std_0 = temp_0.std(0,ddof=1)
    std_1 = temp_1.std(0,ddof=1)
    
    cat_count_0 = {}
    cat_count_1 = {}
    
    for attr_index in cat_attrs:
        temp_list_0 = np.unique(temp_0[:,attr_index],return_counts=True)
        temp_dict_0 = dict(zip(temp_list_0[0],temp_list_0[1]))
        cat_count_0[attr_index] = temp_dict_0
        
        temp_list_1 = np.unique(temp_1[:,attr_index],return_counts=True)
        temp_dict_1 = dict(zip(temp_list_1[0],temp_list_1[1]))
        cat_count_1[attr_index] = temp_dict_1
    
    test_new_labels = []
    for i in range(0,len(test_data)):
        posterior_0 = norm(mean_0,std_0).pdf(test_data[i])
        posterior_1 = norm(mean_1,std_1).pdf(test_data[i])
        post_prob_0 = 1
        post_prob_1 = 1
        for j in range(0,no_attrs-1):
            if j in cat_attrs:
                prob_0 = cat_count_0[j][test_data[i][j]]/len(temp_0)
                prob_1 = cat_count_1[j][test_data[i][j]]/len(temp_1)
                post_prob_0 *= prob_0
                post_prob_1 *= prob_1
                
            else:
                post_prob_0 *= posterior_0[j]
                post_prob_1 *= posterior_1[j]
         
        final_prob_0 = prior_0 * post_prob_0
        final_prob_1 = prior_1 * post_prob_1
        if final_prob_0 > final_prob_1:
            test_new_labels.append(0)
        else:
            test_new_labels.append(1)
                
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
    
    if fold==8:
        start += test_len
        end += test_len+no_samples-(10*test_len)
    else:
        start += test_len
        end += test_len

print("Accuracy: " ,accuracy/k)
print("Precision: " ,precision/k)
print("Recall: " ,recall/k)
print("F1-measure: ",f1/k)
