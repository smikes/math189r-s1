"""
Start file for hw1pr2 of Big Data Summer 2017

This file is used to load and process data for problem 2, and is imported

Note:
1. Please DO NOT change anything in this file.
2. Although these codes are given, please read carefully through the file, and
   make sure you understand how these codes work.
"""
import pandas as pd
import numpy as np

print('==>Loading data...')

# create the headers for data frame since original data dodes not have headers
# read the training data
df_train = pd.read_csv('mnist_train.csv')

# read the test data
df_test = pd.read_csv('mnist_test.csv')

print('==>Data loaded succesfully.')

# Process training data, so that X (pixels) and y (labels) are seperated
X_train = np.array(df_train.iloc[:,1:]) / 256.

y_train = np.array(df_train[:][['label']])


# Process test data, so that X (pixels) and y (labels) are seperated
X_test = np.array(df_test.iloc[:,1:]) / 256.

y_test = np.array(df_test[:][['label']])
