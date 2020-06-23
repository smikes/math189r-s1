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

# read the training data
df_train = pd.read_csv('mnist_train.csv', sep=',', engine='python')

# read the test data
df_test = pd.read_csv('mnist_test.csv', sep=',', engine='python')

print('==>Data loaded succesfully.')


# Process training data, so that X (pixels) and y (labels) are seperated
X_train = df_train[:][0,2:0] / 256
X_train.to_csv('mnist_train_normX.csv')

y_train = np.array(df_train[:][['label']])
y_train.to_csv('mnist_train_normY.csv')


# Process test data, so that X (pixels) and y (labels) are seperated
X_test = df_test[:][0,2:0] / 256
X_test.to_csv('mnist_test_normX.csv')

y_test = df_test[:][['label']]
y_test.to_csv('mnist_test_normY.csv')
