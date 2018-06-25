# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 21:54:05 2018

@author: minco
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
Y_train = train["label"]
X_train = train.drop(labels=["label"], axis=1)
del train

# Split training and validation set
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

# Model: MLPClassifier
#0.948333333333
model = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(50,50,50), random_state=1)
#print(model)
# MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#       beta_2=0.999, early_stopping=False, epsilon=1e-08,
#       hidden_layer_sizes=(50, 50, 50), learning_rate='constant',
#       learning_rate_init=0.001, max_iter=200, momentum=0.9,
#       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
#       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
#       warm_start=False)
model.fit(X_train, Y_train)
Y_val_pred = model.predict(X_val)
print(accuracy_score(Y_val,Y_val_pred))

# Prediction and Submission
#results = model.predict(test)
#submission = pd.DataFrame({"ImageId":list(range(1,len(results)+1)),"Label":results})
#submission.to_csv("MNIST_datagen4.csv", index=False)