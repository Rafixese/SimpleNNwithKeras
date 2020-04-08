#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:44:26 2020

@author: ganja
"""

#%% Imports
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

import matplotlib.pyplot as plt

#%% Dataset prep

# Initialize dataset
dataset = np.array([[0,0,0], 
                    [0,1,0],
                    [1,0,0], 
                    [1,1,1]])

# dataset divide to X and Y

X = dataset[:,:-1]
Y = dataset[:,-1]

#%% Define model structure

model = Sequential()

model.add(Dense(2, activation = 'sigmoid', input_dim = 2))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(SGD(learning_rate = 0.1), loss = 'mse', metrics = ['accuracy'])

#%% Fit model

history = model.fit(X,Y, batch_size=1, epochs = 1000)

#%% learning plots

acc = history.history['accuracy']
loss = history.history['loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label = 'Accuracy')
plt.title('Accuracy of training')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, 'r', label = 'Loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#%% Model test

model.evaluate(X, Y)

y_preds = model.predict(X)
y_preds = y_preds > 0.5