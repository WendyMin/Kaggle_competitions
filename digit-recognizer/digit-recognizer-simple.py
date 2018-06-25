# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 21:36:23 2018

@author: minco

Load data
Check data(training and test)(distribution, null ang missing values)
Normalization
Reshape 784->28*28*1
Label encoding (Y_train:2->[0,0,1,0,0,0,0,0,0,0])
Split training and validation set
Visualize the image
Define the model
CNN: In -> [[Conv2D -> relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> [Dense -> softmax] -> Out
Set the optimizer and annealer: RMSprop 深度学习最优方法
Data augmentation + Fit the model
Evaluate the model
Training and validation curves
Confusion matrix
Display error results
Prediction and submission
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

np.random.seed(2)

# Load data
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
Y_train = train["label"]
X_train = train.drop(labels=["label"], axis=1)
del train

# Check data(training and test)(distribution, null ang missing values)
#g = sns.countplot(Y_train)
#Y_train.value_counts()
#X_train.isnull().any().describe()
#test.isnull().any().describe()

# Normalization
X_train = X_train / 255.0
test = test / 255.0

# Reshape 784->28*28*1
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

# Label encoding (Y_train:2->[0,0,1,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes=10)

# Split training and validation set
random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=random_seed)

# Visualize the image
#g = plt.imshow(X_train[0][:,:,0])

# Define the model
# CNN: In -> [[Conv2D -> relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> [Dense -> softmax] -> Out
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(10, activation="softmax"))

# Set the optimizer and annealer: RMSprop 深度学习最优方法
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizer, loss="categorical_crossentropy",metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

# Data augmentation + Fit the model
datagen = ImageDataGenerator(
        featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, 
        samplewise_std_normalization=False, zca_whitening=False,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False, vertical_flip=False
        )
datagen.fit(X_train)
batch_size = 64
epochs = 50 # 30 is totally enough
history = model.fit_generator(
        datagen.flow(X_train,Y_train,batch_size=batch_size),
        epochs=epochs, 
        validation_data=(X_val,Y_val),
#        verbose=2, 
        steps_per_epoch=X_train.shape[0]//batch_size,
        callbacks=[learning_rate_reduction]
        )

# Evaluate the model
# Training and validation curves
#fig, ax = plt.subplots(2,1)
#ax[0].plot(history.history['loss'], color='b', label="Training loss")
#ax[0].plot(history.history['val_loss'], color='r', label="Validation loss", axes=ax[0])
#legend = ax[0].legend(loc='best', shadow=True)

# Confusion matrix
#def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',cmap=plt.cm.Blues):
#    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.title(title)
#    plt.colorbar()
#    tick_marks = np.arange(len(classes))
#    plt.xticks(tick_marks, classes, rotation=45)
#    plt.yticks(tick_marks, classes)
#    if normalize:
#        cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
#    thresh = cm.max()/2.
#    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
#        plt.text(j, i, cm[i,j], horizontalalignment="center", 
#                 color="white" if cm[i,j]>thresh else "black")
#    plt.tight_layout()
#    plt.ylabel('True label')
#    plt.xlabel('Predicted label')
#
#Y_val_pred = model.predict(X_val)
#Y_val_pred_classes = np.argmax(Y_val_pred, axis=1)
#Y_val_true = np.argmax(Y_val, axis=1)
#confusion_mtx = confusion_matrix(Y_val_true, Y_val_pred_classes)
#plot_confusion_matrix(confusion_mtx, classes=range(10))

# Display error results
#errors = (Y_val_pred_classes - Y_val_true != 0)
#Y_val_pred_classes_errors = Y_val_pred_classes[errors]
#Y_val_pred_errors = Y_val_pred[errors]
#Y_val_true_errors = Y_val_true[errors]
#X_val_errors = X_val[errors]
#def display_errors(errors_index, img_errors, pred_errors, obs_errors):
#    n = 0
#    nrows = 2
#    ncols = 3
#    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
#    for row in range(nrows):
#        for col in range(ncols):
#            error = errors_index[n]
#            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
#            ax[row,col].set_title("Predicted label: {}\nTrue label: {}".format(pred_errors[error],obs_errors[error]))
#            n += 1
#Y_val_pred_errors_prob = np.max(Y_val_pred_errors,axis=1)
#val_true_prob_errors = np.diagonal(np.take(Y_val_pred_errors,Y_val_true_errors,axis=1))
#val_delta_pred_true_errors = Y_val_pred_errors_prob - val_true_prob_errors
#val_sorted_delta_errors = np.argsort(val_delta_pred_true_errors)
#val_most_important_errors = val_sorted_delta_errors[-6:]
#display_errors(val_most_important_errors, X_val_errors, Y_val_pred_classes_errors, Y_val_true_errors)

# Prediction and submission
results = model.predict(test)
results = np.argmax(results, axis=1)
submission = pd.DataFrame({"ImageId":list(range(1,len(results)+1)),"Label":results})
#submission = pd.DataFrame({"ImageId":range(len(results)),"Label":results})
submission.to_csv("MNIST_datagen3.csv", index=False)

