### Import the necessary libraries ###
from __future__ import division, print_function, absolute_import

from skimage import color, io
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

import os
import sys
from glob import glob
import random

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.normalization import local_response_normalization
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy
import pickle

np.set_printoptions(suppress=True)

########################################
### Imports picture file into the model
########################################

# TumorA
# TumorB
# TumorC
# TumorD
# TumorE

if len(sys.argv) == 1:
    raise SystemError("Usage:\n\npython <this_script> dataset_file.pkl")
f = open(sys.argv[1], 'rb')
print("pickle file open")

## Load from the file for X(image data) and Y(tumor type)
allX, allY = pickle.load(f)
# randomly shuffle
allXY =  list(zip(allX, allY))
random.shuffle(allXY)
allX, allY = zip(*allXY)


print("pickle opened")
f.close()

## image size set to 64x64 for faster computations ##
size_image = 64


###################################
# Define model architecture
###################################

# Input is a 64x64 image with 3 color channel
network = input_data(shape=[None, size_image, size_image, 3])

# 1: Convolution layer with 16 filters, each 5x5
conv_1 = conv_2d(network, nb_filter=16, filter_size=5, activation='relu', name='conv_1')
print("layer 1")

# 2: Max pooling layer
network = max_pool_2d(conv_1, 2)
print("layer 2")

# 3: Convolution layer with 32 filters -> filter size = 3x3
conv_2 = conv_2d(network, nb_filter=32, filter_size=3, activation='relu', name='conv_2')
print("layer 3")

# 6: Max pooling layer
#network = max_pool_2d(conv_2, 2)
#print("layer 3a - optional")

# 4: Convolution layer with 64 filters -> filter size = 3x3
conv_3 = conv_2d(conv_2, nb_filter=64, filter_size=3, activation='relu', name='conv_3')
print("layer 4")

# 6: Max pooling layer
network = max_pool_2d(conv_3, 2)
print("layer 5")

# 7: Fully-connected 512 nodes layer -> activation function = ReLU
network = fully_connected(network, 512, activation='relu')
print("layer 6")

# 8: Dropout layer to combat overfitting
network = dropout(network, 0.6)
print("layer 7")

# 9: Fully-connected layer with 3 outputs for three tumor categories
network = fully_connected(network, 3, activation='softmax')
print("layer 8")


# Regression layer with loss=categorical crossentropy, optimizer=adam, learning rate=0.0001
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0001)

# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose = 0)

print("model created done")


###################################################
# Prepare train & test samples and train the model
###################################################

## Using N-fold cross validation

no_folds = 5

accuracy_array = np.zeros((no_folds), dtype='float64') # accuracies of the test dataset for each split in cross validation
accuracy_array2 = np.zeros((no_folds), dtype='float64') # accuracies for the complete dataset for each split in cross validation

i=0 # counter
split_no = 1 # counter for each split

kf = KFold(n_splits=no_folds, shuffle = True, random_state=42) # create split criteria using KFold in Sklearn.model_selection

#train_splits = []
#test_splits = []

# to plot the metrics when training is done
accur_metrics_train = []
accur_metrics_val = []

###################################
# Train model for N epochs
###################################


# reserve some data (10%) for testing
ind_rand = [random.randint(0, len(allX)-1) for _ in range(int(.1*len(allX)))]
allX_test = [allX[i] for i in ind_rand]
allY_test = [allY[i] for i in ind_rand]
# remove the testing data, keep the rest for training+val
allX = [allX[i] for i in range(len(allX)-1) if i not in ind_rand]
allY = [allY[i] for i in range(len(allY)-1) if i not in ind_rand]
allX = np.array(allX)
allY = np.array(allY)

for train_index, val_index in kf.split(allX):

    # split dataset using kf criteria into train and val dataset
    X, X_val = allX[train_index.astype(int)], allX[val_index.astype(int)]
    Y, Y_val = allY[train_index.astype(int)], allY[val_index.astype(int)]

    # create output labels for whole dataset and val dataset
    Y = to_categorical(Y, 3)
    Y_val = to_categorical(Y_val, 3)

    print("train split: " , split_no)
    split_no += 1 # iterate split no

    # Train the network for N epochs per split (shuffles data)  -> total no of training epochs=60
    model.fit(X, Y, n_epoch=5, run_id='tumour_classification', shuffle=True,
        show_metric=True, validation_set = (X_val, Y_val))

    model.save('model_tumour_detector.tflearn')
    print("Network trained")

    # Calculate accuracies for val dataset and whole dataset in each split run
    score = model.evaluate(X_val, Y_val)
    score2 = model.evaluate(X, Y)

    # populate the accuracy arrays
    accuracy_array[i] = score[0] * 100
    accuracy_array2[i] = score2[0] * 100
    accur_metrics_train.append(score)
    accur_metrics_val.append(score2)
    print(i)
    i += 1 # iterate

    print("accuracy checked")
    print("")
    print("accuracy for test dataset: ", accuracy_array) # print accuracy for the test dataset
    print("")
    print("accuracy for whole dataset: ", accuracy_array2) # print accuracy for the whole dataset

print("done training using %d fold validation" %no_folds)

# Retrieve the maximum accuracy of the accuracy arrays
max_accuracy = accuracy_array[np.argmax(accuracy_array)]
max_accuracy = round(max_accuracy, 3)

max_accuracy2 = accuracy_array2[np.argmax(accuracy_array2)]
max_accuracy2 = round(max_accuracy2, 3)

print("")

###################################################
## Test the model to predict labels ###############
###################################################


#no_iteration = 100
#kf = KFold(n_splits=no_iteration)
#x_splits = kf.split(allX)

# initiate y_label
y_label = 0

# counters
j = 0
k = 0
c = 0
b = 0

###################################################
# Use test data
###################################################
# create Y_true and y_pred np.arrays to save the corresponding label (true label and predicted label) -> labels are shown at the beginning of the program
y_pred = np.zeros((len(allX_test)), dtype='int32')
y_true = np.zeros((len(allY_test)), dtype='int32')

# split allX and allY into 90 sections
x_list = np.array_split(allX_test, 90)
y_list = np.array_split(allY_test, 90)

i = 0

for j in x_list:

    # get the (i)th section from x_list and y_list to x_test and y_test (arrays renew for each j)
    x_test = x_list[i]
    y_test = y_list[i]

    # y_label=predict results for the (i)th section in x_test
    y_label = model.predict(x_test)

    b = 0 # b is reset in each (j)th iteration
    for k in y_label:
        y_pred[c] = np.argmax(y_label[b]) # get the index of the maximum probability (prediction) for (b)th array in y_label
        y_true[c] = y_test[b] # (b)th element is copied to y_true array
        c += 1
        b += 1
    i += 1


#print("j is", j, "k is ", k, " splits are ", kf.split(allX))

##################################
# Test prints ####################
##################################

print("Prediction finished", c)
print("")
print(len(y_true), " bla bla ", len(y_pred))
print("")

print("calculate accuracy")
acc_final = accuracy_score(y_true, y_pred)
print(acc_final, "\n")

print("calculate f1 score")
f1Score = f1_score(y_true, y_pred, average=None)
print(f1Score, "\n")

print("calculate precision:")
prec = precision_score(y_true, y_pred, average=None)
print(prec, "\n")

print("calculate recall")
recall = recall_score(y_true, y_pred, average=None)
print(recall, "\n")

print("calculate confusion matrix")
confusionMatrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
print("confusion Matrix Created")
print(confusionMatrix)


##################################
## Print the Results #############
##################################

print("")
print("")
print ("-----------------------------------------------------------------------------")
print ( "    Cancer Tumor detector using Convolutional Neural Networks - 3-Fold cross validation") 
print ("Author - Narmada Balasooriya")
print ("Editor - Leontios Mavropalias")
print ("-----------------------------------------------------------------------------")
print("")
print("accuracy for the test dataset")
print(accuracy_array)
print("")
print("accuracy for the whole dataset")
print(accuracy_array2)
print("")
print("Maximum accuracy for test dataset: ", max_accuracy, '%')
print("")
print("Maximum accuracy for whole dataset: ", max_accuracy2, '%')
print("")
print("F1, precision, recall for the whole dataset")
print(f1Score, prec, recall)
print("")
print("confusion Matrix")
print(confusionMatrix)
print("")
print(classification_report(y_pred,y_true))
print ("-----------------------------------------------------------------------------")
