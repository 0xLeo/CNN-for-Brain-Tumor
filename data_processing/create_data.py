#!/usr/bin/env python
## Author: Narmada M. Balasooriya   ##
##         University of Peradeniya ##
##         Sri Lanka                ##
## Editor: Leontios Mavropalias     ##
##         UoC                      ##

#####################################
## Import the necessary libraries ###
#####################################

from __future__ import division, print_function, absolute_import

from skimage import color, io
import numpy as np
import os
from glob import glob
import cv2

from six.moves import cPickle
import pickle


np.set_printoptions(suppress=True)
########################################
### Imports picture files
########################################

# TumorA = 0
# TumorB = 1
# TumorC = 2

###vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
### Change below for your dataset
files_path_tumorA = '/mnt/share/mri_dataset/cheng/lbl_orig/1/'
files_path_tumorB = '/mnt/share/mri_dataset/cheng/lbl_orig/2/'
files_path_tumorC = '/mnt/share/mri_dataset/cheng/lbl_orig/3/'

tumorA_path = os.path.join(files_path_tumorA, '*.tiff')
tumorB_path = os.path.join(files_path_tumorB, '*.tiff')
tumorC_path = os.path.join(files_path_tumorC, '*.tiff')
###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

print("tumor A path")

# BUG: only tumor A is fuled, the rest empty
tumorA = sorted(glob(tumorA_path))
tumorB = sorted(glob(tumorB_path))
tumorC = sorted(glob(tumorC_path))

n_files = len(tumorA) + len(tumorB) + len(tumorC) 
print("######print here")
print(n_files)
print("##########")

size_image = 64

allX = np.zeros((n_files, size_image, size_image, 3), dtype='uint8')
ally = np.zeros((n_files), dtype='int32')
count = 0
y_count = 0
for f in tumorA:
    try:
        #img = io.imread(f)
        img = cv2.imread(f)
        new_img = cv2.resize(img,(size_image,size_image))
        allX[count] = np.array(new_img)
        ally[y_count] = 0
        count += 1
        y_count += 1
    except exception as e:
        print("Tumour A not set")
        raise e

print("tumorA done")
for f in tumorB:
    try:
        #img = io.imread(f)
        img = cv2.imread(f)
        new_img = cv2.resize(img,(size_image,size_image))
        allX[count] = np.array(new_img)
        ally[y_count] = 1
        count += 1
        y_count += 1
    except exception as e:
        print("Tumour B not set")
        raise e
print("tumorB done")

for f in tumorC:
    try:
        #img = io.imread(f)
        img = cv2.imread(f)
        new_img = cv2.resize(img,(size_image,size_image))
        allX[count] = np.array(new_img)
        ally[y_count] = 2
        count += 1
        y_count += 1
    except exception as e:
        print("Tumour C not set")
        raise e

print("tumorC done")
print("data are split")

pkl_fname = 'dataset_cheng_full_nonseg_lbl.pkl'
f = open(pkl_fname, 'wb')

print("pickle file open")
cPickle.dump((allX, ally), f, protocol=cPickle.HIGHEST_PROTOCOL)
print("pickle dumped at:\n%s" % pkl_fname)
f.close()

print("finished")
