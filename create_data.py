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
from scipy.misc import imresize
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
# TumorD = 3
# TumorE = 4

files_path_tumorA = '/mnt/share/mri_dataset/cheng/non_seg/1'
files_path_tumorB = '/mnt/share/mri_dataset/cheng/non_seg/2'
files_path_tumorC = '/mnt/share/mri_dataset/cheng/non_seg/3'
files_path_healthy = '/mnt/share/mri_dataset/cheng/non_seg/4'
files_path_tumor_unknown = '/mnt/share/mri_dataset/cheng/non_seg/5'

tumorA_path = os.path.join(files_path_tumorA, '*.tiff')
tumorB_path = os.path.join(files_path_tumorB, '*.tiff')
tumorC_path = os.path.join(files_path_tumorC, '*.tiff')
no_tumor_path = os.path.join(files_path_healthy, '*.tiff')
tumor_unknown_path = os.path.join(files_path_tumor_unknown, '*.tiff')

print("tumor A path")

# BUG: only tumor A is fuled, the rest empty
tumorA = sorted(glob(tumorA_path))
tumorB = sorted(glob(tumorB_path))
tumorC = sorted(glob(tumorC_path))
no_tumor = sorted(glob(no_tumor_path))
tumor_unknown = sorted(glob(tumor_unknown_path))

n_files = len(tumorA) + len(tumorB) + len(tumorC) + len(no_tumor) + len(tumor_unknown)
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
        new_img = imresize(img, (size_image, size_image, 3))
        allX[count] = np.array(new_img)
        ally[y_count] = 0
        count += 1
        y_count += 1
    except:
        continue

print("tumorA done")
for f in tumorB:
    try:
        #img = io.imread(f)
        img = cv2.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
        allX[count] = np.array(new_img)
        ally[y_count] = 1
        count += 1
        y_count += 1
    except:
        continue
print("tumorB done")
for f in tumorC:
    try:
        #img = io.imread(f)
        img = cv2.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
        allX[count] = np.array(new_img)
        ally[y_count] = 2
        count += 1
        y_count += 1
    except:
        continue
print("tumorC done")
for f in no_tumor:
    try:
        #img = io.imread(f)
        img = cv2.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
        allX[count] = np.array(new_img)
        ally[y_count] = 3
        count += 1
        y_count += 1
    except:
        continue
print("tumorD done")
for f in tumor_unknown:
    try:
        #img = io.imread(f)
        img = cv2.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
        allX[count] = np.array(new_img)
        ally[y_count] = 4
        count += 1
        y_count += 1
    except:
        continue
print("tumorE done")
print("images are arrayed")

print("data are split")

f = open('dataset_cheng_non_seg.pkl', 'wb')

print("pickle file open")
cPickle.dump((allX, ally), f, protocol=cPickle.HIGHEST_PROTOCOL)
print("pickle dumped")
f.close()

print("finished")
