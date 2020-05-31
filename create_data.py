## Author: Narmada M. Balasooriya   ##
##         University of Peradeniya ##
##         Sri Lanka                ##

#####################################
## Import the necessary libraries ###
#####################################

from __future__ import division, print_function, absolute_import

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

# TumorA = 
# TumorB = 
# TumorC = 
# TumorD = 
# TumorE = 

# TODO: rename: these are actually 5 tumour types
files_path_tumorA = '/mnt/share/mri_dataset/cheng/full/1'
files_path_tumorB = '/mnt/share/mri_dataset/cheng/full/2'
files_path_tumorC = '/mnt/share/mri_dataset/cheng/full/3'
files_path_healthy = '/mnt/share/mri_dataset/cheng/full/4'
files_path_tumor_unknown = '/mnt/share/mri_dataset/cheng/full/5'

tumorA_path = os.path.join(files_path_tumorA, '*.tiff')
tumorB_path = os.path.join(files_path_tumorB, '*.tiff')
tumorC_path = os.path.join(files_path_tumorC, '*.tiff')
no_tumor_path = os.path.join(files_path_healthy, '*.tiff')
tumor_unknown_path = os.path.join(files_path_tumor_unknown, '*.tiff')

print("tumor A path")

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
        img = cv2.imread(f)
        new_img = cv2.resize(img,(size_image,size_image))
        allX[count] = np.array(new_img)
        ally[y_count] = 0
        count += 1
        y_count += 1
    except:
        raise SystemError("Uh-oh, something wrong with label A...")

print("tumorA done")
for f in tumorB:
    try:
        img = cv2.imread(f)
        new_img = cv2.resize(img,(size_image,size_image))
        allX[count] = np.array(new_img)
        ally[y_count] = 1
        count += 1
        y_count += 1
    except:
        raise SystemError("Uh-oh, something wrong with label B...")

print("tumorB done")
for f in tumorC:
    try:
        img = cv2.imread(f)
        new_img = cv2.resize(img,(size_image,size_image))
        allX[count] = np.array(new_img)
        ally[y_count] = 2
        count += 1
        y_count += 1
    except:
        raise SystemError("Uh-oh, something wrong with label C...")

print("tumorC done")
for f in no_tumor:
    try:
        img = cv2.imread(f)
        new_img = cv2.resize(img,(size_image,size_image))
        allX[count] = np.array(new_img)
        ally[y_count] = 3
        count += 1
        y_count += 1
    except:
        raise SystemError("Uh-oh, something wrong with label D...")

print("no tumor done")
for f in tumor_unknown:
    try:
        img = cv2.imread(f)
        new_img = cv2.resize(img,(size_image,size_image))
        allX[count] = np.array(new_img)
        ally[y_count] = 4
        count += 1
        y_count += 1
    except:
        raise SystemError("Uh-oh, something wrong with label E...")

print("unknown done")
print("images are arrayed")


f = open('full_dataset_final.pkl', 'wb')

print("pickle file open")
cPickle.dump((allX, ally), f, protocol=cPickle.HIGHEST_PROTOCOL)
print("pickle dumped")
f.close()

print("finished")
