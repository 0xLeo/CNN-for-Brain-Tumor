#!/usr/bin/env python

import cv2
import numpy as np
from k_means_cv import kmeans_wrapper as km
import sys
import os

if __name__ == '__main__':
    if len(sys.argv) == 2:
        im = cv2.imread(sys.argv[1])
        fname = sys.argv[1].split(os.sep)[-1]
        fname_no_ext = fname.split('.')[0]
    else:
        raise SystemError("Usage:\n\npython <this_script> image_file_to_segment")
im = cv2.GaussianBlur(im, (13,13), 2.5, sigmaY = 2.5)
strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, strel)
im = cv2.morphologyEx(im, cv2.MORPH_OPEN, strel)
im, _ = km(im, 4, True)
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

cv2.imwrite("%04d.png" % int(fname_no_ext), im)
#cv2.imshow("",imBW)
#cv2.waitKey(20000)
#cv2.destroyAllWindows()
