'''
Image preprocessing function for the Single Shot Corrective CNN (SSC-CNN)

This function crops the training images to the same size image to be used for training.
It uses the labels file and cropping the hand in that region.

Copyright (c) 2021 Indian Institute of Technology Madras

Permission is hereby granted, free of charge, to any person obtaining a copy of this software 
and associated documentation files (the "Software"), to deal in the Software without restriction, 
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

# Import libraries
import csv
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from utils import world2pixel, pixel2world, fx, fy, u0, v0, depth_thres, cropSize
from smallestenclosingcircle import make_circle


################################-Editable Code-########################################

LabelsFile = 'path/to/hands2017/dataset/test/Training_Annotation.csv'
ImagesFolder = 'path/to/hands2017/dataset/training/images/'

# For the test images use the following two lines
# LabelsFile = 'path/to/hands2017/dataset/test/test_annotation_frame.csv'
# ImagesFolder = 'path/to/hands2017/dataset/frame/images/'

#######################################################################################

dfLabels = pd.read_csv(LabelsFile)
imageList = dfLabels.iloc[:, 0].values
numOfPeople = dfLabels.shape[0]

for i in range(0, numOfPeople):
    # Reading labels file and openning image
    image = np.array(Image.open(ImagesFolder + imageList[i]))
    label = np.asarray(dfLabels.iloc[i, 1:].values, dtype='float32')
    labelP = world2pixel(label.reshape((21, 3)).copy(), fx, fy, u0, v0)
    c = sum(labelP.reshape((21, 3))) / 21.
    centerDepth = c[2]

    # Using the points from label, create the smallest enclosing circle
    circx, circy, radius = make_circle(labelP[:, 0:2].copy())
    center = np.asarray([circx, circy])
    radius = radius + 16
    lefttop_pixel = center - radius
    rightbottom_pixel = center + radius

    new_Xmin = max(lefttop_pixel[0], 0)
    new_Ymin = max(lefttop_pixel[1], 0)
    new_Xmax = min(rightbottom_pixel[0], 639)
    new_Ymax = min(rightbottom_pixel[1], 479)

    # Check if cropped region is too small and skip image if so.
    if new_Xmin > 640 or abs(new_Xmin - new_Xmax) < 20:
        continue

    # Crop image
    imCrop = image.copy()[int(new_Ymin):int(new_Ymax),
                          int(new_Xmin):int(new_Xmax)]
    imgResize = np.asarray(cv2.resize(
        imCrop, (cropSize, cropSize), interpolation=cv2.INTER_NEAREST), dtype='float32')

    # Mean centering the image
    imgResize[np.where(imgResize >= centerDepth + depth_thres)] = centerDepth
    imgResize[np.where(imgResize <= centerDepth - depth_thres)] = centerDepth
    maxPixel = imgResize.max()
    imgResize = np.rint((imgResize / maxPixel) * 255)

    # Marking the depth on image to process during training phase
    mark = np.rint((centerDepth / maxPixel) * 255)
    imgResize[0, 0] = mark

    nameOfFile = imageList[i]

    ################################-Editable Code-########################################

    cv2.imwrite('Cropped/Train/' + nameOfFile, np.uint8(imgResize))

    # For test images, use following line
    # cv2.imwrite('Cropped/Test/' + nameOfFile, np.uint8(imgResize))

    # Checking progress. Comment if not needed
    if i % 100 == 0:
        print('Processed ' + str(i) + ' images')

    #######################################################################################
