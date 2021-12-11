'''
Testing code for the Single Shot Corrective CNN (SSC-CNN)

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

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
from ModelDesign import GetModelPre
from smallestenclosingcircle import make_circle
from utils import world2pixel, fx, fy, u0, v0, cropSize, depth_thres, pixel2world, PointRotate3D, finder, \
    unit_vector, equation_plane, axisfinder

################################-Editable Code-########################################

testImagesFolder = 'path/to/hands2017/dataset/frame/images/'
testLabelsFile = 'path/to/hands2017/dataset/test/test_annotation_frame.csv'
print('Loading Files')

# Path to the model
# This can be changed to the trained weights or you can leave as is for pretrained model
model = GetModelPre('pretrainedWeights.h5')

#######################################################################################


dfLabels = pd.read_csv(testLabelsFile)
imageList = dfLabels.iloc[:, 0].values
nb_samples = len(imageList)


################################-Editable Code-########################################

# Use below lines to replicate reported results
# fil = open("indexes.txt", "r")
# index = np.asarray(fil.readline().split(',')).astype(np.int)

index = range(0, nb_samples)

#######################################################################################

print('Testing Model')
error = 0.0


def preprocess(image):
    '''
    Takes a single image as input and normalizes for zero mean using the pre-calculated value stored in the first pixel. 
    The output of this function then ranges from -1 to 1
    '''
    mark = image[0, 0].copy()
    image[0, 0] = (image[0, 1] + image[1, 0]) / 2.
    image = image - mark
    image = image / 255.
    return image


def chain(p1, p2, p3, l1, l2, l3, a1, a2, a3, a4):
    '''
    Function to construct the joint locations given the first joint location (p2) using the other two points as reference.
    The lengths of digits are l1, l2, and l3. The joint angles are a1, a2, a3, and a4.
    '''
    vec = p2 - p1
    vec = unit_vector(vec)
    digit1 = p2 + (vec * l1)
    digit2 = digit1 + (vec * l2)
    digit3 = digit2 + (vec * l3)

    h = finder(p1, p2, p3)

    t1 = digit2 + (4 * h)
    t2 = digit2 - (4 * h)
    digit3 = PointRotate3D(t2, t1, digit3, np.deg2rad(a4))

    t1 = digit1 + 4 * h
    t2 = digit1 - 4 * h
    digit3 = PointRotate3D(t2, t1, digit3, np.deg2rad(a3))
    digit2 = PointRotate3D(t2, t1, digit2, np.deg2rad(a3))

    t1 = p2 + 4 * h
    t2 = p2 - 4 * h
    digit3 = PointRotate3D(t2, t1, digit3, np.deg2rad(a2))
    digit2 = PointRotate3D(t2, t1, digit2, np.deg2rad(a2))
    digit1 = PointRotate3D(t2, t1, digit1, np.deg2rad(a2))

    newvec = p2 - digit1
    plane = equation_plane(p1, p2, digit1)
    o = axisfinder(newvec, digit1, plane)
    t1 = p2 + (4 * o)
    t2 = p2 - (4 * o)
    digit3 = PointRotate3D(t2, t1, digit3, np.deg2rad(a1))
    digit2 = PointRotate3D(t2, t1, digit2, np.deg2rad(a1))
    digit1 = PointRotate3D(t2, t1, digit1, np.deg2rad(a1))
    return digit1, digit2, digit3


# Test commences
for i in index:
    image = np.array(Image.open(testImagesFolder + imageList[i]))
    label = np.asarray(dfLabels.iloc[i, 1:].values, dtype='float32')
    labelP = world2pixel(label.reshape((21, 3)).copy(), fx, fy, u0, v0)
    circx, circy, radius = make_circle(labelP[:, 0:2].copy())

    center = np.asarray([circx, circy])
    radius = radius + 16
    lefttop_pixel = center - radius
    rightbottom_pixel = center + radius

    new_Xmin = max(lefttop_pixel[0], 0)
    new_Ymin = max(lefttop_pixel[1], 0)
    new_Xmax = min(rightbottom_pixel[0], image.shape[1] - 1)
    new_Ymax = min(rightbottom_pixel[1], image.shape[0] - 1)

    if new_Xmin > 640 or abs(new_Xmin - new_Xmax) < 20:
        continue

    imCrop = image.copy()[int(new_Ymin):int(new_Ymax),
                          int(new_Xmin):int(new_Xmax)]
    imgResize = np.asarray(cv2.resize(imCrop, (cropSize, cropSize), interpolation=cv2.INTER_NEAREST),
                           dtype='float32')
    c = sum(labelP.reshape((21, 3))) / 21.
    centerDepth = c[2]
    imgResize[np.where(imgResize >= centerDepth + depth_thres)] = centerDepth
    imgResize[np.where(imgResize <= centerDepth - depth_thres)] = centerDepth
    maxPixel = imgResize.max()

    mark = centerDepth / maxPixel
    imgResize = (imgResize / maxPixel) - mark
    trueJoints = np.reshape(label, (21, 3))

    img = np.squeeze(np.stack((imgResize,) * 3, -1))
    img = np.reshape(img, (1, 176, 176, 3))
    y_pred = model.predict(img)

    locs = np.reshape(y_pred[0, 0:18], [6, 3]) * \
        np.asarray([cropSize, cropSize, 1000.0], dtype=np.float)
    leng = y_pred[0, 18:33] * 80
    angles = y_pred[0, 33:53]

    newLoc1 = ((locs[:, 0] * (new_Xmax - new_Xmin)) / cropSize) + new_Xmin
    newLoc2 = ((locs[:, 1] * (new_Ymax - new_Ymin)) / cropSize) + new_Ymin
    newLoc3 = locs[:, 2] + centerDepth
    newLoc = pixel2world(
        np.stack([newLoc1, newLoc2, newLoc3], axis=1), fx, fy, u0, v0)
    newAng = np.stack([(120 * angles[0]) - 30, 80 * angles[1], 90 * angles[2], (45 * angles[3]) - 90,
                       (120 * angles[4]) - 30, 130 * angles[5], (130 *
                                                                 angles[6]) - 40, (30 * angles[7]) - 15,
                       (120 * angles[8]) - 30, 130 * angles[9], (130 *
                                                                 angles[10]) - 40, (30 * angles[11]) - 15,
                       (120 * angles[12]) - 30, 130 * angles[13], (130 *
                                                                   angles[14]) - 40, (30 * angles[15]) - 15,
                       (120 * angles[16]) - 30, 130 * angles[17], (130 * angles[18]) - 40, (30 * angles[19]) - 15])

    digit1, digit2, digit3 = chain(newLoc[0], newLoc[1], newLoc[2], leng[0], leng[1], leng[2],
                                   newAng[0], newAng[1], newAng[2], newAng[3])
    digit4, digit5, digit6 = chain(newLoc[0], newLoc[2], newLoc[3], leng[3], leng[4], leng[5],
                                   newAng[4], newAng[5], newAng[6], newAng[7])
    digit7, digit8, digit9 = chain(newLoc[0], newLoc[3], newLoc[4], leng[6], leng[7], leng[8],
                                   newAng[8], newAng[9], newAng[10], newAng[11])
    digit10, digit11, digit12 = chain(newLoc[0], newLoc[4], newLoc[5], leng[9], leng[10], leng[11],
                                      newAng[12], newAng[13], newAng[14], newAng[15])
    digit13, digit14, digit15 = chain(newLoc[0], newLoc[5], newLoc[4], leng[12], leng[13], leng[14],
                                      -newAng[16], -newAng[17], -newAng[18], -newAng[19])

    fullList = []

    sumOfError = np.linalg.norm(newLoc[0] - trueJoints[0]) + \
        np.linalg.norm(newLoc[1] - trueJoints[1]) + \
        np.linalg.norm(newLoc[2] - trueJoints[2]) + \
        np.linalg.norm(newLoc[3] - trueJoints[3]) + \
        np.linalg.norm(newLoc[4] - trueJoints[4]) + \
        np.linalg.norm(newLoc[5] - trueJoints[5]) + \
        np.linalg.norm(digit1 - trueJoints[6]) + \
        np.linalg.norm(digit2 - trueJoints[7]) + \
        np.linalg.norm(digit3 - trueJoints[8]) + \
        np.linalg.norm(digit4 - trueJoints[9]) + \
        np.linalg.norm(digit5 - trueJoints[10]) + \
        np.linalg.norm(digit6 - trueJoints[11]) + \
        np.linalg.norm(digit7 - trueJoints[12]) + \
        np.linalg.norm(digit8 - trueJoints[13]) + \
        np.linalg.norm(digit9 - trueJoints[14]) + \
        np.linalg.norm(digit10 - trueJoints[15]) + \
        np.linalg.norm(digit11 - trueJoints[16]) + \
        np.linalg.norm(digit12 - trueJoints[17]) + \
        np.linalg.norm(digit13 - trueJoints[18]) + \
        np.linalg.norm(digit14 - trueJoints[19]) + \
        np.linalg.norm(digit15 - trueJoints[20])

    error += sumOfError / 21.0
    if i % 100 == 0:
        print("Processed " + str(i) + " images...")

overall = error / nb_samples
print("Overall 3D joint Error is : " + str(overall))
