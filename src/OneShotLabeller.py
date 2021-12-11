'''
Dataset image labelling function for training the Single Shot Corrective CNN (SSC-CNN)

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

import csv
import numpy as np
import pandas as pd
from utils import DoFv2, world2pixel, fx, fy, u0, v0
from smallestenclosingcircle import make_circle

################################-Editable Code-########################################

# For the train images
LabelsFile = 'path/to/hands2017/dataset/training/Training_Annotation.csv'
csvfile = "oneShotLabelsTrain.csv"

# For the test images
# LabelsFile = 'path/to/hands2017/dataset/test/test_annotation_frame.csv'
# csvfile = "oneShotLabelsTest.csv"

#######################################################################################

dfLabels = pd.read_csv(LabelsFile)
imageList = dfLabels.iloc[:, 0].values
numOfPeople = imageList.shape[0]

with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(
        ["ID", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9",
         "P10", "P11", "P12", "P13", "P14", "P15", "P16", "P17", "P18",
         "L1", "L2", "L3", "L4", "L5", "L6", "L7",
         "L8", "L9", "L10", "L11", "L12", "L13", "L14", "L15",
         "A1", "A2", "A3", "A4", "A5", "A6", "A7",
         "A8", "A9", "A10", "A11", "A12", "A13", "A14",
         "A15", "A16", "A17", "A18", "A19", "A20",
         "Xmin", "Ymin", "Xmax", "Ymax", "cdepth",
         "F19", "F20", "F21", "F22", "F23", "F24", "F25", "F26", "F27", "F28", "F29", "F30", "F31",
         "F32", "F33", "F34", "F35", "F36", "F37", "F38", "F39", "F40", "F41", "F42", "F43", "F44", "F45", "F46",
         "F47", "F48", "F49", "F50", "F51", "F52", "F53", "F54", "F55", "F56", "F57", "F58", "F59", "F60", "F61",
         "F62", "F63"])
    for i in range(0, numOfPeople):
        label = np.asarray(dfLabels.iloc[i, 1:].values, dtype='float32')
        labelP = world2pixel(label.reshape((21, 3)).copy(), fx, fy, u0, v0)
        # for i in range(i, i+1):
        c = sum(labelP.reshape((21, 3))) / 21.
        centerDepth = c[2]

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

        label = np.asarray(dfLabels.iloc[i, 1:].values, dtype='float32')
        others = label[18:].copy()
        pvals = label[0:18].copy()
        label = label.reshape((21, 3))

        def dist(a, b):
            return np.linalg.norm(label[a] - label[b])

        ite = 0
        lens = np.zeros(15)
        lens[ite] = dist(1, 6)
        ite = ite + 1
        lens[ite] = dist(6, 7)
        ite = ite + 1
        lens[ite] = dist(7, 8)
        ite = ite + 1
        lens[ite] = dist(2, 9)
        ite = ite + 1
        lens[ite] = dist(9, 10)
        ite = ite + 1
        lens[ite] = dist(10, 11)
        ite = ite + 1
        lens[ite] = dist(3, 12)
        ite = ite + 1
        lens[ite] = dist(12, 13)
        ite = ite + 1
        lens[ite] = dist(13, 14)
        ite = ite + 1
        lens[ite] = dist(4, 15)
        ite = ite + 1
        lens[ite] = dist(15, 16)
        ite = ite + 1
        lens[ite] = dist(16, 17)
        ite = ite + 1
        lens[ite] = dist(5, 18)
        ite = ite + 1
        lens[ite] = dist(18, 19)
        ite = ite + 1
        lens[ite] = dist(19, 20)

        dofs = DoFv2(label)
        newLabels = np.zeros(103)
        newLabels[0:18] = pvals
        newLabels[18:33] = lens
        newLabels[33:53] = dofs
        newLabels[53] = new_Xmin
        newLabels[54] = new_Ymin
        newLabels[55] = new_Xmax
        newLabels[56] = new_Ymax
        newLabels[57] = centerDepth
        newLabels[58:103] = others

        newLabelList = newLabels.tolist()
        newLabelList.insert(0, imageList[i])
        writer.writerow(newLabelList)
        if i % 100 == 0:
            print('Processed ' + str(i) + ' Images...')
