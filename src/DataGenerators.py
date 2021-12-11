'''
Data generator functions for the Single Shot Corrective CNN (SSC-CNN)

Creates datagenerators for testing and training the model.

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

# Import Libraries
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from utils import cropSize

################################-Editable Code-########################################

# File locations to be filled
trainLabelsFile = "oneShotLabelsTrain.csv"
testLabelsFile = "oneShotLabelsTest.csv"
trainImagesFolder = "Cropped/Train/"
testImagesFolder = "Cropped/Test/"
batch_size = 256

#######################################################################################


def preprocess(image):
    '''
    Takes a single image as input and normalizes for zero mean using the pre-calculated value stored in the first pixel. 
    The output of this function then ranges from -1 to 1
    '''
    mark = image[0, 0].copy()
    image[0, 0] = (image[0, 1] + image[1, 0]) / 2.0
    image = image - mark
    image = image / 255.0
    return image


datagen = ImageDataGenerator(preprocessing_function=preprocess)
features = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9",
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
            "F62", "F63"]


def GetGenerators():
    '''
    Creates datagenerators for testing and training the model.
    '''
    traindf = pd.read_csv(trainLabelsFile, dtype=str)
    testdf = pd.read_csv(testLabelsFile, dtype=str)
    # load and iterate training dataset
    train_generator = datagen.flow_from_dataframe(
        dataframe=traindf,
        directory=trainImagesFolder,
        x_col="ID",
        y_col=features,
        batch_size=batch_size,
        seed=98,
        shuffle=True,
        class_mode='raw',
        target_size=(cropSize, cropSize))

    # load and iterate validation dataset
    validation_generator = datagen.flow_from_dataframe(
        dataframe=testdf,
        directory=testImagesFolder,
        x_col="ID",
        y_col=features,
        seed=98,
        batch_size=batch_size,
        shuffle=True,
        class_mode='raw',
        target_size=(cropSize, cropSize))
    return train_generator, validation_generator
