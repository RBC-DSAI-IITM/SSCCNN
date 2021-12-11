'''
Training code for the Single Shot Corrective CNN (SSC-CNN)

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

import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import tensorflow.keras.backend as K
from DataGenerators import GetGenerators
from ModelDesign import GetModel, Compile, GetModelPre
import numpy as np
import tensorflow_addons as tfa

tf.config.optimizer.set_jit(True)

checkpointFilePath = "newWeights.h5"
filename = 'history.pickle'
epochs = 500
nworkers = 15
strategy = tf.distribute.MirroredStrategy()

print('Making Model')
with strategy.scope():
    ################################-Editable Code-########################################
    # model = GetModelPre('pretrainedWeights.h5')  # If you want to continue training from a model
    model = GetModel()
    #######################################################################################
    model = Compile(model)

# model.summary()
reduce_lr = ReduceLROnPlateau(
    monitor='loss', factor=0.4, patience=4, verbose=1,
    mode='min', min_delta=0.0001, cooldown=0, min_lr=1e-7
)
checkpoint = ModelCheckpoint(checkpointFilePath, monitor='loss', verbose=1, save_weights_only=False,
                             save_best_only=True, mode='min')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
callbacks_list = [reduce_lr, checkpoint, es]

print('Making Generators')
train_generator, validation_generator = GetGenerators()

print('Training Model')
with strategy.scope():
    hist = model.fit(
        train_generator,
        epochs=epochs,
        verbose=1,
        validation_data=validation_generator,
        # validation_steps=1000,
        workers=nworkers,
        max_queue_size=12,
        callbacks=callbacks_list,
        use_multiprocessing=False)
