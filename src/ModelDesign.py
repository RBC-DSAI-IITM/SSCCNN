'''
Model architecture functions for the Single Shot Corrective CNN (SSC-CNN)

Functions to create the model architecture and the cost function.

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
from tensorflow.keras import Model, optimizers
from tensorflow.keras.applications import resnet50
from tensorflow.keras.layers import BatchNormalization, Flatten, Conv2D, MaxPooling2D, Dropout, Input, Dense, \
    GlobalAveragePooling2D, concatenate
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers

from utils import fx, fy, u0, v0

cropSize = 176
WEIGHTS_PATH = (
    'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = (
    'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
opt = optimizers.Adam(learning_rate=0.00035, amsgrad=True)


def deg2rad(deg):
    '''
    Convert degree to radians
    '''
    pi_on_180 = 0.017453292519943295
    return deg * pi_on_180


def p_inv(matrix):
    '''
    Returns the Moore-Penrose pseudo-inverse
    '''
    s, u, v = tf.linalg.svd(matrix)
    threshold = tf.reduce_max(s) * 1e-5
    s_mask = tf.boolean_mask(s, s > threshold)
    s_inv = tf.linalg.diag(
        tf.concat([1. / s_mask, tf.zeros([tf.size(s) - tf.size(s_mask)])], 0))
    return tf.linalg.matmul(v, tf.linalg.matmul(s_inv, tf.transpose(u)))


def equation_plane(p1, p2, p3):
    '''
    Returns the coefficients of the plane in which the given three points lie on
    '''
    x1, y1, z1 = p1[0], p1[1], p1[2]
    x2, y2, z2 = p2[0], p2[1], p2[2]
    x3, y3, z3 = p3[0], p3[1], p3[2]
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)
    return a, b, c, d


def plane_intersect(a, b):
    '''
    Returns the axis of intersection given two planes
    '''
    a_vec, b_vec = a[:3], b[:3]
    aXb_vec = tf.linalg.cross(a_vec, b_vec, name="crosser")
    A = tf.stack([a_vec, b_vec, aXb_vec])
    d = tf.reshape(tf.stack([-a[3], -b[3], 0.]), [3, 1])
    with tf.device("/cpu:0"):
        ptemp = tf.linalg.matmul(tf.linalg.pinv(A), d, name="solver")
    p_inter = tf.transpose(ptemp)
    return p_inter[0], (p_inter + aXb_vec)[0]


def axisfinder(vec, point, plane):
    '''
    Given a vector, a point and a plane, return the vector perpendicular to the vector from the point such that it lies on the plane
    '''
    px, py, pz = point[0], point[1], point[2]
    p = vec[0]
    q = vec[1]
    r = vec[2]
    s = -(p * px) - (q * py) - (r * pz)
    plane2 = tf.stack([p, q, r, s])
    axis1, axis2 = plane_intersect(plane, plane2)
    v = axis1 - axis2
    return tf.math.l2_normalize(v)


def PointRotate3D(p1, p2, p0, radians):
    '''
    Function to rotate a point p0 around the axis created by points p1 and p2 by the specified radians
    '''
    # Translate so axis is at origin
    p = p0 - p1
    # Initialize point q
    N = (p2 - p1)
    # Rotation axis unit vector
    n = tf.math.l2_normalize(N, name="normalizer")

    # Matrix common factors
    c = tf.math.cos(radians)
    t = 1 - tf.math.cos(radians)
    s = tf.math.sin(radians)
    X = n[0]
    Y = n[1]
    Z = n[2]

    # Matrix 'M'
    d11 = t * X * X + c
    d12 = t * X * Y - s * Z
    d13 = t * X * Z + s * Y
    d21 = t * X * Y + s * Z
    d22 = t * Y * Y + c
    d23 = t * Y * Z - s * X
    d31 = t * X * Z - s * Y
    d32 = t * Y * Z + s * X
    d33 = t * Z * Z + c

    q = [0.0, 0.0, 0.0]
    q[0] = d11 * p[0] + d12 * p[1] + d13 * p[2]
    q[1] = d21 * p[0] + d22 * p[1] + d23 * p[2]
    q[2] = d31 * p[0] + d32 * p[1] + d33 * p[2]
    qt = tf.stack(q)
    # Translate axis and rotated point back to original location
    return qt + p1


def finder(p1, p2, p3):
    '''
    Shorthand function to get vector perpendicular to the plane made by the three points
    '''
    pointvec = p2 - p1
    b = equation_plane(p1, p2, p3)
    z = axisfinder(pointvec, p2, b)
    return z


def chain(p1, p2, p3, l1, l2, l3, a1, a2, a3, a4):
    '''
    Function to construct the joint locations given the first joint location (p2) using the other two points as reference.
    The lengths of digits are l1, l2, and l3. The joint angles are a1, a2, a3, and a4.
    '''
    vec = p2 - p1
    vec = tf.math.l2_normalize(vec)
    digit1 = p2 + (vec * l1)
    digit2 = digit1 + (vec * l2)
    digit3 = digit2 + (vec * l3)

    h = finder(p1, p2, p3)

    t1 = digit2 + (4 * h)
    t2 = digit2 - (4 * h)
    digit3 = PointRotate3D(t2, t1, digit3, deg2rad(a4))

    t1 = digit1 + 4 * h
    t2 = digit1 - 4 * h
    digit3 = PointRotate3D(t2, t1, digit3, deg2rad(a3))
    digit2 = PointRotate3D(t2, t1, digit2, deg2rad(a3))

    t1 = p2 + 4 * h
    t2 = p2 - 4 * h
    digit3 = PointRotate3D(t2, t1, digit3, deg2rad(a2))
    digit2 = PointRotate3D(t2, t1, digit2, deg2rad(a2))
    digit1 = PointRotate3D(t2, t1, digit1, deg2rad(a2))

    newvec = p2 - digit1
    plane = equation_plane(p1, p2, digit1)
    o = axisfinder(newvec, digit1, plane)
    t1 = p2 + (4 * o)
    t2 = p2 - (4 * o)
    digit3 = PointRotate3D(t2, t1, digit3, deg2rad(a1))
    digit2 = PointRotate3D(t2, t1, digit2, deg2rad(a1))
    digit1 = PointRotate3D(t2, t1, digit1, deg2rad(a1))
    return digit1, digit2, digit3


def pixel2worldtf(x, fx, fy, ux, uy):
    '''
    Tensorflow version function to convert pixel space to world space
    '''
    x1 = (x[:, 0] - ux) * x[:, 2] / fx
    x2 = (x[:, 1] - uy) * x[:, 2] / fy
    return tf.stack([x1, x2, x[:, 2]], axis=1)


def fivechainCost(y_true, y_pred):
    '''
    Cost function to calculate error given the predicted and ground truth poses.
    '''
    constMul = tf.constant(([cropSize, cropSize, 1000]), dtype=tf.float32)
    locs = tf.reshape(y_pred[0:18], (6, 3)) * constMul
    leng = y_pred[18:33] * 80
    angles = y_pred[33:53]

    locst = tf.reshape(y_true[0:18], (6, 3))
    lent = y_true[18:33]
    anglest = y_true[33:53]
    digitt = tf.reshape(y_true[58:], (15, 3))

    Xmin = y_true[53]
    Xmax = y_true[55]
    Ymin = y_true[54]
    Ymax = y_true[56]
    centerDepth = y_true[57]

    newLoc1 = ((locs[:, 0] * (Xmax - Xmin)) / cropSize) + Xmin
    newLoc2 = ((locs[:, 1] * (Ymax - Ymin)) / cropSize) + Ymin
    newLoc3 = locs[:, 2] + centerDepth
    newLoc = pixel2worldtf(tf.stack([newLoc1, newLoc2, newLoc3], axis=1), fx, fy, u0, v0)
    newAng = tf.stack([(120 * angles[0]) - 30, 80 * angles[1], 90 * angles[2], (45 * angles[3]) - 90,
                       (120 * angles[4]) - 30, 130 * angles[5], (130 *
                                                                 angles[6]) - 40, (30 * angles[7]) - 15,
                       (120 * angles[8]) - 30, 130 * angles[9], (130 *
                                                                 angles[10]) - 40, (30 * angles[11]) - 15,
                       (120 * angles[12]) - 30, 130 * angles[13], (130 *
                                                                   angles[14]) - 40, (30 * angles[15]) - 15,
                       (120 * angles[16]) - 30, 130 * angles[17], (130 * angles[18]) - 40, (30 * angles[19]) - 15])

    cost = tf.reduce_mean(tf.abs(leng - lent)) + tf.reduce_mean(tf.abs(newAng - anglest))

    cost = cost + ((tf.norm(newLoc[0] - locst[0], ord='euclidean') +
                   tf.norm(newLoc[1] - locst[1], ord='euclidean') +
                   tf.norm(newLoc[2] - locst[2], ord='euclidean') +
                   tf.norm(newLoc[3] - locst[3], ord='euclidean') +
                   tf.norm(newLoc[4] - locst[4], ord='euclidean') +
                   tf.norm(newLoc[5] - locst[5], ord='euclidean')) / 6.0)
    return cost


def HandCostFunctionZ(y_trueb, y_pred):
    '''
    Main cost function for the model. Call the fivechainCost function for each hand in the batch.
    '''
    y_true = tf.strings.to_number(y_trueb, out_type=tf.dtypes.float32)
    batchSize = tf.shape(y_true)
    elems = tf.range(0, batchSize[0], dtype=tf.int32)

    def mapFn(i): return fivechainCost(y_true[i], y_pred[i])

    costs = tf.map_fn(mapFn, elems, fn_output_signature=tf.float32)
    cost2 = tf.reduce_sum(costs) / tf.cast(batchSize[0], dtype=tf.float32)
    return cost2


def GetModel():
    '''
    Initializing model architecture with the various layers
    '''
    resnet_model = resnet50.ResNet50(weights='imagenet', input_shape=(
        cropSize, cropSize, 3), include_top=False)
    for layer in resnet_model.layers:
        layer.trainable = False
    last_layer = resnet_model.get_layer('conv4_block6_out')
    last_output = last_layer.output
    x = Conv2D(filters=1024, kernel_size=3, padding='valid',
               activation='relu')(last_output)
    x = MaxPooling2D(2, 2)(x)

    x = Conv2D(filters=1024, kernel_size=3,
               padding='valid', activation='relu')(x)
    x = MaxPooling2D(2, 2)(x)

    x = Flatten()(x)
    x = BatchNormalization()(x)

    split = Dense(512, activation='relu')(x)
    loc = Dense(256, activation='sigmoid')(split)
    loc = Dense(256, activation='sigmoid')(loc)
    loc = Dropout(0.2)(loc)
    loc = Dense(256, activation='sigmoid')(loc)
    loc = Dense(18, activation='sigmoid', name='LocNet')(loc)
    angles = Dense(256, activation='sigmoid')(split)
    angles = Dense(256, activation='sigmoid')(angles)
    angles = Dropout(0.2)(angles)
    angles = Dense(256, activation='sigmoid')(angles)
    angles = Dense(20, activation='sigmoid', name='AngleNet')(angles)
    lengths = Dense(256, activation='sigmoid')(split)
    lengths = Dense(256, activation='sigmoid')(lengths)
    lengths = Dropout(0.2)(lengths)
    lengths = Dense(256, activation='sigmoid')(lengths)
    lengths = Dense(15, activation='sigmoid', name='LengthNet')(lengths)
    op = concatenate([loc, lengths, angles])
    model = Model(inputs=resnet_model.input, outputs=op, name='HandNet')
    return model


def Compile(model):
    '''
    Function to compile new model if pretrained model is not used
    '''
    model.compile(loss=HandCostFunctionZ, optimizer=opt)
    return model


def GetModelPre(file):
    '''
    Set pretrained model if continuous training is done
    '''
    model = load_model(file, custom_objects={
        'HandCostFunctionZ': HandCostFunctionZ, 'tf': tf})
    return model
