'''
Utilities for the Single Shot Corrective CNN (SSC-CNN)
Some of these functions are in ModelDesign.py but they are TF versions of the function

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

import numpy as np
import matplotlib.pyplot as plt

xmain = np.asarray([1, 0, 0])
ymain = np.asarray([0, 1, 0])
zmain = np.asarray([0, 0, 1])
origin = np.asarray([0, 0, 0])
tensor01 = np.asarray([0, 1])
tensor10 = np.asarray([1, 0])
tensor00 = np.asarray([0, 0])
xy_thres = 100
depth_thres = 150
cropSize = 176

fx = 475.065948
fy = 475.065857
u0 = 315.944855
v0 = 245.287079


def line3D(y_predRot, x, y, ax):
    ax.plot3D(np.asarray([y_predRot[x][0], y_predRot[y][0]]),
              np.asarray([y_predRot[x][1], y_predRot[y][1]]),
              np.asarray([y_predRot[x][2], y_predRot[y][2]]), 'k')


def plot3DJLine(y_values):
    y_predRot = y_values
    plt.gca().cla()
    ax = plt.axes(projection='3d')
    for i in range(1, 2):
        ax.scatter3D(y_predRot[i][0], y_predRot[i][1], y_predRot[i][2], color="k")
    for i in range(6, 9):
        ax.scatter3D(y_predRot[i][0], y_predRot[i][1], y_predRot[i][2], color="r")
    for i in range(9, 12):
        ax.scatter3D(y_predRot[i][0], y_predRot[i][1], y_predRot[i][2], color="g")
    for i in range(12, 15):
        ax.scatter3D(y_predRot[i][0], y_predRot[i][1], y_predRot[i][2], color="b")
    for i in range(15, 18):
        ax.scatter3D(y_predRot[i][0], y_predRot[i][1], y_predRot[i][2], color="m")
    for i in range(18, 21):
        ax.scatter3D(y_predRot[i][0], y_predRot[i][1], y_predRot[i][2], color="c")
    ax.plot3D(np.asarray([y_predRot[1][0], y_predRot[1][0] + 2]),
              np.asarray([y_predRot[1][1], y_predRot[1][1]]),
              np.asarray([y_predRot[1][2], y_predRot[1][2]]), 'r')
    ax.plot3D(np.asarray([y_predRot[1][0], y_predRot[1][0]]),
              np.asarray([y_predRot[1][1], y_predRot[1][1] + 18]),
              np.asarray([y_predRot[1][2], y_predRot[1][2]]), 'g')
    ax.plot3D(np.asarray([y_predRot[1][0], y_predRot[1][0]]),
              np.asarray([y_predRot[1][1], y_predRot[1][1]]),
              np.asarray([y_predRot[1][2], y_predRot[1][2] + 8]), 'b')

    line3D(y_predRot, 0, 1, ax)
    line3D(y_predRot, 0, 2, ax)
    line3D(y_predRot, 0, 3, ax)
    line3D(y_predRot, 0, 4, ax)
    line3D(y_predRot, 0, 5, ax)
    line3D(y_predRot, 1, 6, ax)
    line3D(y_predRot, 6, 7, ax)
    line3D(y_predRot, 7, 8, ax)
    line3D(y_predRot, 2, 9, ax)
    line3D(y_predRot, 9, 10, ax)
    line3D(y_predRot, 10, 11, ax)
    line3D(y_predRot, 3, 12, ax)
    line3D(y_predRot, 12, 13, ax)
    line3D(y_predRot, 13, 14, ax)
    line3D(y_predRot, 4, 15, ax)
    line3D(y_predRot, 15, 16, ax)
    line3D(y_predRot, 16, 17, ax)
    line3D(y_predRot, 5, 18, ax)
    line3D(y_predRot, 18, 19, ax)
    line3D(y_predRot, 19, 20, ax)
    # plt.savefig(name, format='png')
    plt.draw()
    plt.pause(0.1)



def normalize(v):
    '''
    Returns normalized vector
    '''
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def PointRotate3D(p1, p2, p0, radians):
    '''
    Function to rotate a point p0 around the axis created by points p1 and p2 by the specified radians
    '''
    # Translate so axis is at origin
    p = p0 - p1
    # Initialize point q
    N = (p2 - p1)
    # Rotation axis unit vector
    n = normalize(N)

    # Matrix common factors
    c = np.cos(radians)
    t = 1 - np.cos(radians)
    s = np.sin(radians)
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

    q = np.zeros(3)
    q[0] = d11 * p[0] + d12 * p[1] + d13 * p[2]
    q[1] = d21 * p[0] + d22 * p[1] + d23 * p[2]
    q[2] = d31 * p[0] + d32 * p[1] + d33 * p[2]

    # Translate axis and rotated point back to original location
    return q + p1


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    ''' 
    Returns angle between two vectors in 3D
    '''
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return rad


def angle_2d(a, b, c):
    '''
    Returns 2d angle given two points (2D points)
    '''
    v1 = a - b
    v2 = c - b
    v1norm = normalize(v1)
    v2norm = normalize(v2)
    res = v1norm[0] * v2norm[0] + v1norm[1] * v2norm[1]
    angle_rad = np.arccos(res)
    if np.isnan(angle_rad):
        return 0.
    else:
        return np.degrees(angle_rad)


def angle_3d(a, b, c):
    '''
    Returns 3d angle given two points (3D points)
    '''
    v1 = a - b
    v2 = c - b
    v1norm = normalize(v1)
    v2norm = normalize(v2)
    res = v1norm[0] * v2norm[0] + v1norm[1] * v2norm[1] + v1norm[2] * v2norm[2]
    angle_rad = np.arccos(res)
    if np.isnan(angle_rad):
        return 0.
    else:
        return np.degrees(angle_rad)


def getAngleFrom(x, y, z, y_predRot, indices, bias=tensor00):  # for shorthand
    return angle_2d(y_predRot[x][indices],
                    y_predRot[y][indices],
                    y_predRot[z][indices] + bias)


def rotator(jointsToRot, alignment, angle, adder):
    '''
    Shorthand code to rotate point by an angle around an axis
    '''
    sizeOfArray = np.shape(jointsToRot)[0]
    values = np.zeros((sizeOfArray, 3))
    for j in range(0, sizeOfArray):
        values[j, :] = (PointRotate3D(jointsToRot[alignment],
                                      jointsToRot[alignment] + adder,
                                      jointsToRot[j],
                                      np.radians(angle)))
    return values


def aligner(joints, alignJoint1, alignJoint2, alignJoint3):
    # Remove the other two components by testing rotations
    indices = [0, 1]
    angleFirst = getAngleFrom(alignJoint2, alignJoint1,
                              alignJoint1, joints, indices, tensor10)
    # print(angleFirst)
    test = PointRotate3D(joints[alignJoint1],
                         joints[alignJoint1] + zmain,
                         joints[alignJoint2],
                         np.radians(angleFirst))

    testAngle = angle_2d(test[indices],
                         joints[alignJoint1][indices],
                         joints[alignJoint1][indices] + tensor10)

    if np.round(testAngle) == 0:
        joints2 = rotator(joints, alignJoint1, angleFirst, zmain)
    else:
        joints2 = rotator(joints, alignJoint1, 360 - angleFirst, zmain)

    indices = [0, 2]
    angleSecond = getAngleFrom(
        alignJoint2, alignJoint1, alignJoint1, joints2, indices, tensor10)
    # print(angleSecond)

    test = PointRotate3D(joints2[alignJoint1],
                         joints2[alignJoint1] + ymain,
                         joints2[alignJoint2],
                         np.radians(angleSecond))

    testAngle = angle_2d(test[indices],
                         joints2[alignJoint1][indices],
                         joints2[alignJoint1][indices] + tensor10)
    # print(testAngle)
    # temp3 = testAngle

    if np.round(testAngle) == 0:
        joints3 = rotator(joints2, alignJoint1, angleSecond, ymain)
    else:
        joints3 = rotator(joints2, alignJoint1, 360 - angleSecond, ymain)

    indices = [1, 2]
    # Align with another joint for reference
    finalRotation = getAngleFrom(
        alignJoint3, alignJoint1, alignJoint1, joints3, indices, tensor10)
    test = PointRotate3D(joints3[alignJoint1],
                         joints3[alignJoint1] + xmain,
                         joints3[alignJoint3],
                         np.radians(finalRotation))

    testAngle = angle_2d(test[1:3],
                         joints3[alignJoint1][1:3],
                         joints3[alignJoint1][1:3] + tensor10)

    if np.round(testAngle) == 0:
        jointsR = rotator(joints3, alignJoint1, finalRotation, xmain)
    else:
        jointsR = rotator(joints3, alignJoint1, 360 - finalRotation, xmain)
    return jointsR


def fingerShorthand(y, a, b, c, indices):
    y_rot = aligner(y, a, b, c)
    val = 180 - getAngleFrom(a, b, c, y_rot, indices)
    if y_rot[c][1] < y_rot[b][1]:
        return -1 * val
    else:
        return val


def equation_plane(p1, p2, p3):
    '''
    Returns equation of plane given 3 points
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
    Returns axis of intersection of two planes
    '''
    a_vec, b_vec = a[:3], b[:3]
    aXb_vec = np.cross(a_vec, b_vec)
    A = np.asarray([a_vec, b_vec, aXb_vec])
    d = np.reshape(np.asarray([-a[3], -b[3], 0.]), [3, 1])
    ptemp = np.matmul(np.linalg.pinv(A), d)
    p_inter = np.transpose(ptemp)
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
    plane2 = np.asarray([p, q, r, s])
    axis1, axis2 = plane_intersect(plane, plane2)
    v = axis1 - axis2
    return v/np.sqrt((v**2).sum())


def finder(p1, p2, p3):
    '''
    Shorthand function to get vector perpendicular to the plane made by the three points
    '''
    pointvec = p2 - p1
    b = equation_plane(p1, p2, p3)
    z = axisfinder(pointvec, p2, b)
    return z


def DoFv2(y_values):
    '''
    Calculates the individual joint angles given the joint positions of the hand
    '''
    Dofs = np.zeros(20)
    y_predRot = aligner(y_values, 1, 2, 3)
    signValt = 1.0
    if y_predRot[6][2] < y_predRot[2][2]:
        signValt = -1.0
    newPoint = y_predRot[6].copy()
    newPoint[2] = y_predRot[2][2]
    Dofs[0] = signValt * angle_3d(y_predRot[6], y_predRot[1], newPoint)

    indices = [0, 1]
    val = getAngleFrom(2, 1, 6, y_predRot, indices)
    if y_predRot[6][1] > y_predRot[1][1]:
        val = - val
    Dofs[1] = val

    y_predRot = aligner(y_predRot, 1, 6, 2)
    val = 180 - getAngleFrom(1, 6, 7, y_predRot, indices)
    if y_predRot[7][2] < y_predRot[6][2]:
        val = val * -1
    Dofs[2] = val

    h = finder(y_predRot[1], y_predRot[6], y_predRot[2])
    refPoint = y_predRot[6] - (h * 4)
    y_predRotRef = aligner(np.vstack((y_predRot, refPoint)), 6, 7, 21)
    signValt = 1.0
    if y_predRotRef[8][2] < y_predRotRef[7][2]:
        signValt = -1.0
    y_predRot = aligner(y_predRot, 6, 7, 8)
    Dofs[3] = signValt * (180 - getAngleFrom(6, 7, 8, y_predRot, indices))

    # Finger 2 - Index
    # Checking back bending of finger i and m
    y_predRot = aligner(y_predRot, 0, 2, 3)
    signVali = 1.0
    signValm = 1.0
    if y_predRot[9][2] < y_predRot[2][2]:
        signVali = -1.0
    if y_predRot[12][2] < y_predRot[3][2]:
        signValm = -1.0
    newPoint = y_predRot[9].copy()
    newPoint[2] = y_predRot[2][2]
    Dofs[4] = signVali * angle_3d(y_predRot[9], y_predRot[2], newPoint)
    newPoint = y_predRot[12].copy()
    newPoint[2] = y_predRot[3][2]
    Dofs[8] = signValm * angle_3d(y_predRot[12], y_predRot[3], newPoint)

    y_predRot = aligner(y_predRot, 3, 2, 9)
    Dofs[5] = 90 - getAngleFrom(3, 2, 9, y_predRot, indices)
    signVali = 1.0
    if y_predRot[10][2] < y_predRot[9][2]:
        signVali = -1.0

    y_predRot = aligner(y_predRot, 2, 9, 10)
    Dofs[6] = signVali * (180 - getAngleFrom(2, 9, 10, y_predRot, indices))

    h = finder(y_predRot[2], y_predRot[9], y_predRot[3])
    refPoint = y_predRot[9] - (h * 4)
    y_predRotRef = aligner(np.vstack((y_predRot, refPoint)), 9, 10, 21)
    signVali = 1.0
    if y_predRotRef[11][2] < y_predRotRef[10][2]:
        signVali = -1.0
    y_predRot = aligner(y_predRot, 9, 10, 11)
    Dofs[7] = signVali * (180 - getAngleFrom(9, 10, 11, y_predRot, indices))

    # Finger 3 - Middle

    # Dof[8] done already up in index section

    y_predRot = aligner(y_predRot, 4, 3, 12)
    Dofs[9] = 90 - getAngleFrom(4, 3, 12, y_predRot, indices)
    signValm = 1.0
    if y_predRot[13][2] < y_predRot[12][2]:
        signValm = -1.0

    y_predRot = aligner(y_predRot, 3, 12, 13)
    Dofs[10] = signValm * (180 - getAngleFrom(3, 12, 13, y_predRot, indices))

    h = finder(y_predRot[3], y_predRot[12], y_predRot[4])
    refPoint = y_predRot[12] - (h * 4)
    y_predRotRef = aligner(np.vstack((y_predRot, refPoint)), 12, 13, 21)
    signValm = 1.0
    if y_predRotRef[14][2] < y_predRotRef[13][2]:
        signValm = -1.0
    y_predRot = aligner(y_predRot, 12, 13, 14)
    Dofs[11] = signValm * (180 - getAngleFrom(12, 13, 14, y_predRot, indices))

    # Finger 4 - Ring
    # Checking back bending of finger r and p
    y_predRot = aligner(y_predRot, 0, 4, 5)
    signValr = 1.0
    signValp = 1.0
    if y_predRot[15][2] < y_predRot[4][2]:
        signValr = -1.0
    if y_predRot[18][2] < y_predRot[5][2]:
        signValp = -1.0
    newPoint = y_predRot[15].copy()
    newPoint[2] = y_predRot[4][2]
    Dofs[12] = signValr * angle_3d(y_predRot[15], y_predRot[4], newPoint)
    newPoint = y_predRot[18].copy()
    newPoint[2] = y_predRot[5][2]
    Dofs[16] = signValp * angle_3d(y_predRot[18], y_predRot[5], newPoint)

    y_predRot = aligner(y_predRot, 5, 4, 15)
    Dofs[13] = 90 - getAngleFrom(5, 4, 15, y_predRot, indices)
    signValr = 1.0
    if y_predRot[16][2] < y_predRot[15][2]:
        signValr = -1.0

    y_predRot = aligner(y_predRot, 4, 15, 16)
    Dofs[14] = signValr * (180 - getAngleFrom(4, 15, 16, y_predRot, indices))

    h = finder(y_predRot[4], y_predRot[15], y_predRot[5])
    refPoint = y_predRot[15] - (h * 4)
    y_predRotRef = aligner(np.vstack((y_predRot, refPoint)), 15, 16, 21)
    signValr = 1.0
    if y_predRotRef[17][2] < y_predRotRef[16][2]:
        signValr = -1.0
    y_predRot = aligner(y_predRot, 15, 16, 17)
    Dofs[15] = signValr * (180 - getAngleFrom(15, 16, 17, y_predRot, indices))

    # Finger 5 Pinky

    # Dof[16] done already up in ring section

    y_predRot = aligner(y_predRot, 4, 5, 18)
    Dofs[17] = getAngleFrom(4, 5, 18, y_predRot, indices) - 90
    signValp = -1.0
    if y_predRot[19][2] < y_predRot[18][2]:
        signValp = 1.0
    # print(Dofs[17])

    y_predRot = aligner(y_predRot, 5, 18, 19)
    Dofs[18] = signValp * (180 - getAngleFrom(5, 18, 19, y_predRot, indices))

    h = finder(y_predRot[5], y_predRot[18], y_predRot[4])
    # invert as the refpoints is the other way round
    refPoint = y_predRot[18] + (h * 4)
    y_predRotRef = aligner(np.vstack((y_predRot, refPoint)), 18, 19, 21)
    signValp = 1.0
    if y_predRotRef[17][2] < y_predRotRef[16][2]:
        signValp = -1.0
    y_predRot = aligner(y_predRot, 18, 19, 20)
    Dofs[19] = signValp * (180 - getAngleFrom(18, 19, 20, y_predRot, indices))
    Dofs = np.round(Dofs, decimals=2)
    return Dofs


def world2pixel(x, fx, fy, ux, uy):
    '''
    Converts world space to pixel space
    '''
    x[:, 0] = x[:, 0] * fx / x[:, 2] + ux
    x[:, 1] = x[:, 1] * fy / x[:, 2] + uy
    return x


def pixel2world(x, fx, fy, ux, uy):
    '''
    Converts pixel space to world space
    '''
    x[:, 0] = (x[:, 0] - ux) * x[:, 2] / fx
    x[:, 1] = (x[:, 1] - uy) * x[:, 2] / fy
    return x


def coplanar(y_values, x, y, z, alpha):
    '''
    Checks for coplanarity between the joints of one finger
    '''
    y_predRot = aligner(y_values, x, y, z)
    indices = [0, 2]
    val = 90 - angle_2d(y_predRot[y][indices] + tensor01,
                        y_predRot[y][indices],
                        y_predRot[y + 1][indices])
    joints = [y, y + 1]
    y_predRot[joints] = fixer(y_predRot, val, val / 2, val / 2, joints, ymain, alpha)

    joints = [y + 1, y]
    y_predRot[joints] = fixer(y_predRot, val, 0, 0, joints, ymain, alpha)
    return y_predRot


def fixer(y_predRot1, mainAngle, maxAngle, minAngle, joints, axis, alpha):
    if mainAngle > maxAngle:
        # print("Doing max")
        angle = alpha * (mainAngle - maxAngle)
        # print("angle fix is " + str(angle))
        y_predRot1[joints] = rotator(y_predRot1[joints], 0, angle, axis)
    elif mainAngle < minAngle:
        # print("Doing min")
        angle = alpha * (mainAngle - minAngle)
        y_predRot1[joints] = rotator(y_predRot1[joints], 0, 360 + angle, axis)
    return y_predRot1[joints]


def boundcheck(y_predRot1, mainAngle, maxAngle, minAngle, joints, axis, alpha):
    if mainAngle > maxAngle:
        return np.abs(np.abs(mainAngle) - np.abs(maxAngle))
    elif mainAngle < minAngle:
        return np.abs(np.abs(mainAngle) - np.abs(minAngle))
    else:
        return 0.0


def correctionModule(y_values, alpha):
    '''
    Corrector Module for fixing hand joints
    '''
    y_predRot = coplanar(y_values, 1, 6, 8, alpha)
    y_predRot = coplanar(y_predRot, 2, 9, 11, alpha)
    y_predRot = coplanar(y_predRot, 3, 12, 14, alpha)
    y_predRot = coplanar(y_predRot, 4, 15, 17, alpha)
    y_predRot = coplanar(y_predRot, 5, 18, 20, alpha)
    realError = 0.0
    debug = False # For debugging

    # Thumb
    y_predRot = aligner(y_predRot, 1, 2, 3)
    indices = [0, 1]
    valCurrent = getAngleFrom(3, 1, 6, y_predRot, indices) - getAngleFrom(3, 1, 2, y_predRot, indices)
    thumb = [1, 6, 7, 8]
    temp = y_predRot.copy()
    y_predRot[thumb] = fixer(y_predRot, valCurrent, 45, -20, thumb, zmain, alpha)
    setFirst = y_predRot[thumb]
    val = getAngleFrom(3, 1, 6, y_predRot, indices) - getAngleFrom(3, 1, 2, y_predRot, indices)
    checkerA = np.round(boundcheck(y_predRot, val, 45, -20, thumb, zmain, alpha))
    y_predRot[thumb] = fixer(temp, valCurrent, 45, -20, thumb, zmain, -alpha)
    val = getAngleFrom(3, 1, 6, y_predRot, indices) - getAngleFrom(3, 1, 2, y_predRot, indices)
    checkerB = np.round(boundcheck(y_predRot, val, 45, -20, thumb, zmain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[thumb] = setFirst
    if debug:
        print(realError)

    indices = [0, 2]
    valCurrent = getAngleFrom(2, 1, 6, y_predRot, indices)
    if y_predRot[6][2] < y_predRot[1][2]:
        valCurrent = 360. - valCurrent
    temp = y_predRot.copy()
    y_predRot[thumb] = fixer(y_predRot, valCurrent, 45, 0, thumb, ymain, alpha)
    setFirst = y_predRot[thumb]
    val = getAngleFrom(2, 1, 6, y_predRot, indices)
    if y_predRot[6][2] < y_predRot[1][2]:
        val = 360. - val
    checkerA = np.round(boundcheck(y_predRot, val, 45, 0, thumb, ymain, alpha))
    y_predRot[thumb] = fixer(temp, valCurrent, 45, 0, thumb, ymain, -alpha)
    val = getAngleFrom(2, 1, 6, y_predRot, indices)
    if y_predRot[6][2] < y_predRot[1][2]:
        val = 360. - val
    checkerB = np.round(boundcheck(y_predRot, val, 45, 0, thumb, ymain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[thumb] = setFirst
    if debug:
        print(realError)

    y_predRot = aligner(y_predRot, 1, 6, 2)
    valCurrent = 180 - getAngleFrom(1, 6, 7, y_predRot, indices)
    if y_predRot[7][2] < y_predRot[6][2]:
        valCurrent = valCurrent * -1
    thumb = [6, 7, 8]
    temp = y_predRot.copy()
    y_predRot[thumb] = fixer(y_predRot, valCurrent, 80, 0, thumb, ymain, alpha)
    setFirst = y_predRot[thumb]
    val = 180 - getAngleFrom(1, 6, 7, y_predRot, indices)
    if y_predRot[7][2] < y_predRot[6][2]:
        val = val * -1
    checkerA = np.round(boundcheck(y_predRot, val, 80, 0, thumb, ymain, alpha))
    y_predRot[thumb] = fixer(temp, valCurrent, 80, 0, thumb, ymain, -alpha)
    val = 180 - getAngleFrom(1, 6, 7, y_predRot, indices)
    if y_predRot[7][2] < y_predRot[6][2]:
        val = val * -1
    checkerB = np.round(boundcheck(y_predRot, val, 80, 0, thumb, ymain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[thumb] = setFirst
    if debug:
        print(realError)

    indices = [0, 1]
    valCurrent = 90 - getAngleFrom(7, 6, 6, y_predRot, indices, tensor01)
    temp = y_predRot.copy()
    y_predRot[thumb] = fixer(y_predRot, valCurrent, 7, -12, thumb, zmain, alpha)
    setFirst = y_predRot[thumb]
    val = 90 - getAngleFrom(7, 6, 6, y_predRot, indices, tensor01)
    checkerA = np.round(boundcheck(y_predRot, val, 7, -12, thumb, zmain, alpha))
    y_predRot[thumb] = fixer(temp, valCurrent, 7, -12, thumb, zmain, -alpha)
    val = 90 - getAngleFrom(7, 6, 6, y_predRot, indices, tensor01)
    checkerB = np.round(boundcheck(y_predRot, val, 7, -12, thumb, zmain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[thumb] = setFirst
    if debug:
        print(realError)

    indices = [1, 2]
    valCurrent = getAngleFrom(8, 7, 7, y_predRot, indices, tensor01)
    thumb = [7, 8]
    temp = y_predRot.copy()
    y_predRot[thumb] = fixer(y_predRot, valCurrent, 0, 0, thumb, -xmain, alpha)
    setFirst = y_predRot[thumb]
    val = getAngleFrom(8, 7, 7, y_predRot, indices, tensor01)
    checkerA = np.round(boundcheck(y_predRot, val, 0, 0, thumb, -xmain, alpha))
    y_predRot[thumb] = fixer(temp, valCurrent, 0, 0, thumb, -xmain, -alpha)
    val = getAngleFrom(8, 7, 7, y_predRot, indices, tensor01)
    checkerB = np.round(boundcheck(y_predRot, val, 0, 0, thumb, -xmain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[thumb] = setFirst
    if debug:
        print(realError)

    indices = [0, 2]
    valCurrent = 180 - getAngleFrom(6, 7, 8, y_predRot, indices)
    if y_predRot[8][2] < y_predRot[7][2]:
        valCurrent = -1 * valCurrent
    thumb = [7, 8]
    temp = y_predRot.copy()
    y_predRot[thumb] = fixer(y_predRot, valCurrent, 90, -30, thumb, ymain, alpha)
    setFirst = y_predRot[thumb]
    val = 180 - getAngleFrom(6, 7, 8, y_predRot, indices)
    if y_predRot[8][2] < y_predRot[7][2]:
        val = -1 * val
    checkerA = np.round(boundcheck(y_predRot, val, 90, -30, thumb, ymain, alpha))
    y_predRot[thumb] = fixer(temp, valCurrent, 90, -30, thumb, ymain, -alpha)
    val = 180 - getAngleFrom(6, 7, 8, y_predRot, indices)
    if y_predRot[8][2] < y_predRot[7][2]:
        val = -1 * val
    checkerB = np.round(boundcheck(y_predRot, val, 90, -30, thumb, ymain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[thumb] = setFirst
    if debug:
        print(realError)

    # Finger 2 - Index
    y_predRot = aligner(y_predRot, 0, 2, 9)
    indices = [0, 1]
    valCurrent = 180 - getAngleFrom(0, 2, 9, y_predRot, indices)
    if y_predRot[9][1] < y_predRot[2][1]:
        valCurrent = -1 * valCurrent
    index = [2, 9, 10, 11]
    temp = y_predRot.copy()
    y_predRot[index] = fixer(y_predRot, valCurrent, 90, -40, index, -zmain, alpha)
    setFirst = y_predRot[index]
    val = 180 - getAngleFrom(0, 2, 9, y_predRot, indices)
    if y_predRot[9][1] < y_predRot[2][1]:
        val = -1 * val
    checkerA = np.round(boundcheck(y_predRot, val, 90, -40, index, -zmain, alpha))
    y_predRot[index] = fixer(temp, valCurrent, 90, -40, index, -zmain, -alpha)
    val = 180 - getAngleFrom(0, 2, 9, y_predRot, indices)
    if y_predRot[9][1] < y_predRot[2][1]:
        val = -1 * val
    checkerB = np.round(boundcheck(y_predRot, val, 90, -40, index, -zmain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[index] = setFirst
    if debug:
        print(realError)

    y_predRot = aligner(y_predRot, 3, 2, 9)
    valCurrent = 90 - getAngleFrom(3, 2, 9, y_predRot, indices)
    temp = y_predRot.copy()
    y_predRot[index] = fixer(y_predRot, valCurrent, 15, -15, index, -zmain, alpha)
    setFirst = y_predRot[index]
    val = 90 - getAngleFrom(3, 2, 9, y_predRot, indices)
    checkerA = np.round(boundcheck(y_predRot, val, 15, -15, index, -zmain, alpha))
    y_predRot[index] = fixer(y_predRot, valCurrent, 15, -15, index, -zmain, -alpha)
    val = 90 - getAngleFrom(3, 2, 9, y_predRot, indices)
    checkerB = np.round(boundcheck(y_predRot, val, 15, -15, index, -zmain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[index] = setFirst
    if debug:
        print(realError)

    indices = [0, 2]
    y_predRot = aligner(y_predRot, 2, 9, 10)
    valCurrent = 180 - getAngleFrom(2, 9, 10, y_predRot, indices)
    if y_predRot[10][1] < y_predRot[9][1]:
        valCurrent = -1 * valCurrent
    index = [9, 10, 11]
    temp = y_predRot.copy()
    y_predRot[index] = fixer(y_predRot, valCurrent, 130, 0, index, ymain, alpha)
    setFirst = y_predRot[index]
    val = 180 - getAngleFrom(2, 9, 10, y_predRot, indices)
    if y_predRot[10][1] < y_predRot[9][1]:
        val = -1 * val
    checkerA = np.round(boundcheck(y_predRot, val, 130, 0, index, ymain, alpha))
    y_predRot[index] = fixer(temp, valCurrent, 130, 0, index, ymain, -alpha)
    val = 180 - getAngleFrom(2, 9, 10, y_predRot, indices)
    if y_predRot[10][1] < y_predRot[9][1]:
        val = -1 * val
    checkerB = np.round(boundcheck(y_predRot, val, 130, 0, index, ymain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[index] = setFirst
    if debug:
        print(realError)

    indices = [0, 1]
    y_predRot = aligner(y_predRot, 9, 10, 11)
    valCurrent = 180 - getAngleFrom(9, 10, 11, y_predRot, indices)
    if y_predRot[11][1] > y_predRot[10][1]:
        valCurrent = -1 * valCurrent
    index = [10, 11]
    temp = y_predRot.copy()
    y_predRot[index] = fixer(y_predRot, valCurrent, 90, -30, index, -zmain, alpha)
    setFirst = y_predRot[index]
    val = 180 - getAngleFrom(9, 10, 11, y_predRot, indices)
    if y_predRot[11][1] > y_predRot[10][1]:
        val = -1 * val
    checkerA = np.round(boundcheck(y_predRot, val, 90, -30, index, -zmain, alpha))
    y_predRot[index] = fixer(temp, valCurrent, 90, -30, index, -zmain, -alpha)
    val = 180 - getAngleFrom(9, 10, 11, y_predRot, indices)
    if y_predRot[11][1] > y_predRot[10][1]:
        val = -1 * val
    checkerB = np.round(boundcheck(y_predRot, val, 90, -30, index, -zmain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[index] = setFirst
    if debug:
        print(realError)

    # Finger 3 - Middle
    y_predRot = aligner(y_predRot, 0, 3, 12)
    valCurrent = 180 - getAngleFrom(0, 3, 12, y_predRot, indices)
    if y_predRot[12][1] < y_predRot[3][1]:
        valCurrent = -1 * valCurrent
    middle = [3, 12, 13, 14]
    temp = y_predRot.copy()
    y_predRot[middle] = fixer(y_predRot, valCurrent, 90, -40, middle, -zmain, alpha)
    setFirst = y_predRot[middle]
    val = 180 - getAngleFrom(0, 3, 12, y_predRot, indices)
    if y_predRot[12][1] < y_predRot[3][1]:
        val = -1 * val
    checkerA = np.round(boundcheck(y_predRot, val, 90, -40, middle, -zmain, alpha))
    y_predRot[middle] = fixer(temp, valCurrent, 90, -40, middle, -zmain, -alpha)
    val = 180 - getAngleFrom(0, 3, 12, y_predRot, indices)
    if y_predRot[12][1] < y_predRot[3][1]:
        val = -1 * val
    checkerB = np.round(boundcheck(y_predRot, val, 90, -40, middle, -zmain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[middle] = setFirst
    if debug:
        print(realError)

    y_predRot = aligner(y_predRot, 4, 3, 12)
    valCurrent = 90 - getAngleFrom(4, 3, 12, y_predRot, indices)
    temp = y_predRot.copy()
    y_predRot[middle] = fixer(y_predRot, valCurrent, 15, -15, middle, -zmain, alpha)
    setFirst = y_predRot[middle]
    val = 90 - getAngleFrom(4, 3, 12, y_predRot, indices)
    checkerA = np.round(boundcheck(y_predRot, val, 15, -15, middle, -zmain, alpha))
    y_predRot[middle] = fixer(temp, valCurrent, 15, -15, middle, -zmain, -alpha)
    val = 90 - getAngleFrom(4, 3, 12, y_predRot, indices)
    checkerB = np.round(boundcheck(y_predRot, val, 15, -15, middle, -zmain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[middle] = setFirst
    if debug:
        print(realError)

    indices = [0, 2]
    y_predRot = aligner(y_predRot, 3, 12, 13)
    valCurrent = 180 - getAngleFrom(3, 12, 13, y_predRot, indices)
    if y_predRot[13][1] < y_predRot[12][1]:
        valCurrent = -1 * valCurrent
    middle = [12, 13, 14]
    temp = y_predRot.copy()
    y_predRot[middle] = fixer(y_predRot, valCurrent, 130, 0, middle, ymain, alpha)
    setFirst = y_predRot[middle]
    val = 180 - getAngleFrom(3, 12, 13, y_predRot, indices)
    if y_predRot[13][1] < y_predRot[12][1]:
        val = -1 * val
    checkerA = np.round(boundcheck(y_predRot, val, 130, 0, middle, ymain, alpha))
    y_predRot[middle] = fixer(temp, valCurrent, 130, 0, middle, ymain, -alpha)
    val = 180 - getAngleFrom(3, 12, 13, y_predRot, indices)
    if y_predRot[13][1] < y_predRot[12][1]:
        val = -1 * val
    checkerB = np.round(boundcheck(y_predRot, val, 130, 0, middle, ymain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[middle] = setFirst
    if debug:
        print(realError)

    indices = [0, 1]
    y_predRot = aligner(y_predRot, 12, 13, 14)
    valCurrent = 180 - getAngleFrom(12, 13, 14, y_predRot, indices)
    if y_predRot[14][1] < y_predRot[13][1]:
        valCurrent = -1 * valCurrent
    middle = [13, 14]
    temp = y_predRot.copy()
    y_predRot[middle] = fixer(y_predRot, valCurrent, 90, -30, middle, ymain, alpha)
    setFirst = y_predRot[middle]
    val = 180 - getAngleFrom(12, 13, 14, y_predRot, indices)
    if y_predRot[14][1] < y_predRot[13][1]:
        val = -1 * val
    checkerA = np.round(boundcheck(y_predRot, val, 90, -30, middle, ymain, alpha))
    y_predRot[middle] = fixer(temp, valCurrent, 90, -30, middle, ymain, -alpha)
    val = 180 - getAngleFrom(12, 13, 14, y_predRot, indices)
    if y_predRot[14][1] < y_predRot[13][1]:
        val = -1 * val
    checkerB = np.round(boundcheck(y_predRot, val, 90, -30, middle, ymain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[middle] = setFirst
    if debug:
        print(realError)

    # Finger 4 - Ring
    y_predRot = aligner(y_predRot, 0, 4, 15)
    valCurrent = 180 - getAngleFrom(0, 4, 15, y_predRot, indices)
    if y_predRot[15][1] < y_predRot[4][1]:
        valCurrent = -1 * valCurrent
    ring = [4, 15, 16, 17]
    temp = y_predRot.copy()
    y_predRot[ring] = fixer(y_predRot, valCurrent, 90, -40, ring, -zmain, alpha)
    setFirst = y_predRot[ring]
    val = 180 - getAngleFrom(0, 4, 15, y_predRot, indices)
    if y_predRot[15][1] < y_predRot[4][1]:
        val = -1 * val
    checkerA = np.round(boundcheck(y_predRot, val, 90, -40, ring, -zmain, alpha))
    y_predRot[ring] = fixer(temp, valCurrent, 90, -40, ring, -zmain, -alpha)
    val = 180 - getAngleFrom(0, 4, 15, y_predRot, indices)
    if y_predRot[15][1] < y_predRot[4][1]:
        val = -1 * val
    checkerB = np.round(boundcheck(y_predRot, val, 90, -40, ring, -zmain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[ring] = setFirst
    if debug:
        print(realError)

    y_predRot = aligner(y_predRot, 5, 4, 15)
    val1 = 90 - getAngleFrom(5, 4, 15, y_predRot, indices)
    temp = y_predRot.copy()
    y_predRot[ring] = fixer(y_predRot, val1, 15, -15, ring, -zmain, alpha)
    setFirst = y_predRot[ring]
    val = 90 - getAngleFrom(5, 4, 15, y_predRot, indices)
    checkerA = np.round(boundcheck(y_predRot, val, 15, -15, ring, -zmain, alpha))
    y_predRot[ring] = fixer(temp, val1, 15, -15, ring, -zmain, -alpha)
    val = 90 - getAngleFrom(5, 4, 15, y_predRot, indices)
    checkerB = np.round(boundcheck(y_predRot, val, 15, -15, ring, -zmain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[ring] = setFirst
    if debug:
        print(realError)

    indices = [0, 2]
    y_predRot = aligner(y_predRot, 4, 15, 16)
    val1 = 180 - getAngleFrom(4, 15, 16, y_predRot, indices)
    if y_predRot[16][1] < y_predRot[15][1]:
        val1 = -1 * val1
    ring = [15, 16, 17]
    temp = y_predRot.copy()
    y_predRot[ring] = fixer(y_predRot, val1, 130, 0, ring, ymain, alpha)
    setFirst = y_predRot[ring]
    val = 180 - getAngleFrom(4, 15, 16, y_predRot, indices)
    if y_predRot[16][1] < y_predRot[15][1]:
        val = -1 * val
    checkerA = np.round(boundcheck(y_predRot, val, 130, 0, ring, ymain, alpha))
    y_predRot[ring] = fixer(temp, val1, 130, 0, ring, ymain, -alpha)
    val = 180 - getAngleFrom(4, 15, 16, y_predRot, indices)
    if y_predRot[16][1] < y_predRot[15][1]:
        val = -1 * val
    checkerB = np.round(boundcheck(y_predRot, val, 130, 0, ring, ymain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[ring] = setFirst
    if debug:
        print(realError)

    indices = [0, 1]
    y_predRot = aligner(y_predRot, 15, 16, 17)
    val1 = 180 - getAngleFrom(15, 16, 17, y_predRot, indices)
    if y_predRot[17][1] > y_predRot[16][1]:  # greater
        val1 = -1 * val1
    ring = [16, 17]
    temp = y_predRot.copy()
    y_predRot[ring] = fixer(y_predRot, val1, 90, -30, ring, zmain, alpha)
    setFirst = y_predRot[ring]
    val = 180 - getAngleFrom(15, 16, 17, y_predRot, indices)
    if y_predRot[17][1] > y_predRot[16][1]:  # greater
        val = -1 * val
    checkerA = np.round(boundcheck(y_predRot, val, 90, -30, ring, zmain, alpha))
    y_predRot[ring] = fixer(temp, val1, 90, -30, ring, zmain, -alpha)
    val = 180 - getAngleFrom(15, 16, 17, y_predRot, indices)
    if y_predRot[17][1] > y_predRot[16][1]:  # greater
        val = -1 * val
    checkerB = np.round(boundcheck(y_predRot, val, 90, -30, ring, zmain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[ring] = setFirst
    if debug:
        print(realError)

    # Finger 5 Pinky
    y_predRot = aligner(y_predRot, 0, 5, 18)
    val1 = 180 - getAngleFrom(0, 5, 18, y_predRot, indices)
    if y_predRot[18][1] < y_predRot[5][1]:
        val1 = -1 * val1
    pinky = [5, 18, 19, 20]
    temp = y_predRot.copy()
    y_predRot[pinky] = fixer(y_predRot, val1, 90, -40, pinky, -zmain, alpha)
    setFirst = y_predRot[pinky]
    val = 180 - getAngleFrom(0, 5, 18, y_predRot, indices)
    if y_predRot[18][1] < y_predRot[5][1]:
        val = -1 * val
    checkerA = np.round(boundcheck(y_predRot, val, 90, -40, pinky, -zmain, alpha))
    y_predRot[pinky] = fixer(temp, val1, 90, -40, pinky, -zmain, -alpha)
    val = 180 - getAngleFrom(0, 5, 18, y_predRot, indices)
    if y_predRot[18][1] < y_predRot[5][1]:
        val = -1 * val
    checkerB = np.round(boundcheck(y_predRot, val, 90, -40, pinky, -zmain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[pinky] = setFirst
    if debug:
        print(realError)

    y_predRot = aligner(y_predRot, 4, 5, 18)
    val1 = getAngleFrom(4, 5, 18, y_predRot, indices) - 90
    temp = y_predRot.copy()
    y_predRot[pinky] = fixer(y_predRot, val1, 15, -15, pinky, zmain, alpha)
    setFirst = y_predRot[pinky]
    val = getAngleFrom(4, 5, 18, y_predRot, indices) - 90
    checkerA = np.round(boundcheck(y_predRot, val, 15, -15, pinky, zmain, alpha))
    y_predRot[pinky] = fixer(temp, val1, 15, -15, pinky, zmain, -alpha)
    val = getAngleFrom(4, 5, 18, y_predRot, indices) - 90
    checkerB = np.round(boundcheck(y_predRot, val, 15, -15, pinky, zmain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[pinky] = setFirst
    if debug:
        print(realError)

    indices = [0, 1]
    y_predRot = aligner(y_predRot, 5, 18, 19)
    val1 = 180 - getAngleFrom(5, 18, 19, y_predRot, indices)
    if y_predRot[19][1] < y_predRot[18][1]:
        val1 = -1 * val1
    pinky = [18, 19, 20]
    temp = y_predRot.copy()
    y_predRot[pinky] = fixer(y_predRot, val1, 130, 0, pinky, ymain, alpha)
    setFirst = y_predRot[pinky]
    val = 180 - getAngleFrom(5, 18, 19, y_predRot, indices)
    if y_predRot[19][1] < y_predRot[18][1]:
        val = -1 * val
    checkerA = np.round(boundcheck(y_predRot, val, 130, 0, pinky, ymain, alpha))
    y_predRot[pinky] = fixer(temp, val1, 130, 0, pinky, ymain, -alpha)
    val = 180 - getAngleFrom(5, 18, 19, y_predRot, indices)
    if y_predRot[19][1] < y_predRot[18][1]:
        val = -1 * val
    checkerB = np.round(boundcheck(y_predRot, val, 130, 0, pinky, ymain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[pinky] = setFirst
    if debug:
        print(realError)

    indices = [0, 1]
    y_predRot = aligner(y_predRot, 18, 19, 20)
    val1 = 180 - getAngleFrom(18, 19, 20, y_predRot, indices)
    if y_predRot[20][1] > y_predRot[19][1]:
        val1 = -1 * val1
    pinky = [19, 20]
    temp = y_predRot.copy()
    y_predRot[pinky] = fixer(y_predRot, val1, 90, -30, pinky, zmain, alpha)
    setFirst = y_predRot[pinky]
    val = 180 - getAngleFrom(18, 19, 20, y_predRot, indices)
    if y_predRot[20][1] > y_predRot[19][1]:
        val = -1 * val
    checkerA = np.round(boundcheck(y_predRot, val, 90, -30, pinky, zmain, alpha))
    y_predRot[pinky] = fixer(temp, val1, 90, -30, pinky, zmain, -alpha)
    val = 180 - getAngleFrom(18, 19, 20, y_predRot, indices)
    if y_predRot[20][1] > y_predRot[19][1]:
        val = -1 * val
    checkerB = np.round(boundcheck(y_predRot, val, 90, -30, pinky, zmain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[pinky] = setFirst
    if debug:
        print(realError)

    # Match former alignment
    translate = y_values[0] - y_predRot[0]
    y_predRot = y_predRot + translate

    v1 = y_predRot[3] - y_predRot[0]
    v2 = y_values[3] - y_predRot[0]
    rad = angle_between(v1, v2)
    d = np.degrees(rad)
    axisPoint = np.cross(v1, v2)

    test = PointRotate3D(y_predRot[0],
                         y_predRot[0] + unit_vector(axisPoint),
                         y_predRot[3],
                         np.radians(d))
    values = np.zeros((21, 3))

    if np.linalg.norm(np.round(y_values[3] - test)) == 0:
        for j in range(0, 21):
            values[j, :] = PointRotate3D(y_predRot[0],
                                         y_predRot[0] + unit_vector(axisPoint),
                                         y_predRot[j],
                                         np.radians(d))
    else:
        for j in range(0, 21):
            deg = 360 - d
            values[j, :] = PointRotate3D(y_predRot[0],
                                         y_predRot[0] + unit_vector(axisPoint),
                                         y_predRot[j],
                                         np.radians(deg))

    y_predRot = values

    # Final Rotation
    def getProjpoint(point, orig, normal):
        v = point - orig
        dist = np.dot(v, normal)
        projected_point = point - dist * normal
        return projected_point

    norm = unit_vector(y_predRot[3] - y_predRot[0])
    a = getProjpoint(y_predRot[4], y_predRot[0], norm)
    b = getProjpoint(y_values[4], y_predRot[0], norm)
    v1 = a - y_predRot[0]
    v2 = b - y_predRot[0]
    rad1 = angle_between(v1, v2)
    d = np.degrees(rad1)

    test = PointRotate3D(y_predRot[0],
                         y_predRot[3],
                         y_predRot[4],
                         np.radians(d))

    values = np.zeros((21, 3))
    c1 = np.linalg.norm(np.round(y_values[4] - test))

    if c1 == 0:
        for j in range(0, 21):
            values[j, :] = PointRotate3D(y_predRot[0],
                                         y_predRot[3],
                                         y_predRot[j],
                                         np.radians(d))
    else:
        for j in range(0, 21):
            deg1 = 360 - d
            values[j, :] = PointRotate3D(y_predRot[0],
                                         y_predRot[3],
                                         y_predRot[j],
                                         np.radians(deg1))
    y_predRot = values
    return y_predRot, realError
