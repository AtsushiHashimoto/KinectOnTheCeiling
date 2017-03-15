# -*- coding: utf-8 -*-

import glob, os
from math import ceil, pi, sin, cos, atan2, degrees, radians
import PreproScript.convert2WC as convert
import numpy as np


def front_angle(Zrot, Yrot, Xrot):
    Zrot = radians(Zrot)
    Yrot = radians(Yrot)
    Xrot = radians(Xrot)
    Cz = cos(Zrot)
    Sz = sin(Zrot)
    Cy = cos(Yrot)
    Sy = sin(Yrot)
    Cx = cos(Xrot)
    Sx = sin(Xrot)
    # 各軸に関する回転行列
    Rz = np.matrix([[Cz, -Sz, 0], [Sz, Cz, 0], [0, 0, 1]])
    Ry = np.matrix([[Cy, 0, Sy], [0, 1, 0], [-Sy, 0, Cy]])
    Rx = np.matrix([[1, 0, 0], [0, Cx, -Sx], [0, Sx, Cx]])
    Rroot = Rz.dot(Ry.dot(Rx))
    testV = np.dot(Rroot, np.matrix([[0], [0], [1]]))

    return degrees(atan2(testV[0], testV[2]))


def normalize_bvh(inFile, outFile):
    motion = 0
    firstLine = True
    outDir = "/".join(outFile.split("/")[:-1])
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    fout = open(outFile, 'w')
    for line in open(inFile, 'r'):
        if motion == 0:
            if 'CHANNELS 6' in line:
                terms = line.split()
                terms[1] = '7'
                terms.insert(5, 'Yrotation')
                line = " ".join(terms)+"\n"
            elif 'Frame Time:' in line:
                motion = 1
            fout.write(line)
        else:
            parms = [float(p) for p in line.split()]
            parms.insert(3, -front_angle(parms[5], parms[4], parms[3]))
            if firstLine:
                jointTree = convert.makeJointTree(inFile)
                WCParms = convert.parms2WC(parms, jointTree)
                root_height = float(WCParms[1])
                firstLine = False

            parms[0] = 0.0
            parms[1] = parms[1] - root_height
            parms[2] = 0.0
            line = " ".join(str(p) for p in parms)+"\n"
            fout.write(line)

    fout.close()


if __name__ == '__main__':

    OGpath = "../Data/Original/"
    IMpath = "../Data/Intermediate/"
    inAppendix = ".bvh"
    outAppendix = "_norm.bvh"
    filenames = tuple(x[len(OGpath):-len(inAppendix)] for x in glob.glob(OGpath+"*/*"+inAppendix))
    
    for filename in filenames:
   
        inFile = OGpath + filename + inAppendix
        outFile = IMpath + filename + outAppendix
   
        # 画面中央に．
        normalize_bvh(inFile, outFile)
    
