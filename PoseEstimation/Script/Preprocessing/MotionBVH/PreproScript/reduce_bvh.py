# -*- coding: utf-8 -*-

import glob
from math import sqrt
import numpy as np

INTERVAL = 4
REDUCE_TYPE = 1 # 0: INTERVALフレームごと 1: 少なくとも1つの関節が5cm動くごと


def max_dist(pos1, pos2):

    max_dist = 0
    if len(pos1) != len(pos2):
        print("ERROR")
    for i in range(0, len(pos1), 3):
        pos_dif = pos1[i:i+3] - pos2[i:i+3]
        d = sqrt(np.dot(pos_dif, pos_dif))
        max_dist = max(d, max_dist)

    return max_dist


def pos_input(WCFile):

    pos_list = [np.array(tuple(float(p) for p in line.split())) for line in open(WCFile, 'r')]
    
    top = pos_list[0][1]
    bottom = top
    for i in range(len(pos_list[0])):
        if i % 3 == 1:
            top = max(top, pos_list[0][i])
            bottom = min(bottom, pos_list[0][i])
    height = top - bottom

    pos_list = tuple(pos * 1.7 / height for pos in pos_list)
    
    return pos_list


def analyze_position(WCFile, reduceType=0):
    ## データ入力
    posList= pos_input(WCFile)
    nNormPose = len(posList)

    ## 解析
    tmpPos = posList[0]
    outList = [0]
    if reduceType == 0:
        outList = (x for x in range(0, len(posList), INTERVAL))
    elif reduceType == 1:
        i = 1
        for pos in posList[1:]:
            if max_dist(pos, tmpPos) > 0.05:
                tmpPos = pos
                outList.append(i)
            i += 1
    nReducePose = len(outList)
    print(str(nNormPose)+" -> "+str(nReducePose))

    return outList


def reduce_bvh(inFile, WCFile, outFile):
    
    ## 削減対象決定
    outList = analyze_position(WCFile, REDUCE_TYPE)

    ## 削減
    fin = open(inFile, 'r')
    fout = open(outFile, 'w')

    motionflag = 0
    inPosList = []
    for inLine in fin:
        if "Frames:" in inLine:
            inLineItems = inLine.split()
            inLineItems[1] = str(len(outList))
            inLine = " ".join(inLineItems)+"\n"
            fout.write(inLine)
        elif "Frame Time:" in inLine:
            motionflag = 1
            fout.write(inLine)
        elif motionflag == 1:
            inPosList.append(inLine)
        else:
            fout.write(inLine)
    
    for i in outList:
        fout.write(inPosList[i])
    
    fout.close()

    WCLine = open(WCFile, 'r').readlines()

    fWC = open(WCFile, 'w')
    for i in outList:
        fWC.write(WCLine[i])    

    fWC.close()


if __name__ == '__main__':
    IMpath = "../Data/Intermediate/"
    inAppendix = "_norm.bvh"
    outAppendix = "_reduce.bvh"
    WCAppendix = "_WC"
    filenames = list(map(lambda x: x[len(IMpath):-len(WCAppendix)], glob.glob(IMpath+"*/*"+WCAppendix)))
    
    for filename in filenames:
    
        WCFile = IMpath + filename + WCAppendix
        inFile = IMpath + filename + inAppendix
        outFile = IMpath + filename + outAppendix
    
        ## データ削減
        reduce_bvh(inFile, WCFile, outFile)
