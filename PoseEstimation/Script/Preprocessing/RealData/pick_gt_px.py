#!/usr/local/bin/python3
# -*- coding=utf8 -*-

import cv2, argparse, os
import numpy as np
import pandas as pd


def get_args():

    p = argparse.ArgumentParser()
    p.add_argument("-d", "--data_path", type=str)
    args = p.parse_args()

    return args


class PointList:

    def __init__(self, npoints):

        self.npoints = npoints
        self.ptlist = np.empty((npoints, 2), dtype=int)
        self.pos = 0

    def add(self, v, h):

        if self.pos < self.npoints:
            self.ptlist[self.pos, :] = [v, h]
            self.pos += 1
            return True

        return False


def onMouse(event, v, h, flag, params):

    wname, img, ptlist, ratios = params

    if event == cv2.EVENT_MOUSEMOVE:
        img2 = np.copy(img)
        height, width = img2.shape[0], img2.shape[1]
        cv2.line(img2, (v, 0), (v, height - 1), (255, 0, 0))
        cv2.line(img2, (0, h), (width - 1, h), (255, 0, 0))
        cv2.imshow(wname, img2)

    if event == cv2.EVENT_LBUTTONDOWN:
        v = int(v / ratios[0])
        h = int(h / ratios[1])
        if ptlist.add(v, h):
            print('[%d] ( %d, %d )' % (ptlist.pos - 1, v, h))
            if ptlist.pos == ptlist.npoints:
                print('All points have selected.  Press ESC-key.')
        else:
            print('All points have selected.  Press ESC-key.')
            

def main():

    args = get_args()

    wname = "棚の位置アノテーション"
    RH, RV = 3, 3   # reduction ratio of images
    n_points = 1

    drawer_img_path = args.data_path
    save_filename = "%s6drawers_gt_px.csv" % drawer_img_path
    if os.path.exists(save_filename):
        raise IOError("Already existing %s." % save_filename)

    gt_pxs = []

    for d in [0, 10, 11, 21, 22, 32]:
        img_name = "%sdrawer_%d.png" % (drawer_img_path, d)

        print('##### %s' % img_name)
        img_raw = cv2.imread(img_name)
        height, width, channel = img_raw.shape
        img = cv2.resize(img_raw, (width * RV, height * RH))
        ptlist = PointList(n_points)
        cv2.namedWindow(wname)
        cv2.setMouseCallback(wname, onMouse, [wname, img, ptlist, (RV, RH)])
        cv2.imshow(wname, img)

        while cv2.waitKey(0) != 27:
            pass

        gt_pxs.append(ptlist.ptlist[0])
        cv2.destroyAllWindows()

    pd.DataFrame(gt_pxs, dtype=np.int32).to_csv(save_filename, header=False, index=False)

if __name__ == "__main__":
    main()
