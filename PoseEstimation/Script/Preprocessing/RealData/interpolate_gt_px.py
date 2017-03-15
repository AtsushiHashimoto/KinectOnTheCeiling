#!/usr/local/bin/python3

import cv2
import pandas as pd
import numpy as np
from pick_gt_px import get_args


def main():

    args = get_args()
    data_path = args.data_path
    drawers6_gt_fname = data_path + "6drawers_gt_px.csv"
    drawer0_fname = data_path + "drawer_0.png"
    gt_img_path = data_path + "ground_truth.png"
    gt_px_path = data_path + "ground_truth_px.csv"

    x = np.array(pd.read_csv(drawers6_gt_fname, header=None))

    pt_cols = np.array([[x[0, :], x[1, :]],
                        [x[2, :], x[3, :]],
                        [x[4, :], x[5, :]]])

    points = []
    for pt_col in pt_cols:
        for i in range(11):
            point = (pt_col[0] + (pt_col[1] - pt_col[0]) * i / 10).astype(np.int32)
            points.append(point)

    px = np.asarray(cv2.imread(drawer0_fname))
    for point in points:
        px[point[1] + 0, point[0] + 0, :] = np.array([0, 0, 255], dtype=np.uint8)
        px[point[1] + 1, point[0] + 0, :] = np.array([0, 0, 255], dtype=np.uint8)
        px[point[1] + 0, point[0] + 1, :] = np.array([0, 0, 255], dtype=np.uint8)
        px[point[1] - 0, point[0] - 1, :] = np.array([0, 0, 255], dtype=np.uint8)
        px[point[1] - 1, point[0] - 0, :] = np.array([0, 0, 255], dtype=np.uint8)

    cv2.imwrite(gt_img_path, px)

    pd.DataFrame(points).to_csv(gt_px_path, header=False, index=False)

if __name__ == "__main__":
    main()
