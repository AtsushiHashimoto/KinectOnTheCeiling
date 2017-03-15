
import cv2, time
import pandas as pd
import numpy as np
from math import atan2, tan, radians
from pick_gt_px import get_args


def make_point(target_pixel, px, v_center, h_center, v_fov, h_fov):

    v, h = target_pixel
    v_angle = atan2(v_center - v, v_center / tan(v_fov))
    h_angle = atan2(h - h_center, h_center / tan(h_fov))
    z = px[v, h] * 1000. / 255.

    return z * tan(h_angle), z * tan(v_angle), z


def main():

    args = get_args()
    data_path = args.data_path
    bg_img_fname = data_path + "bg_depth.png"
    gt_px_fname = data_path + "ground_truth_px.csv"
    gt_depth_fname = data_path + "gt_depth.csv"
    gt_fname = data_path + "ground_truth.csv"
    gt_img = data_path + "ground_truth.png"

    bg_frame = cv2.imread(bg_img_fname)
    gt_pxs = np.array(pd.read_csv(gt_px_fname, header=None))
    height, width, channel = bg_frame.shape
    print(height, width)

    ground_truth = np.zeros((33, 3))
    v_center = (height - 1) / 2
    h_center = (width - 1) / 2
    v_fov = radians(22.5)
    h_fov = radians(29)

    green = np.array([0, 255, 0])
    gt_depth = np.empty((gt_pxs.shape[0],))
    for i, gt_px in enumerate(gt_pxs):
        ground_truth[i, :] = make_point(tuple((gt_px[1], gt_px[0])), bg_frame[:, :, 0], v_center, h_center, v_fov, h_fov)
        gt_depth[i] = bg_frame[gt_px[1] + 0, gt_px[0] + 0, 0] * 1000. / 255
        bg_frame[gt_px[1] + 0, gt_px[0] + 0, :] = green
        bg_frame[gt_px[1] + 1, gt_px[0] + 0, :] = green
        bg_frame[gt_px[1] + 0, gt_px[0] + 1, :] = green
        bg_frame[gt_px[1] - 1, gt_px[0] - 0, :] = green
        bg_frame[gt_px[1] - 0, gt_px[0] - 1, :] = green

    cv2.imwrite(gt_img, bg_frame)
    pd.DataFrame(gt_depth).to_csv(gt_depth_fname, header=False, index=False)
    pd.DataFrame(ground_truth).to_csv(gt_fname, header=False, index=False)


if __name__ == "__main__":
    main()
