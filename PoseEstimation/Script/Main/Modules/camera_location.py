# -*- coding: utf-8 -*-

import glob, argparse, cv2
import numpy as np
import pandas as pd
from math import atan2, cos, sin, tan, radians, degrees
from .utils import get_parameter
from .coordinate_conversion import make_point_cloud, human_loc2camera_loc, rel2abs


def _cal_human_center_from_camera(params):

    x_tran, z_tran = float(params["Figure xTran"]), float(params["Figure zTran"])
    figure_height = float(params["Figure Height"])
    dollyX, dollyY, dollyZ = float(params["Camera DollyX"]), float(params["Camera DollyY"]), float(params["Camera DollyZ"]) + 287
    camera_pitch = radians(params["Camera Pitch"])
    x = dollyX - x_tran
    y = - dollyY - z_tran * sin(camera_pitch) + (figure_height / 2) * cos(camera_pitch)
    z = dollyZ - (figure_height / 2) * sin(camera_pitch) - z_tran * cos(camera_pitch)

    return np.array([x, y, z])


def cal_camera_location(param_filename):

    param_dict = get_parameter(param_filename)

    c_pitch = param_dict["Camera Pitch"]
    x, y, z = _cal_human_center_from_camera(param_dict)
    h_angle = degrees(atan2(x, z))
    v_angle = c_pitch - degrees(atan2(y, z))

    return v_angle, h_angle


def estimate_camera_location(fname, cparam_fname):

    bg_fname = "/".join(fname.split("/")[:-2]) + "/background.png"
    depth_frame = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    bg_frame = cv2.imread(bg_fname, cv2.IMREAD_GRAYSCALE)

    params = get_parameter(cparam_fname)
    human_center = _estimate_human_center(depth_frame, bg_frame)

    cpitch = params["Camera Pitch"]
    camera_location = human_loc2camera_loc(human_center, radians(cpitch))

    return camera_location


def _estimate_human_center(frame, bg_frame):

    point_cloud = make_point_cloud(frame)
    raw_array = point_cloud[:, :, 2]
    bg_frame = rel2abs(bg_frame)
    sub_array = np.abs(raw_array - bg_frame)
    fg_mask = np.zeros(raw_array.shape, dtype=np.uint8)
    fg_mask[np.where(sub_array > 10)] = 1
    fg_mask[np.where(raw_array == 0)] = 0

    neighbor8 = np.ones((3, 3), dtype=np.uint8)
    eroded_array = cv2.erode(fg_mask, neighbor8, iterations=4)
    processed_fg_mask = cv2.dilate(eroded_array, neighbor8, iterations=4)

    output = cv2.connectedComponentsWithStats(processed_fg_mask.astype(np.uint8), connectivity=8)
    human_mask = np.zeros(raw_array.shape, dtype=np.uint8)
    n_most_popular_label_px = 0
    human_label = 1
    for label in np.unique(output[1])[1:]:
        n_label_px = np.where(output[1]==label)[0].shape[0]
        if n_label_px > n_most_popular_label_px:
            human_label = label
            n_most_popular_label_px = n_label_px
    human_mask[np.where(output[1] == human_label)] = 1

    fg_px = np.where(human_mask == 1)

    human_center_in_px = np.mean(fg_px, axis=1).astype(np.int32)
    human_center = point_cloud[human_center_in_px[0], human_center_in_px[1]]

    return human_center


if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument("-d", "--data_path", type=str, default="../../../Data/")
    args = p.parse_args()

    bpc_path = args.data_path + "Main/BodyPartClassification/"
    filenames = glob.glob(bpc_path+"SyntheticImages/*male/*_param")

    def discretization_setting(discretization_type="uniform", intervals=list((10, 10)), manual_setting_filename=None):
        if discretization_type == "uniform":
            v_interval, h_interval = intervals
            discrete_regions = np.array([[v_phi, h_phi, v_phi + v_interval, h_phi + h_interval]
                                         for v_phi in range(0, 90, v_interval) for h_phi in range(-90, 90, h_interval)])
        elif discretization_type == "manual" and manual_setting_filename is not None:
            discrete_regions = np.array(pd.read_csv(manual_setting_filename, header=None), dtype=np.float32)
        else:
            raise ValueError(discretization_type, "Wrong discretization_type!")

        return discrete_regions

    discrete_regions = discretization_setting(discretization_type="manual", manual_setting_filename=bpc_path+"Intermediate/discretization_setting.csv")

    n_target_files = np.zeros((discrete_regions.shape[0],), dtype=int)
    for i, filename in enumerate(filenames):
        camera_location = cal_camera_location(filename)
        for j, s in enumerate(discrete_regions):
            if s[0] <= camera_location[0] < s[2] and s[1] <= camera_location[1] < s[3]:
                n_target_files[j] += 1
                break

    print(n_target_files)
    print("/%d" % np.sum(n_target_files))
