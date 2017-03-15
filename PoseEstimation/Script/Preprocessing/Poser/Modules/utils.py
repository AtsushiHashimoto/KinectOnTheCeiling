
# -*- coding: utf8 -*-

import poser
import os
import glob
import random
from math import sin, cos, tan, radians


__all__ = ["ps", "enumerate_poses", "make_discrete_regions", "cal_fig_params", "cal_camera_params",
           "set_fig_params", "set_camera_params", "save_param"]


def ps():
    return os.system("ps -p %d u" % os.getpid())


def enumerate_poses(bvh_path):
    motions = glob.glob(bvh_path+"*/*.bvh")
    motion_files = []
    frame_idx_in_file = []
    for motion in motions:
        for line in open(motion):
            if "Frames:" in line:
                params = line.split()
                n_frames = int(params[1])
                motion_files.extend([motion for x in range(n_frames)])
                frame_idx_in_file.extend([x + 1 for x in range(n_frames)])
                break

    return motion_files, frame_idx_in_file


def make_discrete_regions(intervals):
    v_interval, h_interval = intervals
    return [[v_phi, h_phi, v_phi + v_interval, h_phi + h_interval]
            for v_phi in range(0, 90, v_interval) for h_phi in range(-35, 35, h_interval)]


def cal_fig_params(fig_ave_height, fig_height_var, motion_files, frame_idx_in_file, fig_name):

    param_dict = {}
    if fig_name == "female":
        figure_original_height = 173
    elif fig_name == "male":
        figure_original_height = 176
    else:
        raise ValueError("invalid figure_name")

    fs = random.gauss(fig_ave_height / figure_original_height, fig_height_var / figure_original_height)
    fh = figure_original_height * fs  # figure height． フィギュアの大きさは170cmであるとした．
    param_dict["Figure Height"] = fh

    fr = random.uniform(-120, 120)  # figure rotation
    param_dict["Figure Rotation"] = fr

    n_poses = len(motion_files)
    frame = random.randint(0, n_poses - 1)
    param_dict["BVH File Name"] = "/".join(motion_files[frame].split("/")[-2:])
    param_dict["Frame No."] = frame_idx_in_file[frame]

    xTran = 0
    zTran = 0
    param_dict["Figure xTran"] = xTran
    param_dict["Figure zTran"] = zTran

    return param_dict


def cal_camera_params(fh, discrete_region):

    param_dict = {}

    h_fov = 35
    v_fov = 30.1077034261

    v_rcl, h_rcl = [random.uniform(discrete_region[0], discrete_region[2]),
                    max(min(random.uniform(discrete_region[1], discrete_region[3]),  27), -27)]
    r_v_rcl, r_h_rcl = radians(v_rcl), radians(h_rcl)

    cp = max(random.uniform(v_rcl - v_fov + 20, v_rcl + v_fov - 20), 0)  # camera pitch
    r_cp = radians(cp)
    param_dict["Camera Pitch"] = cp

    ch = random.uniform(200, 300)  # camera height
    z_from_fig = min((ch - fh / 2) / sin(r_v_rcl), 350)

    dollyZ = z_from_fig * cos(r_v_rcl - r_cp) + (fh / 2) * sin(r_cp) - 287  # 287はPoser内でz方向に勝手に動かされるoffset
    param_dict["Camera DollyZ"] = dollyZ

    dollyY = z_from_fig * sin(r_v_rcl - r_cp) + (fh / 2) * cos(r_cp)
    param_dict["Camera DollyY"] = dollyY

    dollyX = z_from_fig * tan(r_h_rcl)
    param_dict["Camera DollyX"] = dollyX

    return param_dict


def set_fig_params(scene, param_dict, bvh_path, fig_name):

    poser_unit = 262.128 # cm / POSER_UNIT
    if fig_name == "female":
        figure_original_height = 173
    elif fig_name == "male":
        figure_original_height = 176
    else:
        raise ValueError("invalid figure_name")

    scene.SetFrame(0)
    im = scene.ImExporter()
    im_option = im.ImportOptions("bvh", None)
    im_option[poser.kImOptCodeARMALIGNMENTAXIS] = 1  # x軸に沿って関節を整列
    im_option[poser.kImOptCodeAUTOSCALE] = 1  # BVH内の関節相対位置OFFSETをフィギュアに合わせて拡大縮小
    im.Import("bvh", None, bvh_path + param_dict["BVH File Name"], im_option)
    scene.SetFrame(param_dict["Frame No."] - 1)

    figure = scene.CurrentFigure()
    body = figure.Actors()[0]
    body.SetParameter("xrot", 0)
    body.SetParameter("yrot", param_dict["Figure Rotation"])
    body.SetParameter("zrot", 0)
    body.SetParameter("scale", param_dict["Figure Height"] / figure_original_height)
    body.SetParameter("xtran", param_dict["Figure xTran"] / poser_unit)
    body.SetParameter("ytran", 0)
    body.SetParameter("ztran", param_dict["Figure zTran"] / poser_unit)


def set_camera_params(scene, param_dict):

    poser_unit = 262.128  # cm / POSER_UNIT

    camera = scene.Cameras()[0]  # メインカメラを選択
    camera.SetParameter("focal", 19.6707178338) # 焦点距離を設定

    camera.SetParameter("pitch", -param_dict["Camera Pitch"])  # 仰角が正．俯角が負．
    camera.SetParameter("dollyZ", param_dict["Camera DollyZ"] / poser_unit)
    camera.SetParameter("dollyY", param_dict["Camera DollyY"] / poser_unit)
    camera.SetParameter("dollyX", param_dict["Camera DollyX"] / poser_unit)


def save_param(parm_path, param_dict):

    param_names = ["Figure Name", "BVH File Name", "Frame No.",
                   "Camera Pitch", "Camera DollyZ", "Camera DollyY", "Camera DollyX",
                   "Figure Rotation", "Figure Height", "Figure xTran", "Figure zTran"]
    param_dict["Camera DollyZ"] += 287

    with open(parm_path, "w") as fout:
        for param_name in param_names:
            fout.write("%s:%s\n" % (param_name, str(param_dict[param_name])))
