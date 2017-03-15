# -*- coding: utf-8 -*-

import glob, os, time, datetime, shutil, sys
import numpy as np
import pandas as pd
import multiprocessing as mp
from PIL import Image, ImageDraw
from divide_and_conquer_BPC import discretization_setting
from Modules.utils import get_parameter, get_args, proba_parts2joints, figure_disappears, bvh_exists, enum_test_files
from Modules.coordinate_conversion import point_cloud2hcc, make_point_cloud, project_point_cloud
from Modules.my_mean_shift import MeanShift
from sklearn.externals.joblib import Parallel
from Modules.camera_location import estimate_camera_location, cal_camera_location
from divide_and_conquer_BPC import _is_in_discr_region

Parallel.JOBLIB_START_METHOD = 'forkserver'


class JointPositionPrediction:

    def __init__(self, n_train_images=2000, full_rotation=False, compression_type="gzip",
                 discr_setting_type=None, discr_regions=None,
                 min_proba=0.14):
        self.n_train_images = n_train_images
        self.full_rotation = full_rotation
        self.compression_type = compression_type
        self.discr_setting_type = discr_setting_type
        self.discr_regions = discr_regions
        self.min_proba = min_proba
        self.setting_str = "_" + str(n_train_images)

        self.zeta = np.array([0.1, 0.07, 0.095, 0.095,
                              0.065, 0.065, 0.041, 0.041, 0.026, 0.026, 0.01, 0.01,
                              0.041, 0.041, 0.041, 0.041, 0.02, 0.02])  # meters

        self.joint_colors = np.array([(63,0,0), (127,255,0), (191,255,191), (127,255,127),
                                      (63,127,0), (0,191,63), (127,63,0), (0,63,127), (255,63,255), (63,255,255), (255,63,0), (0,63,255),
                                      (63,0,63), (63,0,127), (255,127,127), (63,255,63), (191,127,63), (63,63,0)])

        self.joint_connection = [[1, 0], [2, 1], [3, 2],
                                 [2, 4], [2, 5], [4, 6], [5, 7], [6, 8], [7, 9], [8, 10], [9, 11],
                                 [3, 12], [3, 13], [12, 14], [13, 15], [14, 16], [15, 17]]

        self.part_names = ["ruHead", "rwHead", "luHead", "lwHead", "neck", "rChest", "lChest", "rWaist", "lWaist", "rShoulder", "lShoulder",
                           "ruArm", "rwArm", "luArm", "lwArm", "rElbow", "lElbow", "rWrist", "lWrist", "rHand", "lHand", "ruLeg", "rwLeg",
                           "luLeg", "lwLeg", "rKnee", "lKnee", "rAnkle", "lAnkle", "rFoot", "lFoot", "Background"]

        self.joint_names = ["Head", "neck", "Chest", "Waist",
                            "rShoulder", "lShoulder", "rElbow", "lElbow", "rWrist", "lWrist", "rHand", "lHand",
                            "rKnee", "lKnee", "rAnkle", "lAnkle", "rFoot", "lFoot"]

    def _save_JPP_ply(self, test_JPP_path, joint_positions):

        ply_header_lines = ["ply", "format ascii 1.0",
                            "element vertex 18", "property float x", "property float y", "property float z",
                            "property uchar red", "property uchar green", "property uchar blue",
                            "element edge 17", "property int vertex1", "property int vertex2",
                            "property uchar red", "property uchar green", "property uchar blue",
                            "end_header"]

        joint_positions = np.c_[joint_positions, self.joint_colors]
        edges = np.c_[np.array(self.joint_connection), np.tile(np.array([0, 255, 0]), (17, 1))]

        with open(test_JPP_path, 'w') as ply_file:
            for h_line in ply_header_lines:
                ply_file.write(h_line+"\n")

            for joint_position in joint_positions:
                ply_file.write(" ".join([str(x) for x in list(joint_position)[:3]]) + " ")
                ply_file.write(" ".join([str(int(x)) for x in list(joint_position)[3:]]) + "\n")

            for edge in edges:
                ply_file.write(" ".join([str(x) for x in list(edge)]) + "\n")

    def predict(self, test_filename):

        start = time.time()

        main_path = "/".join(test_filename.split("/")[:-4]) + "/"
        bpc_path = main_path + "BodyPartClassification/"
        jpp_path = main_path + "JointPositionPrediction/"
        tmp = test_filename.split("/")[-2:]
        test_file_id = "/".join(tmp)
        tmp[0] += "_invalid"
        invalid_test_file_id = "/".join(tmp)
        discr_name = None

        if "SyntheticImages" in test_filename:
            synth_img_path = bpc_path + "SyntheticImages/"
            test_depth_image_path = synth_img_path + test_file_id + " Z.png"
            test_param_path = synth_img_path + test_file_id + "_param"
            polar, azim = cal_camera_location(test_param_path)
            device = "Kinect"
        elif "CapturedImages" in test_filename:
            cap_img_path = bpc_path + "CapturedImages/"
            test_depth_image_path = cap_img_path + test_file_id + ".png"
            test_param_path = cap_img_path + "camera_param"
            _, azim, polar = estimate_camera_location(test_filename + ".png", test_param_path)
            device = "Xtion"
        else:
            raise ValueError("Invalid directory name.")

        if self.discr_setting_type:

            discr_setting_path = bpc_path + "Intermediate/discretization_setting/"
            discr_setting_filename = "%s%s.csv" % (discr_setting_path, self.discr_setting_type)
            if self.discr_regions is None:
                self.discr_regions = discretization_setting(discr_setting_filename)
            for d in self.discr_regions:
                discr_name = str(d)
                test_BPC_proba_path = "%sOutput/%s%s%s_nb_BPC_proba.gz" % (bpc_path, test_file_id,
                                                                           self.setting_str,
                                                                           ("_%s" % discr_name) if self.discr_setting_type is not None else "")
                if os.path.exists(test_BPC_proba_path) and _is_in_discr_region(d, filename_base=("SyntheticImages" in test_filename),
                                                                               camera_location=[polar, azim], filename=test_filename):
                    break

            test_JPP_path = "%sOutput/%s%s_%s_nb_JPP.ply" % (jpp_path, test_file_id, self.setting_str, discr_name)
            test_JPP_image_path = "%sOutput/%s%s_%s_nb_JPP.png" % (jpp_path, test_file_id, self.setting_str, discr_name)
            invalid_test_JPP_path = "%sOutput/%s%s_%s_nb_JPP.ply" % (jpp_path, invalid_test_file_id, self.setting_str, discr_name)
            invalid_test_JPP_image_path = "%sOutput/%s%s_%s_nb_JPP.png" % (jpp_path, invalid_test_file_id, self.setting_str, discr_name)

        else:

            test_BPC_proba_path = "%sOutput/%s%s_nb_BPC_proba.gz" % (bpc_path, test_file_id, self.setting_str)
            test_JPP_path = "%sOutput/%s%s_nb_JPP.ply" % (jpp_path, test_file_id, self.setting_str)
            test_JPP_image_path = "%sOutput/%s%s_nb_JPP.png" % (jpp_path, test_file_id, self.setting_str)
            invalid_test_JPP_path = "%sOutput/%s%s_nb_JPP.ply" % (jpp_path, invalid_test_file_id, self.setting_str)
            invalid_test_JPP_image_path = "%sOutput/%s%s_nb_JPP.png" % (jpp_path, invalid_test_file_id, self.setting_str)

        if not os.path.exists(test_BPC_proba_path):
            print("Doesn't exist!!!")
            return

        test_point_cloud_path = jpp_path + "ByProduct/" + test_file_id + "_points.txt"

        newest_edit_date = datetime.datetime(2017, 2, 1, 0, 0, 0)
        if os.path.exists(test_JPP_path) and os.path.exists(test_JPP_image_path):
            JPP_out_stat = os.stat(test_JPP_path)
            if newest_edit_date < datetime.datetime.fromtimestamp(JPP_out_stat.st_mtime):
                return
        elif os.path.exists(invalid_test_JPP_image_path) and os.path.exists(invalid_test_JPP_path):
            return

        # 人物領域を抽出

        test_target_pixels_path = bpc_path + "Intermediate/" + test_file_id + "_target_pixels.gz"
        target_pixels = np.array(pd.read_csv(test_target_pixels_path, header=None))
        v_min, h_min = np.min(target_pixels, axis=0)
        v_max, h_max = np.max(target_pixels, axis=0)
        n_px_within_rect = (v_max - v_min + 1) * (h_max - h_min + 1)
        target_rect = np.array([[v_min, h_min], [v_max, h_max]])

        # 人物領域中の画素を3D点群に変換．保存．
        depth_px = np.asarray(Image.open(test_depth_image_path))[:, :, 0]
        params = get_parameter(test_param_path)
        test_abs_depth_px = make_point_cloud(depth_px, params, target_rect)[v_min:v_max+1, h_min:h_max+1, :].reshape((n_px_within_rect, 3))
        pd.DataFrame(np.c_[test_abs_depth_px, np.ones((n_px_within_rect, 3)) * 255]).to_csv(test_point_cloud_path, sep=";", index=False, header=False)
        test_abs_depth_px /= 100  # オーバーフローを防ぐ為に単位をcm → m

        # 各点が各部位に属する確率を保持する行列を読み込み．
        n_parts = len(self.part_names)
        height, width = depth_px.shape
        part_proba = np.array(pd.read_csv(test_BPC_proba_path, compression=self.compression_type, header=None, dtype=np.float64))
        part_proba = part_proba.reshape((height, width, n_parts))
        part_proba = part_proba[v_min:v_max+1, h_min:h_max+1, :]
        part_proba = part_proba.reshape((n_px_within_rect, n_parts))
        joint_proba = proba_parts2joints(part_proba)

        # 関節位置をMeanShiftにより求める．
        n_joints = len(self.joint_names)
        joint_positions = np.zeros((n_joints, 3))
        visible_joints = np.empty((0,), dtype=np.int32)
        for j, joint_name in enumerate(self.joint_names):

            print("Predicting %s position..." % joint_name)
            target_idx = np.where(joint_proba[:, j] >= self.min_proba)
            if target_idx[0].shape[0] == 0:
                print("%s is invisible." % joint_name)
                continue

            weights = np.array([joint_proba[p, j] * test_abs_depth_px[p, 2]**2 for p in range(n_px_within_rect)], dtype=np.float64)
            n_jobs = int(mp.cpu_count() / 4)
            ms = MeanShift(bandwidth=0.065, n_jobs=n_jobs).fit(test_abs_depth_px[target_idx], weights=weights[target_idx])
            if ms.cluster_centers_ is None:
                return
            maximum_points = ms.cluster_centers_

            print("\t The number of maximum point: %d" % len(maximum_points))
            joint_positions[j, :] = maximum_points[0] + np.array([0, 0, self.zeta[j]])
            visible_joints = np.r_[visible_joints, j]

        joint_positions *= 100  # m → cm

        # 関節位置推定結果の描画
        depth_px = np.repeat(depth_px, 3).reshape((height, width, 3))
        depth_px, joint_pos_2d = project_point_cloud(joint_positions, depth_px, visible_joints, device)
        img = Image.fromarray(depth_px)
        draw = ImageDraw.Draw(img)
        for p, c in self.joint_connection:
            if p in visible_joints and c in visible_joints:
                draw.line((joint_pos_2d[1][p], joint_pos_2d[0][p], joint_pos_2d[1][c], joint_pos_2d[0][c]),
                          fill=tuple(self.joint_colors[c, :]))
        img.save(test_JPP_image_path)

        # 関節位置推定結果の保存
        if "SyntheticImages" in test_filename:
            joint_positions[visible_joints] = point_cloud2hcc(joint_positions[visible_joints], params)
        self._save_JPP_ply(test_JPP_path, joint_positions)

        print("Whole predicting time: %.7f" % (time.time() - start))

if __name__ == "__main__":

    args = get_args()

    jpp = JointPositionPrediction(n_train_images=args.n_train_images, full_rotation=args.full_rotation, compression_type="gzip",
                                  discr_setting_type=args.discr_setting_type, discr_regions=None,
                                  min_proba=0.14)

    test_filenames = enum_test_files(args.data_path, args.test_path, args.n_test_images)
    # 関節位置推定 Joint Position Prediction(JPP)
    print("====Joint Position Prediction====")
    if "CapturedVideos" in args.test_path:
        for i, test_filename in enumerate(test_filenames):
            test_file_id = "/".join(test_filename.split("/")[-2:])
            print("%d: %s" % (i, test_file_id))
            jpp.video_predict(test_filename)
    elif "SyntheticImages" in args.test_path or "CapturedImages" in args.test_path:
        for i, test_filename in enumerate(test_filenames):
            test_file_id = "/".join(test_filename.split("/")[-2:])
            print("%d: %s" % (i, test_file_id))
            jpp.predict(test_filename)
    else:
        raise ValueError("Invalid test file path.")
