# -*- coding: utf-8 -*-

import cv2, glob
import numpy as np
import pandas as pd
from os import path
from math import isnan
from sklearn.metrics.pairwise import euclidean_distances
from JPP_precision import load_JPP_ply
from Modules.utils import get_parameter, get_args, figure_disappears, enum_test_files
from Modules.features_labels import make_labels
from Modules.coordinate_conversion import project_point_cloud


def make_ground_truth(test_filename):

    n_joints = 19
    ground_truth = np.ones((n_joints, 2))

    label_img = cv2.imread("%s.png" % test_filename)[:, :, :3][:, :, ::-1]

    label_array = make_labels(label_img).reshape(label_img.shape[:2])
    parts2joints_map = np.array((0, 0, 0, 0, 1, 2, 2, 3, 3, 4, 5, 18, 18, 18, 18, 6, 7, 8, 9, 10, 11, 18, 18, 18, 18, 12, 13, 14, 15, 16, 17, 18))

    for j in range(n_joints):
        ground_truth[j, :] = np.mean(np.array(np.where(parts2joints_map[label_array] == j)), axis=1)

    return ground_truth[:-1, :]


def JPP_precision():

    args = get_args()

    discr_setting_type = args.discr_setting_type
    num_train_images = args.n_train_images
    data_path = args.data_path
    jpp_path = data_path + "Main/JointPositionPrediction/"
    jpp_gt_path = jpp_path + "GroundTruth/"
    jpp_out_path = jpp_path + "Output/"
    eval_path = jpp_path + "Evaluation/"
    test_path = args.test_path

    n_test_images = args.n_test_images

    device = "Kinect" if "SyntheticImages" in test_path else "Xtion"

    target_joint_names = ["Head", "neck", "Chest", "Waist",
                          "rShoulder", "lShoulder", "rElbow", "lElbow", "rWrist", "lWrist", "rHand", "lHand",
                          "rKnee", "lKnee", "rAnkle", "lAnkle", "rFoot", "lFoot"]
    n_joints = len(target_joint_names)

    test_filenames = enum_test_files(data_path, args.test_path, n_test_images)

    setting_str = "_" + str(num_train_images) + ("_%s" % discr_setting_type if discr_setting_type else "")

    average_error_path = eval_path + "JPP_average_error_px" + setting_str + ".csv"
    sum_prediction_error = np.zeros((n_joints+1,))

    for test_filename in test_filenames:

        test_filename_id = "/".join(test_filename.split("/")[-2:])
        print(test_filename_id)
        test_JPP_path = jpp_out_path + test_filename_id + setting_str + "_JPP.ply"
        test_gt_path = jpp_gt_path + test_filename_id + "_px_gt.csv"
        error_path = eval_path + test_filename_id + setting_str + "_JPP_error_px.csv"

        if path.exists(test_gt_path):
            gt_joint_positions = np.array(pd.read_csv(test_gt_path, header=None))
        else:
            gt_joint_positions = make_ground_truth(test_filename)

        joint_positions_3d = load_JPP_ply(test_JPP_path)
        visible_joints = []
        for j, joint_position in enumerate(joint_positions_3d):
            if joint_position != (0, 0):
                visible_joints.append(j)
        visible_joints = np.array(visible_joints)
        depth_img = cv2.imread(test_filename + " Z.png", flags=0)
        params = get_parameter(test_filename + "_param")
        _, joint_positions_2d = project_point_cloud(joint_positions_3d, depth_img, visible_joints, device)
        joint_positions_2d = np.array(joint_positions_2d).transpose()

        error_per_joint = np.zeros((18,))
        for j, (gt, p) in enumerate(zip(gt_joint_positions, joint_positions_2d)):
            if ((not isnan(gt[0])) and (not isnan(gt[1]))) and (p[0] != 0 or p[1] != 0):
                error_per_joint[j] = euclidean_distances(gt.reshape((1, -1)), p.reshape((1, -1))) * joint_positions_3d[j, 2] / 200.
            elif (isnan(gt[0]) and isnan(gt[1])) and (p[0] == 0 and p[1] == 0):
                error_per_joint[j] = np.nan
            else:
                error_per_joint[j] = 20 * joint_positions_3d[j, 2] / 200.

        mean_error = np.nanmean(error_per_joint)
        prediction_error = np.r_[error_per_joint, mean_error]
        sum_prediction_error += prediction_error

        pd.DataFrame(prediction_error, index=target_joint_names+["Mean"]).to_csv(error_path, header=False)
        print("\tMean Error is %f" % mean_error)

    mean_errors = sum_prediction_error / n_test_images
    pd.DataFrame(mean_errors, index=target_joint_names+["Mean"]).to_csv(average_error_path, header=False)
    print("Mean error is %f" % mean_errors[-1])


if __name__ == "__main__":
    JPP_precision()
