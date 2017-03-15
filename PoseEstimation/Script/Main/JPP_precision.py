# -*- coding: utf-8 -*-

import glob, os
import numpy as np
import pandas as pd
from os import path
from math import radians
from Modules.utils import get_parameter, get_args, figure_disappears, enum_test_files
from Modules.coordinate_conversion import point_cloud2hcc, rotation_matrix
from divide_and_conquer_BPC import discretization_setting


def make_JPP_ply(test_JPP_path, joint_positions):

    ply_header_lines = ["ply", "format ascii 1.0",
                        "element vertex 18", "property float x", "property float y", "property float z",
                        "property uchar red", "property uchar green", "property uchar blue",
                        "element edge 17", "property int vertex1", "property int vertex2",
                        "property uchar red", "property uchar green", "property uchar blue",
                        "end_header"]
    part_labels = np.array([(63,0,0), (127,255,0), (191,255,191), (127,255,127), (63,127,0), (0,191,63),
                            (127,63,0), (0,63,127), (255,63,255), (63,255,255), (255,63,0), (0,63,255),
                            (63,0,63), (63,0,127), (255,127,127), (63,255,63), (191,127,63), (63,63,0)])
    joint_connection = [[1, 0], [2, 1], [3, 2],
                        [2, 4], [2, 5], [4, 6], [5, 7], [6, 8], [7, 9], [8, 10], [9, 11],
                        [3, 12], [3, 13], [12, 14], [13, 15], [14, 16], [15, 17]]

    joint_positions = np.c_[joint_positions, part_labels]
    edges = np.c_[np.array(joint_connection), np.tile(np.array([0, 255, 0]), (17, 1))]

    with open(test_JPP_path, 'w') as ply_file:
        for h_line in ply_header_lines:
            ply_file.write(h_line+"\n")

        for joint_position in joint_positions:
            ply_file.write(" ".join([str(x) for x in list(joint_position)[:3]]) + " ")
            ply_file.write(" ".join([str(int(x)) for x in list(joint_position)[3:]]) + "\n")

        for edge in edges:
            ply_file.write(" ".join([str(x) for x in list(edge)]) + "\n")


def load_JPP_ply(ply_path):

    n_remaining_elements = 18
    vertex_flag = False
    joint_positions = []
    with open(ply_path) as ply_file:
        for line in ply_file:
            if "end_header" in line:
                vertex_flag = True
            elif vertex_flag and n_remaining_elements > 0:
                joint_positions.append(line.split()[:3])
                n_remaining_elements -= 1

    return np.array(joint_positions, dtype=np.float32)


def make_ground_truth(param_dict, wc_path, visible_joints):

    figure_name = param_dict["Figure Name"]
    bvh_file_id = param_dict["BVH File Name"]
    no_frame = int(param_dict["Frame No."] - 1)
    figure_rotation = param_dict["Figure Rotation"]
    if figure_name == "female":
        fig_height = param_dict["Figure Height"] / 170. * 173
    elif figure_name == "male":
        fig_height = param_dict["Figure Height"] / 170. * 176
    else:
        raise ValueError("Invalid figure name.")

    wc_filename = wc_path + bvh_file_id.replace(".bvh", "_" + figure_name + "_pos")
    wc_joint_positions = np.array(pd.read_csv(wc_filename, sep=" ", header=None, dtype=np.float32))

    fig_height_in_bvh = np.max(wc_joint_positions[0, 1::3]) - np.min(wc_joint_positions[0, 1::3])
    ground_truth_tmp = wc_joint_positions[no_frame, :].reshape((38, 3)) * (fig_height / fig_height_in_bvh)

#    t_pose_gt = wc_joint_positions[0, :].reshape((38, 3))
#    fig_height_in_bvh = (t_pose_gt[18, :] + t_pose_gt[19, :]) / 2. - (3 * t_pose_gt[4, :] + t_pose_gt[6, :]) / 4.
#    ground_truth_tmp = wc_joint_positions[no_frame, :].reshape((38, 3)) * (fig_height / fig_height_in_bvh)

    female_chest_offset = np.array([0, 0, 7])
    a = 2.5
    ground_truth = np.zeros((18, 3))
    ground_truth[0, :] = (ground_truth_tmp[18, :] + ground_truth_tmp[19, :]) / 2  # Head
    ground_truth[1, :] = (ground_truth_tmp[16, :] + ground_truth_tmp[18, :]) / 2  # neck
    ground_truth[2, :] = (ground_truth_tmp[15, :] + ground_truth_tmp[13, :]) / 2 + female_chest_offset  # Chest
    ground_truth[3, :] = (a * ground_truth_tmp[13, :] + (1-a) * ground_truth_tmp[15, :])  # Waist
    ground_truth[4:10, :] = ground_truth_tmp[[30, 21, 31, 22, 32, 23], :]  # Shoulder, Elbow, Wrist
    ground_truth[10, :] = (3 * ground_truth_tmp[32, :] + ground_truth_tmp[35, :]) / 4  # lHand 手の甲
    ground_truth[11, :] = (3 * ground_truth_tmp[23, :] + ground_truth_tmp[26, :]) / 4  # rHand 手の甲
    #ground_truth[10, :] = (ground_truth_tmp[32, :] + ground_truth_tmp[35, :]) / 2  # lHand 指の付け根
    #ground_truth[11, :] = (ground_truth_tmp[23, :] + ground_truth_tmp[26, :]) / 2  # rHand 指の付け根
    ground_truth[12:16, :] = ground_truth_tmp[[9, 3, 10, 4], :]  # Knee, Ankle
    ground_truth[16, :] = (3 * ground_truth_tmp[10, :] + ground_truth_tmp[12, :]) / 4  # lFoot
    ground_truth[17, :] = (3 * ground_truth_tmp[4, :] + ground_truth_tmp[6, :]) / 4  # rFoot

    for i in range(ground_truth.shape[0]):
        ground_truth[i, :] = np.dot(ground_truth[i, :], rotation_matrix(radians(-figure_rotation), "y"))

#    gt_human_bottom = np.array([0, np.min(ground_truth[:, 1]), 0])
#    gt_human_bottom = np.array([0, ground_truth[3, 1], 0])
#    ground_truth -= np.tile(gt_human_bottom, (ground_truth.shape[0], 1))
    gt_human_center = np.mean(ground_truth[visible_joints], axis=0)
#    gt_human_center = np.array([ground_truth[3, 0], 0, ground_truth[3, 2]])
    ground_truth -= np.tile(gt_human_center, (ground_truth.shape[0], 1))

    return ground_truth


def JPP_precision():

    args = get_args()

    discr_setting_type = args.discr_setting_type
    n_train_images = args.n_train_images

    data_path = args.data_path
    jpp_path = data_path + "Main/JointPositionPrediction/"
    bpc_path = data_path + "Main/BodyPartClassification/"
    jpp_gt_path = jpp_path + "GroundTruth/"
    jpp_out_path = jpp_path + "Output/"
    eval_path = jpp_path + "Evaluation/"
    wc_path = data_path + "Preprocessing/MotionBVH/WC/"

    target_joint_names = ["Head", "neck", "Chest", "Waist",
                          "rShoulder", "lShoulder", "rElbow", "lElbow", "rWrist", "lWrist", "rHand", "lHand",
                          "rKnee", "lKnee", "rAnkle", "lAnkle", "rFoot", "lFoot"]
    gt_joint_names = ["Hip",
                      "lButtock", "Left Thigh", "Left Shin", "Left Foot", "lToe", "Site",
                      "rButtock", "Right Thigh", "Right Shin", "Right Foot", "rToe", "Site",
                      "Waist", "Abdomen", "Chest", "Neck", "Neck1", "Head", "Site",
                      "Left Collar", "Left Shoulder", "Left Forearm", "Left Hand" "LeftFingerBase", "LFingers", "Site", "lThumb1", "Site",
                      "Right Collar", "Right Shoulder", "Right Forearm", "Right Hand", "RightFingerBase", "RFingers", "Site", "rThumb1", "Site"]
    joint_connection = [[1, 0], [2, 1], [3, 2],
                        [2, 4], [2, 5], [4, 6], [5, 7], [6, 8], [7, 9], [8, 10], [9, 11],
                        [3, 12], [3, 13], [12, 14], [13, 15], [14, 16], [15, 17]]
    n_joints = len(target_joint_names)

    n_test_images = args.n_test_images
    test_filenames = enum_test_files(args.data_path, args.test_path, n_test_images)
    test_filename_ids = ["/".join(f.split("/")[-2:]) for f in test_filenames]

    setting_str = "_" + str(n_train_images) + ("_%s" % discr_setting_type if discr_setting_type else "")

    if discr_setting_type:
        discr_setting_path = bpc_path + "Intermediate/discretization_setting/"
        discr_setting_filename = "%s%s.csv" % (discr_setting_path, discr_setting_type)

    average_precision_path = eval_path + "JPP_average_precision" + setting_str + "_nb.csv"
    average_error_path = eval_path + "JPP_average_error" + setting_str + "_nb.csv"
    all_errors_path = eval_path + "JPP_all_errors" + setting_str + "_nb.csv"
    n_per_joint_correct = np.zeros((n_joints, 1))
    prediction_errors = np.zeros((n_joints+1, n_test_images))
    n_pass_files = 0

    for i, test_filename in enumerate(test_filenames):

        test_filename_id = test_filename_ids[i]
        print("%d: %s" % (i, test_filename_id))
        if discr_setting_type:
            discr_idx = int(test_filename_id.split("_")[-1])
            discr_regions = discretization_setting(discr_setting_filename)
            for d in discr_regions:
                if discr_idx in d:
                    discr_name = str(d)
        test_param_path = test_filename + "_param"
        test_gt_path = jpp_gt_path + test_filename_id + "_gt.ply"
        test_JPP_path = jpp_out_path + test_filename_id + "_" + str(n_train_images) + ("_%s" % discr_name if discr_setting_type else "") + "_nb_JPP.ply"
#        test_JPP_hcc_path = JPP_OUT_PATH + test_filename_id + setting_str + "_JPP_hcc.ply"
        error_path = eval_path + test_filename_id + setting_str + "_nb_JPP_error.csv"

        if not os.path.exists(test_JPP_path):
            print("Pass %s" % (test_JPP_path))
            prediction_errors[:, i] = np.nan
            n_pass_files += 1
            continue

        predicted_joint_positions = load_JPP_ply(test_JPP_path)

        visible_joints = []
        invisible_joints = []
        for j in range(n_joints):
            if np.sum(np.abs(predicted_joint_positions[j, :])) == 0:
                invisible_joints.append(j)
                print("%s is invisible." % target_joint_names[j])
            else:
                visible_joints.append(j)
        visible_joints = np.array(visible_joints)
        invisible_joints = np.array(invisible_joints)

        param_dict = get_parameter(test_param_path)
        if path.exists(test_gt_path) and False:
            ground_truth = load_JPP_ply(test_gt_path)
        else:
            ground_truth = make_ground_truth(param_dict, wc_path, visible_joints)
            make_JPP_ply(test_gt_path, ground_truth)

        per_joint_error = np.sqrt(np.sum((ground_truth - predicted_joint_positions) ** 2, axis=1))
        if invisible_joints.shape[0] != 0:
            per_joint_error[invisible_joints] = np.nan
        mean_error = np.nanmean(per_joint_error, axis=0)
        prediction_error = np.r_[per_joint_error, mean_error]
        prediction_errors[:, i] = prediction_error

        pd.DataFrame(prediction_error, index=target_joint_names+["Mean"]).to_csv(error_path, header=False)
        print("\tMean Error is %fcm" % mean_error)

        for j in range(n_joints):
            if per_joint_error[j] <= 10:
                n_per_joint_correct[j] += 1

    print(n_pass_files)

    average_precision = n_per_joint_correct / n_test_images * 100
    mean_average_precision = np.nanmean(average_precision)
    pd.DataFrame(np.r_[average_precision.flatten(), mean_average_precision], index=target_joint_names+["Mean"]).to_csv(average_precision_path, header=False)
    print("mAP is %f%%" % np.mean(average_precision))

    mean_errors = np.nanmean(prediction_errors, axis=1)
    pd.DataFrame(mean_errors, index=target_joint_names+["Mean"]).to_csv(average_error_path, header=False)
    print("Mean error is %fcm" % mean_errors[-1])

    pd.DataFrame(prediction_errors, columns=test_filename_ids, index=target_joint_names+["Mean"]).to_csv(all_errors_path, header=True)


if __name__ == "__main__":
    JPP_precision()
