# -*- coding: utf-8 -*-

import glob, os, cv2
import numpy as np
import pandas as pd
from Modules.utils import get_args, figure_disappears

args = get_args()

NUM_TRAIN_IMAGES = args.n_train_images
DISCR_SETTING_TYPE = args.discr_setting_type
NUM_TARGET_PIXELS_PER_IMAGE = 2000
DATA_PATH = args.data_path
BPC_PATH = DATA_PATH + "Main/BodyPartClassification/"
OUT_PATH = BPC_PATH + "Output/"
EVAL_PATH = BPC_PATH + "Evaluation/"
IMAGES_PATH = BPC_PATH + "SyntheticImages/"
INTERMEDIATE_PATH = BPC_PATH + "Intermediate/"


def BPC_precision():

    test_images_order_path = INTERMEDIATE_PATH + "test_input_order.csv"
    n_test_images = args.n_test_images

    part_labels = tuple([(63,0,0), (0,63,0), (255,0,0), (127,0,63), (127,255,0), (191,255,191), (255,255,191), (127,255,127), (191,191,191), (63,127,0),
                         (0,191,63), (255,255,0), (255,191,0), (0,255,255), (0,191,255), (127,63,0), (0,63,127), (255,63,255), (63,255,255), (255,63,0),
                         (0,63,255), (127,63,255), (127,63,63), (63,127,255), (255,63,63), (63,0,63), (63,0,127), (255,127,127), (63,255,63), (191,127,63),
                         (63,63,0), (255,255,255), (0,0,0)])

    part_names = ["ruHead", "rwHead", "luHead", "lwHead", "neck", "rChest", "lChest", "rWaist", "lWaist", "rShoulder", "lShoulder",
                  "ruArm", "rwArm", "luArm", "lwArm", "rElbow", "lElbow", "rWrist", "lWrist", "rHand", "lHand", "ruLeg", "rwLeg",
                  "luLeg", "lwLeg", "rKnee", "lKnee", "rAnkle", "lAnkle", "rFoot", "lFoot", "Background"]

    n_parts = len(part_names)

    if "*male" in args.test_path:
        test_filenames = \
            np.array([IMAGES_PATH + f for f in
                      np.array(pd.read_csv(test_images_order_path, dtype=str, header=None))]).flatten()
        np.random.seed(1)
        test_filenames = np.array(["%s_%d" % (f, d)
                                   for f, d in zip(test_filenames, np.random.randint(0, 63, test_filenames.shape[0]))
                                   if not figure_disappears("%s_%d.png" % (f, d))])[:n_test_images]
    else:
        test_filenames = np.array([f.replace(" Z.png", "") for f in glob.glob(IMAGES_PATH + "テスト/* Z.png")])

    n_test_files = test_filenames.shape[0]
    setting_str = "_" + str(NUM_TRAIN_IMAGES)

    average_precision_path = EVAL_PATH + "BPC_average_precision" + setting_str + ("%s" % DISCR_SETTING_TYPE if DISCR_SETTING_TYPE else "") + ".csv"
    precision = np.zeros((n_test_files, n_parts))

    for i, test_filename in enumerate(test_filenames):
        test_filename_id = "/".join(test_filename.split("/")[-2:])
        print("%d: %s" % (i, test_filename_id))
        ground_truth_path = IMAGES_PATH + test_filename_id + ".png"
        classification_path = glob.glob(OUT_PATH + test_filename_id + setting_str + ("_[*" if DISCR_SETTING_TYPE else "") + "_BPC.png")[0]
        precision_path = EVAL_PATH + test_filename_id + setting_str + ("_%s" % DISCR_SETTING_TYPE if DISCR_SETTING_TYPE else "") + "_BPC_precision.csv"
        if os.path.exists(precision_path):
            per_part_precision = np.array(pd.read_csv(precision_path, index_col=0)).flatten()
        else:

            gt_img = cv2.imread(ground_truth_path)
            bpc_img = cv2.imread(classification_path)

            gt_px = np.asarray(gt_img)[:, :, :3][:, :, ::-1]
            bpc_px = np.asarray(bpc_img)[:, :, :3][:, :, ::-1]

            height = gt_px.shape[0]
            width = gt_px.shape[1]

            n_px_per_part_correct = np.zeros(n_parts)
            n_px_per_part = np.zeros(n_parts)
            for v in range(height):
                for h in range(width):
                    try:
                        idx_part = part_labels.index(tuple(gt_px[v, h]))
                    except:
                        continue # labelingの失敗画素はスキップ
                    if idx_part >= 31:
                        n_px_per_part[31] += 1
                        if tuple((255, 255, 255)) == tuple(bpc_px[v, h]) or tuple((0, 0, 0)) == tuple(bpc_px[v, h]):
                            n_px_per_part_correct[31] += 1
                    else:
                        n_px_per_part[idx_part] += 1
                        if tuple(gt_px[v, h]) == tuple(bpc_px[v, h]):
                            n_px_per_part_correct[idx_part] += 1

            per_part_precision = n_px_per_part_correct / n_px_per_part * 100

        mean_precision = np.nanmean(per_part_precision[:-1])

        precision[i, :] = per_part_precision

        pd.DataFrame(np.r_[per_part_precision, mean_precision], index=part_names+["Mean"]).to_csv(precision_path, header=False)
        print("\tMean precision is %f%%" % mean_precision)

    average_per_joint_precision = np.nanmean(precision, axis=0)
    average_mean_precision = np.nanmean(average_per_joint_precision[:-1])
    pd.DataFrame(np.r_[average_per_joint_precision, average_mean_precision], index=part_names+["Mean"]).to_csv(average_precision_path, header=False)
    print("Average mean precision is %f%%" % average_mean_precision)

if __name__ == "__main__":
    BPC_precision()
