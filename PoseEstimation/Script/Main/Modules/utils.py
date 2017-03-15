
import argparse, cv2, os, glob
import numpy as np
import pandas as pd
from distutils.util import strtobool

__all__ = ["get_parameter", "get_args", "rgb_dist", "proba_parts2joints", "figure_disappears", "ensure_dir",
           "bvh_exists", "enum_train_files", "enum_test_files"]


def get_parameter(param_filename, tf=None):

    param_dict = {}
    if tf is None:
        with open(param_filename, 'r') as fin:
            for line in fin.readlines():
                items = line.split(":")
                try:
                    if items[0] == "Camera DollyZ":
                        param_dict[items[0]] = float(items[1]) + 287
                    else:
                        param_dict[items[0]] = float(items[1])
                except ValueError:
                    param_dict[items[0]] = items[1].replace("\n", "")
    else:
        p_file = tf.extractfile(param_filename)
        for line in p_file.readlines():
            items = line.decode("utf-8").split(":")
            try:
                if items[0] == "Camera DollyZ":
                    param_dict[items[0]] = float(items[1]) + 287
                else:
                    param_dict[items[0]] = float(items[1])
            except ValueError:
                param_dict[items[0]] = items[1].replace("\n", "")

    return param_dict


def get_args():

    p = argparse.ArgumentParser()
    p.add_argument("-d", "--data_path", type=str, default="../../Data/")
    p.add_argument("-t", "--test_path", type=str, default="SyntheticImages/*male/")
    p.add_argument("-n", "--n_train_images", type=int, default=2000)
    p.add_argument("-N", "--n_test_images", type=int, default=100)
    p.add_argument("-f", "--full_rotation", type=str, default="False")
    p.add_argument("-D", "--discr_setting_type", type=str, default=None)
    p.add_argument("-s", "--n_sep", type=int, default=1)
    args = p.parse_args()
    args.full_rotation = bool(strtobool(args.full_rotation))

    return args


def rgb_dist(px_val, part_label):
    return np.sum((px_val.astype(np.float32) - part_label.astype(np.float32))**2)**(1./2.)


def proba_parts2joints(part_proba, also_bg=False):

    if also_bg:
        joint_proba = np.zeros((part_proba.shape[0], 19))
        joint_proba[:, 18] = part_proba[:, 31]  # background
    else:
        joint_proba = np.zeros((part_proba.shape[0], 18))
    joint_proba[:, 0] = np.sum(part_proba[:, :4], axis=1)  # Head
    joint_proba[:, 1] = part_proba[:, 4]  # neck
    joint_proba[:, 2] = np.sum(part_proba[:, 5:7], axis=1)  # Chest
    joint_proba[:, 3] = np.sum(part_proba[:, 7:9], axis=1)  # Waist
    joint_proba[:, 4:6] = part_proba[:, 9:11]  # Shoulder
    joint_proba[:, 6:12] = part_proba[:, 15:21]  # Elbow, Wrist, Hand
    joint_proba[:, 12:18] = part_proba[:, 25:31]  # Knee, Ankle, Foot

    return joint_proba


def figure_disappears(label_filename):
    label_px = cv2.imread(label_filename)[:, :, :3][:, :, ::-1]
    tmp = np.sum(label_px, axis=2)
    if np.where((255 * 3 - 63 >= tmp) & (tmp >= 63))[0].shape[0] > 10:
        return False
    else:
        return True


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def bvh_exists(data_path, fname):

    bvh_path = data_path + "Preprocessing/MotionBVH/Regularized/"
    param_fname = fname + "_0_param"
    params = get_parameter(param_fname)

    return os.path.exists(bvh_path + params["BVH File Name"])


def enum_train_files(data_path, n_train_images, bpc_model, full_rotation):

    bpc_path = data_path + "Main/BodyPartClassification/"
    intermediate_path = bpc_path + "Intermediate/"
    images_path = bpc_path + "SyntheticImages/"

    active_idxs = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14,
                   16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30,
                   32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46,
                   48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 59, 60, 61, 62]

    train_images_order_path = intermediate_path + "input_order.csv"
    if os.path.exists(train_images_order_path):
        train_filenames = \
            np.array([images_path + f for f in
                      np.array(pd.read_csv(train_images_order_path, dtype=str, header=None))]).flatten()
    else:
        train_filenames = np.array([images_path+"male/%05d" % i for i in range(7500)])
        train_filenames = np.append(train_filenames,
                                    np.array([images_path+"female/%05d" % i for i in range(7500)]))
        np.random.seed(1)
        np.random.shuffle(train_filenames)
        train_filename_ids = ["/".join(f.split("/")[-2:]) for f in train_filenames]
        pd.DataFrame(train_filename_ids).to_csv(train_images_order_path, header=False, index=False)

    #if not full_rotation:

    #    faced_idx = []

    #    for i, train_filename in enumerate(train_filenames):
    #        fr = get_parameter(train_filename+"_0_param")["Figure Rotation"]
    #        if 0 <= fr % 360 <= 120 or 240 <= fr % 360 <= 360:
    #            faced_idx.append(i)
    #            if len(faced_idx) >= n_train_images:
    #                break

    #    train_filenames = train_filenames[faced_idx]

    if bpc_model.__name__ is "BodyPartClassification":
        np.random.seed(1)
        discr_idxs = np.random.randint(0, 63, train_filenames.shape[0])
        train_fnames_list = []
        for f, i in zip(train_filenames, discr_idxs):
            if i % 8 != 7:
                train_fnames_list.append("%s_%d" % (f, i))
            else:
                train_fnames_list.append("%s_%d" % (f, np.random.choice(active_idxs)))

        train_filenames = np.array(train_fnames_list)

    return train_filenames[:n_train_images]


def enum_test_files(data_path, test_path, n_test_images):

    bpc_path = data_path + "Main/BodyPartClassification/"
    intermediate_path = bpc_path + "Intermediate/"
    images_path = bpc_path + "SyntheticImages/"

    active_idxs = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14,
                   16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30,
                   32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46,
                   48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 59, 60, 61, 62]

    if "CapturedVideos" in test_path:
        print(bpc_path + test_path + "*.mov")
        test_filenames = np.array([f.replace(".mov", "") for f in glob.glob(bpc_path + test_path + "*.mov")])
    elif "CapturedImages" in test_path:
        test_filenames = np.array([f.replace(".png", "") for f in glob.glob(bpc_path + test_path + "*.png")])
    elif "SyntheticImages" in test_path:
        if "*male" in test_path:
            test_images_order_path = intermediate_path + "test_input_order.csv"
            if os.path.exists(test_images_order_path):
                test_filenames = \
                    np.array([images_path + f for f in
                              np.array(pd.read_csv(test_images_order_path, dtype=str, header=None))]).flatten()
            else:
                test_filenames = np.array([images_path+"male/%05d" % i for i in range(7500, 8000)])
                test_filenames = np.append(test_filenames,
                                           np.array([images_path+"female/%05d" % i for i in range(7500, 8000)]))
                np.random.seed(1)
                np.random.shuffle(test_filenames)
                test_filename_ids = ["/".join(f.split("/")[-2:]) for f in test_filenames]
                pd.DataFrame(test_filename_ids).to_csv(test_images_order_path, header=False, index=False)
            np.random.seed(1)
            discr_idxs = np.random.randint(0, 63, test_filenames.shape[0])
            target_list = []
            for f, i in zip(test_filenames, discr_idxs):
                if i % 8 != 7:
                    target_list.append("%s_%d" % (f, i))
                else:
                    target_list.append("%s_%d" % (f, np.random.choice(active_idxs)))

            test_filenames = np.array([f for f in target_list
                                       if not figure_disappears(f+".png") and bvh_exists(data_path, "_".join(f.split("_")[:-1]))])
        else:
            test_filenames = np.array([f.replace(" Z.png", "") for f in glob.glob(bpc_path + test_path + "* Z.png")])
    else:
        raise ValueError("Invalid test file path.")

    return test_filenames[:n_test_images]

