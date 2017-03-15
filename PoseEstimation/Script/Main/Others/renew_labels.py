
import os, random, cv2
import pandas as pd
import numpy as np
import multiprocessing as mp
from Modules.features_labels import make_labels
from Modules.utils import get_args, figure_disappears
from Modules.image_processing import human_rectangle


def _discretization_setting(discr_setting_filename=None):

    discr_regions = []
    with open(discr_setting_filename, 'r') as fin:
        for line in fin:
            discr_regions.append([int(float(item)) for item in line.split(",") if (item != "" and item != "\n")])

    return discr_regions


def _save_data(data, path, compression_type="gzip"):
    pd.DataFrame(data).to_csv(path, compression=compression_type, header=False, index=False)


def main():

    for_cl = True

    args = get_args()
    data_path = args.data_path
    bpc_path = data_path + "Main/BodyPartClassification/"
    img_path = bpc_path + "SyntheticImages/"
    intermediate_path = bpc_path + "Intermediate/"
    train_fnames_order_path = intermediate_path + "input_order.csv"

    target_files = []
    train_fnames = np.array(pd.read_csv(train_fnames_order_path, header=None)).flatten()[:args.n_train_images]

    if args.discr_setting_type is None:

        if for_cl:
            for f in train_fnames:
                for d in range(64):
                    target_files.append("%s_%d" % (f, d))
        else:
            np.random.seed(1)
            discr_idxs = np.random.randint(0, 63, train_fnames.shape[0])
            for f, i in zip(train_fnames, discr_idxs):
                if i % 8 != 7:
                    target_files.append("%s_%d" % (f, i))
                else:
                    target_files.append("%s_%d" % (f, np.random.choice([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14,
                                                                        16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30,
                                                                        32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46,
                                                                        48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 59, 60, 61, 62])))
    else:
        discr_type_fname = intermediate_path + "discretization_setting/" + args.discr_setting_type + ".csv"

        discr_regions = _discretization_setting(discr_type_fname)

        min_discr_regions = list(range(64))
        for discr_region in discr_regions:
            np.random.seed(1)
            for train_fname in train_fnames:
                np.random.shuffle(min_discr_regions)
                appended = False
                for min_discr_region in min_discr_regions:
                    tmp_train_fname = "%s_%d" % (train_fname, min_discr_region)
                    if min_discr_region in discr_region:
                        if not appended:
                            target_files.append(tmp_train_fname)
                            appended = True
                        if os.path.exists("%s%s_features.gz" % (intermediate_path, tmp_train_fname)):
                            target_files[-1] = tmp_train_fname
                            break

    s_processes = []
    for i, target in enumerate(target_files):
        print("%5d: %s" % (i, target))
        random.seed(1)
        label_img_name = img_path + target + ".png"
        labels_fname = intermediate_path + target + "_labels.gz"
        features_fname = intermediate_path + target + "_features.gz"
        if figure_disappears(label_img_name) or os.path.exists(labels_fname):
            continue
        if os.path.exists(labels_fname) and not os.path.exists(features_fname):
            print("Remove %s." % target)
            os.remove(labels_fname)
        if os.path.exists(features_fname) and not os.path.exists(labels_fname):
            label_px = np.asarray(cv2.imread(label_img_name), dtype=np.uint8)[:, :, :3][:, :, ::-1]
            min_v, max_v, min_h, max_h = human_rectangle(label_px)
            target_pixels = np.array([[random.randint(min_v, max_v), random.randint(min_h, max_h)]
                                      for x in range(min((max_h - min_h + 1) * (max_v - min_v + 1), 2000))],
                                     dtype=np.uint16)

            labels = make_labels(label_px, target_pixels)
            s_process = mp.Process(target=_save_data, args=(labels, labels_fname))
            s_process.start()
            s_processes.append(s_process)

            if len(s_processes) > 4:
                for s_process in s_processes:
                    s_process.join()
                s_processes = []

    for s_process in s_processes:
        s_process.join()


if __name__ == "__main__":
    main()
