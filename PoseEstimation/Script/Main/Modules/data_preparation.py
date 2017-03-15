# -*- coding: utf-8 -*-

import os, cv2, time, gc, pyximport, tarfile
import numpy as np
import pandas as pd
import multiprocessing as mp
pyximport.install(setup_args={'include_dirs': np.get_include()})
from functools import partial
from .features_labels import make_features_labels, make_features, make_offsets
from .utils import get_args, get_parameter
from .image_processing import segment_human_region

__all__ = ["prepare_train_data", "prepare_test_data", "prepare_offsets", "extract_and_save_features_labels"]


def prepare_offsets(offset_path, n_offsets=500):

    if os.path.exists(offset_path):
        print("Loading offsets...")
        offsets = np.array(pd.read_csv(offset_path, header=None, dtype=np.int32))
    else:
        print("Making offsets...")
        offsets = make_offsets(n_offsets=n_offsets, random_seed=1)
        pd.DataFrame(offsets).to_csv(offset_path, header=False, index=False)

    return offsets


def _load_data_append(path_tup, n_samples, lock, i, data_dict_tup):

    max_float = np.finfo(np.float16).max
    filename_id = "/".join(path_tup[0].split("/")[-2:]).replace("_features.gz", "")
    try:
        data1 = np.array(pd.read_csv(path_tup[0], header=None, dtype=np.float16))
        data2 = np.array(pd.read_csv(path_tup[1], header=None, dtype=np.uint8)).reshape((-1, 1))
    except pd.io.common.EmptyDataError:
        pd.DataFrame([filename_id]).to_csv("empty_fnames.csv", header=False, index=False, mode='a')
        data1 = np.empty((0, 500))
        data2 = np.empty((0, 1))

    data1[data1 > max_float] = max_float
    data1[data1 < -max_float] = -max_float

    lock.acquire()
    n_samples.value += data1.shape[0]
    data_dict_tup[0][i] = data1
    data_dict_tup[1][i] = data2.reshape((-1, 1))
    lock.release()


def _save_data(data_tup, path_tup, compression_type="gzip"):
    try:
        pd.DataFrame(data_tup[0]).to_csv(path_tup[0], compression=compression_type, header=False, index=False)
        pd.DataFrame(data_tup[1]).to_csv(path_tup[1], compression=compression_type, header=False, index=False)
    except KeyboardInterrupt:
        os.remove(path_tup[0])
        os.remove(path_tup[1])


def prepare_train_data(train_filenames, offsets=None, n_target_pixels_per_image=2000, compression_type="gzip", load=True):

    # Prepare train data
    max_float = np.finfo(np.float16).max
    n_offsets = offsets.shape[0]

    bpc_path = "/".join(train_filenames[0].split("/")[:-3]) + "/"
    intermediate_path = bpc_path + "Intermediate/"
    images_path = bpc_path + "SyntheticImages/"
    arc_filename = images_path + "data_arc.tar"
    n_train_images = train_filenames.shape[0]

    if os.path.exists(arc_filename):
        tf = tarfile.open(arc_filename)
    else:
        tf = None

    lock = mp.Lock()
    manager = mp.Manager()
    features_dict = manager.dict()
    labels_dict = manager.dict()
    n_samples = mp.Value('i', 0)
    l_processes = []
    s_processes = []
    partial_lda = partial(_load_data_append, data_dict_tup=tuple([features_dict, labels_dict]))
    for i, train_filename in enumerate(train_filenames):

        filename_id = "/".join(train_filename.split("/")[-2:])
        print("%5d: %s" % (i, filename_id))

        features_path = intermediate_path + filename_id + "_features.gz"
        labels_path = intermediate_path + filename_id + "_labels.gz"

        if os.path.exists(features_path) and os.path.exists(labels_path) and os.stat(features_path).st_size > 100 and load:
            print("Loading...")
            tmp = time.time()
            #features_dict[i] = np.array(pd.read_csv(features_path, header=None, dtype=np.float16))
            #labels_dict[i] = np.array(pd.read_csv(labels_path, header=None, dtype=np.uint8)).reshape((-1, 1))
            #n_samples.value += features_dict[i].shape[0]
            l_process = mp.Process(target=partial_lda,
                                   args=((features_path, labels_path), n_samples, lock, i))
            l_process.start()
            l_processes.append(l_process)

            print("Took %fsec for loading data." % (time.time() - tmp))
        else:
            print("Making...")
            tmp_features, tmp_labels \
                = make_features_labels(train_filename, offsets, n_target_pixels_per_image, tf=tf)
            tmp = time.time()
            #_save_data((tmp_features, tmp_labels), (features_path, labels_path), compression_type)
            s_process = mp.Process(target=_save_data,
                                   args=((tmp_features, tmp_labels), (features_path, labels_path), compression_type))
            s_process.start()
            s_processes.append(s_process)

            #print("Took %fsec for saving data." % (time.time() - tmp))

            tmp = time.time()
            tmp_features[tmp_features > max_float] = max_float
            tmp_features[tmp_features < -max_float] = -max_float
            features_dict[i] = tmp_features
            labels_dict[i] = tmp_labels.reshape((-1, 1))
            n_samples.value += tmp_features.shape[0]
            print("Took %fsec for append data." % (time.time() - tmp))

        if len(s_processes) + len(l_processes) > mp.cpu_count() / 24:
            for l_process in l_processes:
                l_process.join()
            for s_process in s_processes:
                s_process.join()
            l_processes = []
            s_processes = []

    for l_process in l_processes:
        l_process.join()
    for s_process in s_processes:
        s_process.join()

    tmp = time.time()
    features = np.empty((n_samples.value + 32, n_offsets), dtype=np.float16)
    labels = np.empty((n_samples.value + 32, 1), dtype=np.uint8)

    sample_weight = np.r_[np.ones((n_samples.value,), dtype=np.int32), np.zeros((32,), dtype=np.int32)]
    features_dict[n_train_images] = np.zeros((32, n_offsets), dtype=np.float16)
    labels_dict[n_train_images] = np.arange(32, dtype=np.uint8).reshape((-1, 1))

    start = 0
    error_subs = []
    features_list = [features_dict[i] for i in range(n_train_images+1)]
    labels_list = [labels_dict[i] for i in range(n_train_images+1)]
    for i, (sub_features, sub_labels) in enumerate(zip(features_list, labels_list)):
        print(i)
        n_sub_samples = sub_features.shape[0]
        if sub_features.shape[0] != sub_labels.shape[0]:
            print(train_filenames[i])
            error_subs.append(i)
            features[start:start+n_sub_samples, :] = 0
            labels[start:start+n_sub_samples] = 0
            sample_weight[start:start+n_sub_samples] = 0
        else:
            features[start:start+n_sub_samples, :] = sub_features
            labels[start:start+n_sub_samples] = sub_labels
        start += n_sub_samples

    del features_list, labels_list
    gc.collect()
    print("Took %fsec for concatenating." % (time.time() - tmp))

    return features, labels, sample_weight


def prepare_test_data(test_filename, test_features_path, target_pixels_path,
                      offsets=None, compression_type="gzip"):

    max_float = np.finfo(np.float16).max

    if "CapturedVideos" in test_filename:

        cap = cv2.VideoCapture(test_filename + ".mov")
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = frame.shape
        media_shape = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), height, width)
        if os.path.exists(test_features_path):
            print("Loading test features...")
            features = np.array(pd.read_csv(test_features_path, compression=compression_type, header=None, dtype=np.float16))
            target_pixels = np.array(pd.read_csv(target_pixels_path, compression=compression_type, header=None, dtype=np.uint32))
        else:
            print("Making test features...")
            features = np.empty((0, offsets.shape[0]))
            target_pixels = np.empty((0, 3))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            processed_out = cv2.VideoWriter(test_filename + "_processed.mov", fourcc, 30.0, (width, height), isColor=False)
            bg_frame = np.asarray(cv2.imread(test_filename + ".png", cv2.IMREAD_GRAYSCALE))
            f = 0

            while cap.isOpened():

                print("%d frame:" % f)
                ret, raw_frame = cap.read()

                if ret:
                    segmented_frame, mask = segment_human_region(raw_frame, bg_frame)  #raw_frameはcm，bg_frameは256階調
                    processed_out.write(segmented_frame)
                    if np.sum(mask) != 0:
                        tmp_features, tmp_target_pixels = make_features(segmented_frame, offsets, mask)
                        features = np.append(features, tmp_features, axis=0)
                        tmp_target_pixels = np.c_[f * np.ones((tmp_target_pixels.shape[0], 1)), tmp_target_pixels]
                        target_pixels = np.append(target_pixels, tmp_target_pixels, axis=0)
                else:
                    break

                f += 1

            processed_out.release()

            try:
                pd.DataFrame(features).to_csv(test_features_path, compression=compression_type, header=False, index=False)
                pd.DataFrame(target_pixels).to_csv(target_pixels_path, compression=compression_type, header=False, index=False)
            except KeyboardInterrupt:
                os.remove(test_features_path)
                os.remove(target_pixels_path)

        cap.release()

    elif "CapturedImages" in test_filename:

        cap_images_path = "/".join(test_filename.split("/")[:-2]) + "/"
        bg_filename = cap_images_path + "background.png"
        depth_img = cv2.imread(test_filename + ".png")
        media_shape = depth_img.shape[:2]
        if os.path.exists(test_features_path) and os.path.exists(target_pixels_path):
            print("Loading test features...")
            features = np.array(pd.read_csv(test_features_path, compression=compression_type, header=None, dtype=np.float16))
            target_pixels = np.array(pd.read_csv(target_pixels_path, compression=compression_type, header=None, dtype=np.uint16))
        else:
            print("Making test features...")

            bg_frame = np.asarray(cv2.imread(bg_filename, cv2.IMREAD_GRAYSCALE))
            raw_frame = np.asarray(depth_img)[:, :, 0]
            segmented_frame, mask = segment_human_region(raw_frame, bg_frame)  #raw_frame,bg_frameは256階調
            features, target_pixels = make_features(segmented_frame, offsets, mask)

            try:
                pd.DataFrame(features).to_csv(test_features_path, compression=compression_type, header=False, index=False)
                pd.DataFrame(target_pixels).to_csv(target_pixels_path, compression=compression_type, header=False, index=False)
            except KeyboardInterrupt:
                os.remove(test_features_path)
                os.remove(target_pixels_path)

    elif "SyntheticImages" in test_filename:

        depth_img = cv2.imread(test_filename + " Z.png")
        media_shape = depth_img.shape[:2]
        if os.path.exists(test_features_path):
            print("Loading test features...")
            features = np.array(pd.read_csv(test_features_path, compression=compression_type, header=None, dtype=np.float16))
            target_pixels = np.array(pd.read_csv(target_pixels_path, compression=compression_type, header=None, dtype=np.uint16))
        else:
            print("Making test features...")
            test_depth_px = np.asarray(depth_img)[:, :, 0]
            label_img = cv2.imread(test_filename + ".png")
            test_label_px = np.asarray(label_img)[:, :, :3][:,:,::-1]
            params = get_parameter(test_filename + "_param")
            features, target_pixels = make_features(test_depth_px, offsets, test_label_px, params=params)

            try:
                pd.DataFrame(features).to_csv(test_features_path, compression=compression_type, header=False, index=False)
                pd.DataFrame(target_pixels).to_csv(target_pixels_path, compression=compression_type, header=False, index=False)
            except KeyboardInterrupt:
                os.remove(test_features_path)
                os.remove(target_pixels_path)

    else:
        raise ValueError("Invalid test file path.")

    features[features > max_float] = max_float
    features[features < -max_float] = -max_float

    return features, media_shape, target_pixels


def extract_and_save_features_labels():

    args = get_args()

    n_target_pixels_per_image=2000
    compression_type="gzip"

    data_path = args.data_path
    bpc_path = data_path + "Main/BodyPartClassification/"
    intermediate_path = bpc_path + "Intermediate/"
    images_path = bpc_path + "SyntheticImages/"
    arc_filename = images_path + "data_arc.tar"

    if os.path.exists(arc_filename):
        tf = tarfile.open(arc_filename)
    else:
        tf = None

    train_images_order_path = intermediate_path + "input_order.csv"
    train_filenames = np.array(["%s%s_%d" % (images_path, f, d)
                                for f in np.array(pd.read_csv(train_images_order_path, dtype=str, header=None)).flatten()
                                for d in range(64)])

    offset_path = intermediate_path + "offsets.csv"
    offsets = prepare_offsets(offset_path)

    # Prepare train data
    bpc_path = "/".join(train_filenames[0].split("/")[:-3]) + "/"
    intermediate_path = bpc_path + "Intermediate/"

    for i, train_filename in enumerate(train_filenames):

        filename_id = "/".join(train_filename.split("/")[-2:])
        print("%3d: %s" % (i, filename_id))

        features_path = intermediate_path + filename_id + "_features.gz"
        labels_path = intermediate_path + filename_id + "_labels.gz"
        if os.path.exists(features_path) and os.path.exists(labels_path):
            continue

        print("Making...")
        tmp_features, tmp_labels \
            = make_features_labels(train_filename, offsets, n_target_pixels_per_image, tf=tf)
        try:
            pd.DataFrame(tmp_features).to_csv(features_path, compression=compression_type, header=False, index=False)
            pd.DataFrame(tmp_labels).to_csv(labels_path, compression=compression_type, header=False, index=False)
        except KeyboardInterrupt:
            os.remove(features_path)
            os.remove(labels_path)

