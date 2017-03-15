# -*- coding: utf-8 -*-

import time, cv2, os
import numpy as np
import multiprocessing as mp
from scipy import stats
import pandas as pd
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from Modules.data_preparation import prepare_train_data, prepare_test_data, prepare_offsets
from Modules.utils import get_parameter, get_args, figure_disappears, bvh_exists, enum_train_files, enum_test_files

__all__ = ["BodyPartClassification"]


class BodyPartClassification:

    def __init__(self, n_train_images=2000, n_target_pixels_per_image=2000, n_offsets=500, n_sep=1):
        self.n_target_pixels_per_image = n_target_pixels_per_image
        self.n_offsets = n_offsets
        self.train_setting_str = "_" + str(n_train_images)
        self.test_setting_str = "_" + str(n_train_images)
        self.n_sep = n_sep
        self.compression_type = "gzip"
        self.offsets = None
        self.rf = []
        self.part_labels = np.array([(63,0,0), (0,63,0), (255,0,0), (127,0,63), (127,255,0), (191,255,191), (255,255,191), (127,255,127), (191,191,191), (63,127,0),
                                     (0,191,63), (255,255,0), (255,191,0), (0,255,255), (0,191,255), (127,63,0), (0,63,127), (255,63,255), (63,255,255), (255,63,0),
                                     (0,63,255), (127,63,255), (127,63,63), (63,127,255), (255,63,63), (63,0,63), (63,0,127), (255,127,127), (63,255,63), (191,127,63),
                                     (63,63,0), (255,255,255), (0,0,0)])

    def train(self, train_filenames):

        n_train_images = train_filenames.shape[0]

        bpc_path = "/".join(train_filenames[0].split("/")[:-3]) + "/"
        intermediate_path = bpc_path + "Intermediate/"
        evaluation_path = bpc_path + "Evaluation/"
        offset_path = intermediate_path + "offsets.csv"
        pkl_path = intermediate_path + "pkl/RF" + self.train_setting_str + "_not_balanced.gz"
        fitting_time_path = "%strain_time_%d" % (evaluation_path, n_train_images)

        self.offsets = prepare_offsets(offset_path, self.n_offsets)

        if os.path.exists(pkl_path):
            print("Loading Random Forest...")
            self.rf = joblib.load(pkl_path)
            #self.rf = None
        else:
            fitting_time = 0
            self.rf = []
            # n_sep > 1の時は学習データ分割によるメモリ消費量削減
            stride = int(n_train_images / self.n_sep)
            n_rem_estimators = 10
            n_rem_sep = self.n_sep
            n_jobs = int(mp.cpu_count() / 2)
            for i in range(0, n_train_images, stride):
                features, labels, sample_weight = \
                    prepare_train_data(train_filenames[i: min(i+stride, n_train_images)],
                                       self.offsets, self.n_target_pixels_per_image, self.compression_type)
                print("Training Random Forest...")
                n_estimators = int(n_rem_estimators / n_rem_sep)
                n_rem_estimators -= n_estimators
                n_rem_sep -= 1
                rf = RandomForestClassifier(n_estimators=n_estimators, random_state=1, max_depth=17,
                                            class_weight=None, criterion="entropy", n_jobs=n_jobs)
                #rf = RandomForestClassifier(n_estimators=n_estimators, random_state=1, max_depth=17,
                #                            class_weight="balanced", criterion="entropy", n_jobs=mp.cpu_count())
                fit_start = time.time()
                rf.fit(features, np.ravel(labels), sample_weight)
                fit_end = time.time()
                fitting_time += fit_end - fit_start
                print("Took %fsec for fitting random forest." % (fit_end - fit_start))

                del features, labels, sample_weight

                self.rf.append(rf)

            print("Saving Random Forest...")
            tmp = time.time()
            joblib.dump(self.rf, pkl_path, compress=3)
            print("Took %fsec for saving random forest." % (time.time() - tmp))

            pd.DataFrame([fitting_time]).to_csv(fitting_time_path, header=False, index=False, mode='a')

    def predict(self, test_filename, save=True):

        bpc_path = "/".join(test_filename.split("/")[:-3]) + "/"
        intermediate_path = bpc_path + "Intermediate/"
        out_path = bpc_path + "Output/"

        n_part_labels = self.part_labels.shape[0] - 1

        test_filename_id = "/".join(test_filename.split("/")[-2:])
        test_feature_path = intermediate_path + test_filename_id + "_features.gz"
        target_pixels_path = intermediate_path + test_filename_id + "_target_pixels.gz"

        test_BPC_image_path = out_path + test_filename_id + self.test_setting_str + "_nb_BPC.png"
        test_BPC_proba_path = out_path + test_filename_id + self.test_setting_str + "_nb_BPC_proba.gz"
        if os.path.exists(test_BPC_proba_path) and os.path.exists(test_BPC_image_path):
            return None, None, None

        features, image_shape, target_pixels = prepare_test_data(test_filename, test_feature_path, target_pixels_path,
                                                                 self.offsets, self.compression_type)

        height, width = image_shape

        test_predict = np.ones((height, width, self.n_sep), dtype=np.uint8) * 31
        test_predict_proba = np.zeros((height, width, n_part_labels))
        test_predict_proba[:, :, 31] = 1
        test_predict_proba[target_pixels[:, 0], target_pixels[:, 1], 31] = 0

        # n_sep > 1の時はメモリ消費量削減のための分割処理
        print("Predicting test data label...")
        tmp = time.time()
        for s, rf in enumerate(self.rf):

            tmp_predicts = rf.predict(features)
            tmp_predict_probas = rf.predict_proba(features)
            for i, target_pixel in enumerate(target_pixels):
                test_predict[target_pixel[0], target_pixel[1], s] = tmp_predicts[i]
                test_predict_proba[target_pixel[0], target_pixel[1], :] += tmp_predict_probas[i, :]

        print("Took %fsec for predict." % (time.time() - tmp))
        test_predict_proba /= self.n_sep

        # 分類結果の描画
        predict_px = np.ones((image_shape[0], image_shape[1], 3), dtype=np.uint8) * 255
        for v, h in target_pixels:
            predict_px[v, h, :] = self.part_labels[int(stats.mode(test_predict[v, h, :])[0])]

        if save:
            cv2.imwrite(test_BPC_image_path, predict_px[:, :, ::-1])

        # 分類結果の確率分布をデータで保存
        test_predict_proba = test_predict_proba.reshape((height * width, n_part_labels))
        if save:
            pd.DataFrame(test_predict_proba).to_csv(test_BPC_proba_path, compression=self.compression_type, header=False, index=False)

        return predict_px, test_predict_proba, target_pixels

    def video_predict(self, test_filename):

        bpc_path = "/".join(test_filename.split("/")[:-3]) + "/"
        intermediate_path = bpc_path + "Intermediate/"
        out_path = bpc_path + "Output/"

        n_part_labels = self.part_labels.shape[0] - 1

        test_filename_id = "/".join(test_filename.split("/")[-2:])
        print(test_filename_id)
        test_feature_path = intermediate_path + test_filename_id + "_features.gz"
        target_pixels_path = intermediate_path + test_filename_id + "_target_pixels.gz"
        test_BPC_video_path = out_path + test_filename_id + self.test_setting_str + "_BPC.mov"
        test_BPC_proba_path = out_path + test_filename_id + self.test_setting_str + "_BPC_proba.gz"

        features, video_shape, target_pixels = prepare_test_data(test_filename, test_feature_path, target_pixels_path,
                                                                 self.offsets, self.compression_type)

        n_frames, height, width = video_shape

        test_predict = np.ones((n_frames, height, width, self.n_sep), dtype=np.uint8) * 31
        test_predict_proba = np.zeros((n_frames, height, width, n_part_labels))
        test_predict_proba[:, :, :, 31] = 1
        for f, v, h in target_pixels:
            test_predict_proba[f, v, h, 31] = 0

        # n_sep > 1の時はメモリ消費量削減のための分割処理
        for s in range(self.n_sep):
            rf = self.rf[s]

            print("Predicting test data label...")
            rf.n_jobs = 1
            tmp_predicts = rf.predict(features)
            tmp_predict_probas = rf.predict_proba(features)
            for i, target_pixel in enumerate(target_pixels):
                f, v, h = target_pixel
                test_predict[f, v, h, s] = tmp_predicts[i]
                test_predict_proba[f, v, h, :] += tmp_predict_probas[i, :]

        test_predict_proba /= self.n_sep

        # 分類結果の描画
        predict_px = np.ones((n_frames, height, width, 3), dtype=np.uint8) * 255
        tmp = -1
        for f, v, h in target_pixels:
            if tmp < f:
                tmp = f
                print("frame%d" % f)
            predict_px[f, v, h, :] = self.part_labels[int(stats.mode(test_predict[f, v, h, :])[0])]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        predict_out = cv2.VideoWriter(test_BPC_video_path, fourcc, 30.0, (width, height))
        for frame_px in predict_px[:, :, :, ::-1]:
            predict_out.write(frame_px)

        # 分類結果の確率分布をデータで保存
        test_predict_proba = test_predict_proba.reshape((n_frames * height * width, n_part_labels))
        pd.DataFrame(test_predict_proba).to_csv(test_BPC_proba_path, compression=self.compression_type, header=False, index=False)

        return predict_px, test_predict_proba, target_pixels


def run_bpc(bpc_model=BodyPartClassification):

    args = get_args()

    bpc_args = {"n_sep": args.n_sep, "n_train_images": args.n_train_images, }

    n_train_images = args.n_train_images
    n_test_images = args.n_test_images
    full_rotation = args.full_rotation
    if bpc_model is not BodyPartClassification:
        bpc_args["discr_setting_type"] = args.discr_setting_type
    data_path = args.data_path

    train_filenames = enum_train_files(data_path, n_train_images, bpc_model, full_rotation)

    if bpc_model is not None:
        print("====%s====" % bpc_model.__name__)
        bpc = bpc_model(**bpc_args)
    else:
        raise ValueError

    bpc.train(train_filenames)

    test_filenames = enum_test_files(data_path, args.test_path, n_test_images)

    if "CapturedVideos" in args.test_path:
        for i, test_filename in enumerate(test_filenames):
            test_filename_id = "/".join(test_filename.split("/")[-2:])
            print("%d: %s" % (i, test_filename_id))
            _, _, _ = bpc.video_predict(test_filename)
    elif "CapturedImages" in args.test_path or "SyntheticImages" in args.test_path:
        for i, test_filename in enumerate(test_filenames):
            test_filename_id = "/".join(test_filename.split("/")[-2:])
            print("%d: %s" % (i, test_filename_id))
            _, _, _ = bpc.predict(test_filename)
    else:
        raise ValueError("Invalid test path.")

if __name__ == "__main__":
    run_bpc(BodyPartClassification)

