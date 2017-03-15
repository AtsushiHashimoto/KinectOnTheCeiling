# -*- coding: utf-8 -*-

import sys, os
import numpy as np
import pandas as pd
from Modules.camera_location import cal_camera_location, estimate_camera_location
from body_part_classification import BodyPartClassification, run_bpc

__all__ = ["DivConqBPC", "discretization_setting"]


def discretization_setting(discr_setting_filename=None):

    discr_regions = []
    with open(discr_setting_filename, 'r') as fin:
        for line in fin:
            discr_regions.append([int(float(item)) for item in line.split(",") if (item != "" and item != "\n")])

    return discr_regions


def _is_in_discr_region(discr_region, filename_base=False, camera_location=None, filename=None):
    if filename_base:
        test_minimum_discr_region = int(filename.split("_")[-1])
        if test_minimum_discr_region in discr_region:
            return True
        else:
            return False
    else:
        minimum_discr_regions = [[v, h, v + 10, h + 10] for v in range(10, 90, 10) for h in range(-35, 45, 10)]
        for i in discr_region:
            s = minimum_discr_regions[i]
            if s[0] <= camera_location[0] < s[2] and s[1] <= camera_location[1] < s[3]:
                return True

        return False


class DivConqBPC(BodyPartClassification):

    def __init__(self, n_train_images=2000, n_target_pixels_per_image=2000, n_offsets=500, n_sep=1,
                 discr_setting_type=None, discr_regions=None, intervals=None, for_clustering=False):
        BodyPartClassification.__init__(self, n_train_images, n_target_pixels_per_image, n_offsets, n_sep)
        self.rfs = []
        self.discr_setting_type = discr_setting_type
        self.discr_regions = discr_regions
        self.intervals = intervals
        self.for_clustering = for_clustering

    def train(self, train_filenames):

        # Main
        intermediate_path = "/".join(train_filenames[0].split("/")[:-3]) + "/Intermediate/"
        if self.discr_regions is None:
            discr_setting_path = intermediate_path + "discretization_setting/"
            discr_setting_filename = "%s%s.csv" % (discr_setting_path, self.discr_setting_type)
            self.discr_regions = discretization_setting(discr_setting_filename)

        setting_str = self.train_setting_str
        min_discr_regions = list(range(64))
        for discr_region in self.discr_regions:
            np.random.seed(1)
            self.train_setting_str = setting_str + "_" + str(discr_region)
            target_files = []
            for train_filename in train_filenames:
                train_filename_id = "/".join(train_filename.split("/")[-2:])
                appended = False
                np.random.shuffle(min_discr_regions)
                for min_discr_region in min_discr_regions:
                    tmp_train_filename = "%s_%d" % (train_filename, min_discr_region)
                    if min_discr_region in discr_region:
                        if not appended:
                            target_files.append(tmp_train_filename)
                            appended = True
                        if os.path.exists("%s%s_%d_features.gz" % (intermediate_path, train_filename_id, min_discr_region)) \
                                and os.path.exists("%s%s_%d_labels.gz" % (intermediate_path, train_filename_id, min_discr_region)):
                            target_files[-1] = tmp_train_filename
                            break

            #pd.DataFrame(["/".join(f.split("/")[-2:]) for f in target_files]).to_csv("/Volumes/PoseEstimation/Data/Main/BodyPartClassification/target_files.csv", header=False, index=False, mode='a')

            if discr_region[0] % 8 != 7:
                BodyPartClassification.train(self, np.array(target_files))

            self.rfs.append(self.rf)

        self.train_setting_str = setting_str

    def predict(self, test_filename, save=True):

        img = None
        predict_probability = None
        target_pixels = None
        setting_str = self.test_setting_str
        if "CapturedVideos" in test_filename:
            cl_exst_probas = estimate_camera_location(test_filename+".mov", self.discr_regions)
            for i, discr_region in enumerate(self.discr_regions):

                self.rf = self.rfs[i]
                self.test_setting_str = setting_str + "_" + str(discr_region)
                if self.rf is None:
                    raise RuntimeError("The target Random Forest is untrained.")
                else:
                    if predict_probability is None:
                        predict_probability = cl_exst_probas[i] * BodyPartClassification.predict(self, test_filename, save=save)[1]
                    else:
                        predict_probability += cl_exst_probas[i] * BodyPartClassification.predict(self, test_filename, save=save)[1]

        elif "CapturedImages" in test_filename:

            rad, azim, polar = estimate_camera_location(test_filename + ".png")
            for i, discr_region in enumerate(self.discr_regions):
                self.rf = self.rfs[i]
                self.test_setting_str = setting_str + "_" + str(discr_region)
                if _is_in_discr_region(discr_region, filename_base=False, camera_location=[polar, azim]) or self.for_clustering:
                    if self.rf is None:
                        raise RuntimeError("The target Random Forest is untrained.")
                    else:
                        img, predict_probability, target_pixels = BodyPartClassification.predict(self, test_filename, save=save)
                        break
                else:
                    continue

        elif "SyntheticImages" in test_filename:
            camera_location = cal_camera_location(test_filename + "_param")
            for i, discr_region in enumerate(self.discr_regions):
                self.rf = self.rfs[i]
                self.test_setting_str = setting_str + "_" + str(discr_region)
                if _is_in_discr_region(discr_region, filename_base=True, camera_location=camera_location, filename=test_filename) or self.for_clustering:
                    if self.rf is None:
                        raise RuntimeError("The target Random Forest is untrained.")
                    else:
                        img, predict_probability, target_pixels = BodyPartClassification.predict(self, test_filename, save=save)
                        break
                else:
                    continue

        else:
            raise ValueError("Invalid test file path.")

        self.test_setting_str = setting_str

        return img, predict_probability, target_pixels


if __name__ == "__main__":
    run_bpc(DivConqBPC)
