
import glob, os
import numpy as np
import pandas as pd
from math import sqrt
from Modules.utils import get_args
from JPP_precision import load_JPP_ply
from divide_and_conquer_BPC import discretization_setting


def mode(arr):
    r = [0] * (max(arr) + 1)
    for a in arr:
        r[a] += 1

    return r.index(max(r))


def euclid_dists(drawer_locations, jpp_result):
    return np.array([sqrt(np.sum((x - jpp_result) ** 2)) for x in drawer_locations])


def predict_drawer():

    args = get_args()
    n_train_images = args.n_train_images
    data_path = args.data_path
    discr_setting_type = args.discr_setting_type
    jpp_path = data_path + "Main/JointPositionPrediction/"
    output_path = jpp_path + "Output/"
    drawers_location_fname = jpp_path + "GroundTruth/drawers/ground_truth.csv"
    discr_setting_fname = "%sMain/BodyPartClassification/Intermediate/discretization_setting/%s.csv" % (data_path, discr_setting_type)

    drawers_locations = np.array(pd.read_csv(drawers_location_fname, header=None))

    test_path = data_path + "Main/BodyPartClassification/" + args.test_path
    video_names = ["/".join(f.split("/")[-2:]).replace("_raw.avi", "") for f in glob.glob(test_path + "*_raw.avi")]
    cap_img_path = data_path + "Main/BodyPartClassification/CapturedImages/"

    for video_name in video_names:
        print(video_name)
        jpp_results_fnames = []
        frame = 0
        if discr_setting_type:
            discr_regions = discretization_setting(discr_setting_fname)
            while True:
                for discr_region in discr_regions:
                    discr_name = str(discr_region)
                    jpp_results_fname = "%s%s_%d_%d%s_nb_JPP.ply" % (output_path, video_name, frame, n_train_images,
                                                                     "_" + discr_name if discr_setting_type is not None else "")
                    if os.path.exists(jpp_results_fname):
                        jpp_results_fnames.append(jpp_results_fname)
                        break

                frame += 1
                test_fname = "%s%s_%d.png" % (cap_img_path, video_name, frame)
                if len(jpp_results_fnames) != frame or not os.path.exists(test_fname):
                    break
        else:
            while True:
                jpp_results_fname = "%s%s_%d_%d_nb_JPP.ply" % (output_path, video_name, frame, n_train_images)
                if os.path.exists(jpp_results_fname):
                    jpp_results_fnames.append(jpp_results_fname)
                else:
                    break
                frame += 1

        predicted_drawers = []
        drawers_min_dists = {}
        drawers_min_frame = {}

        for i in range(33):
            drawers_min_dists[i] = 100
            drawers_min_frame[i] = 0

        for frame, jpp_results_fname in enumerate(jpp_results_fnames):
            jpp_results = load_JPP_ply(jpp_results_fname)
            rhand_jpp_result = jpp_results[10]
            lhand_jpp_result = jpp_results[11]
            distr = euclid_dists(drawers_locations, rhand_jpp_result)
            distl = euclid_dists(drawers_locations, lhand_jpp_result)
            dist_array = np.array([distr, distl])
            min_dists = np.min(dist_array, axis=0)
            tmp_predicted = np.argmin(min_dists)
            if drawers_min_dists[tmp_predicted] > np.min(min_dists):
                drawers_min_dists[tmp_predicted] = np.min(min_dists)
                drawers_min_frame[tmp_predicted] = frame
            predicted_drawers.append(np.argmin(min_dists))

        predicted_drawers_fname = output_path + video_name + "_nb_drawers.csv"
        pd.DataFrame(predicted_drawers).to_csv(predicted_drawers_fname, header=False, index=False)

        predicted_drawer_fname = output_path + video_name + "_nb_drawer.csv"
        if len(predicted_drawers) != 0:
            predicted_drawer = min(drawers_min_dists, key=drawers_min_dists.get)
            print("drawer: %d (frame: %d, dist: %f)" % (predicted_drawer,
                                                        drawers_min_frame[predicted_drawer],
                                                        drawers_min_dists[predicted_drawer]))
            pd.DataFrame([predicted_drawer]).to_csv(predicted_drawer_fname, header=False, index=False)
        else:
            print("Can't predict.")
            pd.DataFrame([-1]).to_csv(predicted_drawer_fname, header=False, index=False)


if __name__ == "__main__":
    predict_drawer()
