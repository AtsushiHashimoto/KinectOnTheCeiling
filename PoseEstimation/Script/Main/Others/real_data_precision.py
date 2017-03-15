
import glob
import numpy as np
import pandas as pd
from Modules.utils import get_args


def real_data_precision():

    args = get_args()
    data_path = args.data_path
    jpp_path = data_path + "Main/JointPositionPrediction/"
    output_path = jpp_path + "Output/"

    test_path = data_path + "Main/BodyPartClassification/" + args.test_path
    video_names = ["/".join(f.split("/")[-2:]).replace("_raw.avi", "") for f in glob.glob(test_path + "*_raw.avi")]

    precision = 0
    for video_name in video_names:
        print(video_name)
        gt = int(video_name.split("_")[-1])

        predicted_drawer_fname = "%s%s_nb_drawer.csv" % (output_path, video_name)
        predicted_drawer = np.array(pd.read_csv(predicted_drawer_fname, header=None), dtype=np.int32)[0]
        precision += 1 if predicted_drawer == gt else 0

    precision /= len(video_names)
    print(precision * 100)

if __name__ == "__main__":
    real_data_precision()
