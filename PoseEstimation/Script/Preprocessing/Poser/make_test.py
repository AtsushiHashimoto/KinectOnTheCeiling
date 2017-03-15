
# -*- coding: utf-8 -*-

import random, os, poser, datetime
import numpy as np
from Modules import enumerate_poses, make_discrete_regions, cal_fig_params, cal_camera_params, \
                    set_fig_params, set_camera_params, save_param


def main():

    n_images = 100  # 生成画像数

    newest_edit_date = datetime.datetime(2016, 10, 14, 21, 30, 0)

    root_path = "/Volumes/PoseEstimation/Data/"
    bvh_path = root_path + "Preprocessing/MotionBVH/Regularized/"
    image_path = root_path + "Main/BodyPartClassification/SyntheticImages/"
    train_path = image_path + "train/"
    document_path = root_path + "Preprocessing/Poser/"

    fig_names = ["female", "male"]

    # 身長分布は文科省のデータを参考に作成
    fig_ave_height_dict = {"female": 157.9, "male": 170.7}
    fig_height_var_dict = {"female": 5.33, "male": 5.8}

    # 画像名idの桁数を設定
    n_digit = 1
    q = n_images
    while q >= 10:
        q /= 10
        n_digit += 1

    # 全てのbvhファイルを走査して取りうる姿勢を列挙．
    motion_files, frame_idx_in_file = enumerate_poses(bvh_path)

    # カメラ位置の離散化方法を生成
    discrete_regions = make_discrete_regions([10, 10])

    file_ids = [(("0" * n_digit) + str(i))[-n_digit:] for i in range(n_images)]
    np.random.seed(1)
    np.random.shuffle(file_ids)

    for k, fig_name in enumerate(fig_names):

        param_dict = {}

        for i in range(n_images / len(fig_names)):
            print(k*7500+i)

            file_id = file_ids[n_images / len(fig_names) * k + i] + "_" + str(len(discrete_regions) - 1)
            param_filename = train_path + file_id + "_param"

            if os.path.exists(param_filename):
                param_file_stat = os.stat(param_filename)
                if newest_edit_date < datetime.datetime.fromtimestamp(param_file_stat.st_mtime):
                    continue

            poser.CloseDocument(1)
            poser.OpenDocument(document_path + fig_name + ".pz3")
            poser.SetRenderInSeparateProcess(0)
            poser.SetNumRenderThreads(8)

            random.seed(i + fig_names.index(fig_name) * (n_images / len(fig_names)) + 15000)  # seedの設定

            scene = poser.Scene()
            scene.SetOutputRes(512, 424)  # 出力画像のサイズを設定

            scene.SetRenderAntiAliased(0)

            # レンダリング設定
            firefly_options = scene.CurrentFireFlyOptions()
            firefly_options.SetExtraOutput(2,1)
            firefly_options.SetShadows(0)
            firefly_options.SetUseSSS(0)
            firefly_options.SetRayTracing(0)
            firefly_options.SetPixelSamples(1)

            param_dict["Figure Name"] = fig_name

            # フィギュアのパラメタ設定
            param_dict.update(cal_fig_params(fig_ave_height_dict[fig_name], fig_height_var_dict[fig_name],
                                             motion_files, frame_idx_in_file, fig_name))

            set_fig_params(scene, param_dict, bvh_path, fig_name)

            for j, discrete_region in enumerate(discrete_regions):

                file_id = file_ids[n_images / len(fig_names) * k + i] + "_" + str(j)
                param_filename = train_path + file_id + "_param"
                image_filename = train_path + file_id + ".png"

                # 各カメラパラメタを生成・計算
                fig_height = float(param_dict["Figure Height"])
                param_dict.update(cal_camera_params(fig_height, discrete_region))

                # カメラパラメタを設定
                set_camera_params(scene, param_dict)

                # レンダリング
                scene.Render()

                # パラメタを保存
                save_param(param_filename, param_dict)

                # 画像を保存．SaveImageによりRGBと深度両方を保存してくれる．
                scene.SaveImage("PNG", image_filename)


if __name__ == "__main__":
    main()

