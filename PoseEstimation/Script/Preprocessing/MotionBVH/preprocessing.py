# -*- coding: utf-8 -*-

import glob
from .PreproScript import normalize_bvh, reduce_bvh, make_convert


def main():
    in_path = "../Data/Original/"
    out_path = "../Data/Intermediate/"
    origin_app = ".bvh"
    norm_app = "_norm.bvh"
    wc_app = "_WC"
    reduce_app = "_reduce.bvh"
    file_names = tuple(x[len(in_path):-len(origin_app)] for x in glob.glob(in_path+"*/*"+origin_app))

    for file_name in file_names:

        print(file_name)

        origin_file = in_path + file_name + origin_app
        norm_file = out_path + file_name + norm_app
        wc_file = out_path + file_name + wc_app
        reduce_file = out_path + file_name + reduce_app

        # 画面の中心に
        normalize_bvh(origin_file, norm_file)

        # 座標変換
        make_wc(norm_file, wc_file)

        # 5cm動く毎にサンプリング
        reduce_bvh(norm_file, wc_file, reduce_file)


if __name__ == '__main__':
    main()
