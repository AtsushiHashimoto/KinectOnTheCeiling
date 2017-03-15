# -*- coding: utf-8 -*-

import glob


def analyze_dir():
    in_path = "../Data/Intermediate/"
    in_appendix = "_reduce.bvh"

    file_names = (x[len(in_path):-len(in_appendix)] for x in glob.glob(in_path+"*/*"+in_appendix))

    with open("./file_names.dat", 'w') as f:
        for filename in file_names:
            f.write(filename+"\n")


if __name__ == "__main__":
    analyze_dir()
