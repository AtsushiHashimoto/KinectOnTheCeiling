# -*- coding=utf8 -*-

import os, time
import scipy.stats
import numpy as np
import pandas as pd
from Modules.utils import get_args, proba_parts2joints, figure_disappears
from divide_and_conquer_BPC import DivConqBPC


def is_contig(d1, d2):

    for min_discr_region1 in d1:
        if (min_discr_region1 % 8 < 7 and min_discr_region1 + 1 in d2) or \
                (min_discr_region1 < 57 and min_discr_region1 + 8 in d2):
            return True

    return False


def ret_contig_regions(discr_regions, discr_region1):
    contig_regions = set()
    for discr_region2 in discr_regions:
        if is_contig(discr_region1, discr_region2):
            contig_regions.add(tuple(discr_region2))

    contig_regions = [list(contig_region) for contig_region in contig_regions]

    return contig_regions


def list2index_name(l):
    return " ".join(str(l).split(","))


def index_name2list(s):
    return [int(d) for d in s[1:-1].split()]


def opposite(x):
    centers = np.arange(3, 60, 8)
    for c in centers:
        if abs(x - c) <= 3:
            return 2 * c - x

    return x


def merge_regions(discr_regions, merge_tup):

    discr_regions[merge_tup[0]].extend(discr_regions[merge_tup[1]])
    discr_regions[merge_tup[0]].sort()
    discr_regions.pop(merge_tup[1])

    return discr_regions


def update_sim_mat(sim_mat, merge_tups):

    sim_mat = np.delete(sim_mat, merge_tups[1], axis=0)
    sim_mat = np.delete(sim_mat, merge_tups[1], axis=1)
    sim_mat[merge_tups[0], :] = 1
    sim_mat[:, merge_tups[0]] = 1

    return sim_mat


def ret_sym_mat_idx(mat_idx, region_name_list):
    sym_index_list = [opposite(idx) for idx in index_name2list(region_name_list[mat_idx])]
    sym_index_list.sort()
    sym_mat_idx = region_name_list.index(list2index_name(sym_index_list))
    return sym_mat_idx


def symmetrize_sim_mat(sim_mat, region_name_list):

    for mat_idx1, region_name1 in enumerate(region_name_list[:-1]):
        index_list1 = index_name2list(region_name1)
        if np.any(np.mod(index_list1, 8) > 3):
            continue
        else:
            sym_mat_idx1 = ret_sym_mat_idx(mat_idx1, region_name_list)
            for region_name2 in region_name_list[mat_idx1:]:
                mat_idx2 = region_name_list.index(region_name2)
                index_list2 = index_name2list(region_name2)
                if np.any(np.mod(index_list1, 8) == 7) or np.any(np.mod(index_list2, 8) == 7) or not is_contig(index_list1, index_list2):
                    continue
                sym_mat_idx2 = ret_sym_mat_idx(mat_idx2, region_name_list)
                if sim_mat[mat_idx1, mat_idx2] > 0:
                    print(region_name_list[mat_idx1])
                    print(region_name_list[mat_idx2])
                    print(sim_mat[mat_idx1, mat_idx2])
                if sim_mat[sym_mat_idx1, sym_mat_idx2] > 0:
                    print(region_name_list[sym_mat_idx1])
                    print(region_name_list[sym_mat_idx2])
                    print(sim_mat[sym_mat_idx1, sym_mat_idx2])
                sim_mat[mat_idx1, mat_idx2] = (sim_mat[mat_idx1, mat_idx2] + sim_mat[sym_mat_idx1, sym_mat_idx2]) / 2.
                sim_mat[mat_idx2, mat_idx1] = (sim_mat[mat_idx2, mat_idx1] + sim_mat[sym_mat_idx1, sym_mat_idx2]) / 2.
                sim_mat[sym_mat_idx1, sym_mat_idx2] = sim_mat[mat_idx1, mat_idx2]
                sim_mat[sym_mat_idx2, sym_mat_idx1] = sim_mat[mat_idx2, mat_idx1]

    return sim_mat


def symmetry_merge(sim_mat, discr_regions):

    tmp_sim_mat = symmetrize_sim_mat(sim_mat, region_index_list)
    sim_max = np.max(tmp_sim_mat[np.where(tmp_sim_mat != 1)])
    print("Max similarity: %f" % sim_max)

    merge_tups = np.where(tmp_sim_mat == sim_max)
    merge_tup1 = [merge_tups[0][0], merge_tups[1][0]]
    print(merge_tups)
    if (not is_center(discr_regions[merge_tup1[0]])) and (not is_center(discr_regions[merge_tup1[1]])):
        merge_tup2 = [merge_tups[0][2], merge_tups[1][2]]
    else:
        merge_tup2 = None

    print("Merge: %s %s" % (discr_regions[merge_tup1[0]], discr_regions[merge_tup1[1]]))
    discr_regions = merge_regions(discr_regions, merge_tup1)
    sim_mat = update_sim_mat(sim_mat, merge_tup1)

    if merge_tup2 is not None:
        if merge_tup1[1] < merge_tup2[0]:
            merge_tup2[0] -= 1
        elif merge_tup1[1] == merge_tup2[0]:
            merge_tup2[0] = merge_tup1[0]
        if merge_tup1[1] < merge_tup2[1]:
            merge_tup2[1] -= 1
        elif merge_tup1[1] == merge_tup2[1]:
            merge_tup2[1] = merge_tup1[0]
        print("Merge: %s %s" % (discr_regions[merge_tup2[0]], discr_regions[merge_tup2[1]]))
        discr_regions = merge_regions(discr_regions, merge_tup2)
        sim_mat = update_sim_mat(sim_mat, merge_tup2)

    return discr_regions, sim_mat


def is_active(d):

    for non_active_d in [7, 15, 23, 31, 39, 47, 55, 63]:
        if non_active_d in d:
            return False

    return True


def is_center(d):

    for c in [3, 11, 19, 27, 35, 43, 51, 59]:
        if c in d:
            return True

    return False


def bpc_similarity(bpc1, bpc2, test_filenames):

    epsilon = 1e-10
    sum_info_loss = 0
    n_px = 0
    start_each_pair = time.time()
    d1, d2 = bpc1.discr_regions[0], bpc2.discr_regions[0]
    for test_filename in test_filenames:

        test_filename1 = "%s_%d" % (test_filename, np.random.choice(d1))
        test_filename2 = "%s_%d" % (test_filename, np.random.choice(d2))

        if not os.path.exists(test_filename1 + " Z.png") or not os.path.exists(test_filename2 + " Z.png") \
                or figure_disappears(test_filename1 + ".png") or figure_disappears(test_filename2 + ".png"):
            continue

        img, img11_proba, target_pixels1 = bpc1.predict(test_filename1, save=False)
        _, img22_proba, target_pixels2 = bpc2.predict(test_filename2, save=False)
        _, img12_proba, _ = bpc1.predict(test_filename2, save=False)
        _, img21_proba, _ = bpc2.predict(test_filename1, save=False)

        img_width = img.shape[1]
        n_target_pixels1 = target_pixels1.shape[0]
        n_target_pixels2 = target_pixels2.shape[0]

        new_img11_proba = np.zeros((n_target_pixels1, 32))
        new_img22_proba = np.zeros((n_target_pixels2, 32))
        new_img12_proba = np.zeros((n_target_pixels2, 32))
        new_img21_proba = np.zeros((n_target_pixels1, 32))

        for i, (v, h) in enumerate(target_pixels1):
            new_img11_proba[i, :] = img11_proba[img_width * v + h, :]
            new_img21_proba[i, :] = img21_proba[img_width * v + h, :]
        for i, (v, h) in enumerate(target_pixels2):
            new_img22_proba[i, :] = img22_proba[img_width * v + h, :]
            new_img12_proba[i, :] = img12_proba[img_width * v + h, :]

        for px_proba_11, px_proba_21 in zip(proba_parts2joints(new_img11_proba), proba_parts2joints(new_img21_proba)):
            non_zero_idx = np.where(px_proba_21 > epsilon)
            if np.sum(px_proba_11[non_zero_idx]) > 0:
                sum_info_loss += scipy.stats.entropy(px_proba_11[non_zero_idx], px_proba_21[non_zero_idx])
                n_px += 1
        for px_proba_22, px_proba_12 in zip(proba_parts2joints(new_img22_proba), proba_parts2joints(new_img12_proba)):
            non_zero_idx = np.where(px_proba_12 > epsilon)
            if np.sum(px_proba_22[non_zero_idx]) > 0:
                sum_info_loss += scipy.stats.entropy(px_proba_22[non_zero_idx], px_proba_12[non_zero_idx])
                n_px += 1

    if n_px != 0:
        sim = - sum_info_loss / n_px
    else:
        sim = - np.inf

    end_each_pair = time.time()
    print("Took %fsec for each pair." % (end_each_pair - start_each_pair))

    return sim


def camera_location_clustering():

    args = get_args()

    M = 1  #目標分割数
    n_test_images = args.n_test_images

    n_train_images = args.n_train_images

    bpc_args = {"n_sep": 1, "n_train_images": n_train_images, "for_clustering": True}

    data_path = args.data_path
    bpc_path = data_path + "Main/BodyPartClassification/"
    intermediate_path = bpc_path + "Intermediate/"
    images_path = bpc_path + "SyntheticImages/"
    train_images_order_path = intermediate_path + "input_order.csv"
    test_images_order_path = intermediate_path + "test_input_order.csv"

    discr_setting_path = bpc_path + "Intermediate/discretization_setting/"

    train_filenames = \
        np.array([images_path + f for f in
                  np.array(pd.read_csv(train_images_order_path, dtype=str, header=None))]).flatten()[:n_train_images]

    test_filenames = \
        np.array([images_path + f for f in
                  np.array(pd.read_csv(test_images_order_path, dtype=str, header=None))]).flatten()[:n_test_images]

    discr_regions = [[i] for i in range(64) if i % 8 != 7]

    bpc_args1 = bpc_args.copy()
    bpc_args2 = bpc_args.copy()

    np.random.seed(1)
    start = time.time()

    sim_mat = np.ones((len(discr_regions), len(discr_regions)))

    iteration = 0

    while len(discr_regions) > M:

        print("The number of discr locations: %d" % len(discr_regions))
        start_each_iteration = time.time()
        sim_mat_path = "%s/similarity_matrix/sim_mat_%d_%d_%d.csv" % (intermediate_path, n_train_images,
                                                                            n_test_images, iteration)

        if os.path.exists(sim_mat_path):
            sim_mat = np.array(pd.read_csv(sim_mat_path, index_col=0, header="infer"))

        for a, d1 in enumerate(discr_regions[:-1]):

            bpc_args1["discr_regions"] = [d1]
            contig_regions = ret_contig_regions(discr_regions[a+1:], d1)

            for d2 in contig_regions:

                print("Comparing " + str(d1) + ", " + str(d2))
                b = discr_regions.index(d2)
                bpc_args2["discr_regions"] = [d2]

                if sim_mat[a, b] < 0:
                    print("Similarity: %f" % sim_mat[a, b])
                    continue
                else:
                    try:
                        bpc1 = DivConqBPC(**bpc_args1)
                        bpc2 = DivConqBPC(**bpc_args2)

                        bpc1.train(train_filenames)
                        bpc2.train(train_filenames)

                        if is_active(d1) and is_active(d2):
                            sim = bpc_similarity(bpc1, bpc2, test_filenames)
                        else:
                            sim = - np.inf

                        sim_mat[a, b] = sim
                        sim_mat[b, a] = sim
                        print("Similarity: %f" % sim)
                    except:
                        region_index_list = [list2index_name(d) for d in discr_regions]
                        pd.DataFrame(sim_mat, index=region_index_list, columns=region_index_list).to_csv(sim_mat_path, index=True, header=True)
                        raise KeyboardInterrupt

        region_index_list = [list2index_name(d) for d in discr_regions]
        pd.DataFrame(sim_mat, index=region_index_list, columns=region_index_list).to_csv(sim_mat_path, index=True, header=True)

        discr_regions = symmetry_merge(sim_mat, discr_regions)

        save_path = "%stype_%d_%d_%d.csv" % (discr_setting_path, n_train_images, n_test_images, iteration)
        pd.DataFrame(discr_regions).to_csv(save_path, header=False, index=False)

        end_each_iteration = time.time()
        print("Took %fsec for each iteration." % (end_each_iteration - start_each_iteration))

        iteration += 1

    end = time.time()
    print("Took %fsec for whole the clustering." % (end - start))


if __name__ == "__main__":
    camera_location_clustering()
