# -*- coding=utf8 -*-

import random, time, argparse, glob, os, tarfile, cv2, sys
cimport numpy as np
import numpy as np
from cython.parallel cimport prange
import pandas as pd
from math import tan, radians
from distutils.util import strtobool
from libc.stdlib cimport free
from .coordinate_conversion import rel2abs
from .utils import get_parameter, rgb_dist
from .image_processing import extract_human_region, human_rectangle

import itertools as iters
import multiprocessing as mp
from functools import partial

__all__ = ["make_offsets", "make_features", "make_labels", "make_features_labels"]


def make_offsets(n_offsets=500, max_offset=150*200, random_seed=1):

    random.seed(random_seed)
    offsets = []
    for i in range(n_offsets):
        if random.choice([True, False]): # 1/2の確率で自身の画素をみる
            offsets.append((random.randint(-max_offset / 5, max_offset / 5) * 5, random.randint(-max_offset / 5, max_offset / 5) * 5,
                            random.randint(-max_offset / 5, max_offset / 5) * 5, random.randint(-max_offset / 5, max_offset / 5) * 5))
        else:
            offsets.append((0, 0,
                            random.randint(-max_offset / 5, max_offset / 5) * 5, random.randint(-max_offset / 5, max_offset / 5) * 5))

    return np.array(offsets, np.int32)


cdef void _set_bg_450cm(np.ndarray[DOUBLE_t, ndim=2] abs_px, object label_px):
    # Kinectである程度の精度が保証されるのが，深度450cmまでなので，最深値を450cmとする．

    cdef UINT16_t v, h
    cdef np.ndarray[UINT16_t, ndim=2] target_pixels
    cdef tuple idxs
    cdef np.ndarray[DOUBLE_t, ndim=2] new_abs_px = np.ones((abs_px.shape[0], abs_px.shape[1]), dtype=np.float64) * 450.
    target_pixels = extract_human_region(label_px)

    idxs = tuple(target_pixels.transpose())
    new_abs_px[idxs] = abs_px[idxs]
    abs_px[:] = new_abs_px


def _depth_difference_features(np.ndarray[INT32_t, ndim=1] target_pixel, np.ndarray[DOUBLE_t, ndim=2] abs_px,
                               np.ndarray[INT32_t, ndim=2] offsets):
    cdef np.ndarray[INT32_t, ndim=1] offset
    return np.array([_depth_difference_feature(abs_px, target_pixel, offset)
                     for offset in (offsets / abs_px[target_pixel[0], target_pixel[1]]).astype(np.int32)])


cdef inline DOUBLE_t _depth_difference_feature(np.ndarray[DOUBLE_t, ndim=2] abs_px, np.ndarray[INT32_t, ndim=1] target_pixel,
                                               np.ndarray[INT32_t, ndim=1] offset):

    cdef DOUBLE_t val1, val2
    cdef np.ndarray[INT32_t, ndim=1] tmp_target_pixel = target_pixel + offset

    try:
        val1 = abs_px[tmp_target_pixel[0], tmp_target_pixel[1]]
    except IndexError:
        val1 = 450.

    try:
        val2 = abs_px[tmp_target_pixel[2], tmp_target_pixel[3]]
    except IndexError:
        val2 = 450.

    return val1 - val2


def make_features(np.ndarray[UINT8_t, ndim=2] px, np.ndarray[INT32_t, ndim=2] offsets, np.ndarray[UINT8_t, ndim=3] label_px,
                  dict params=None, np.ndarray[UINT16_t, ndim=2] target_pixels=None,
                  np.ndarray[UINT16_t, ndim=2] target_rect=None):

    cdef DOUBLE_t start, end, old, tmp
    cdef np.ndarray[DOUBLE_t, ndim=2] abs_px
    cdef np.ndarray[DOUBLE_t, ndim=2] features

    start = time.time()

    if target_pixels is None:
        target_pixels = extract_human_region(label_px)

    if params is None:
        offsets = (offsets * (256 * tan(radians(29)) / (160 * tan(radians(35))))).astype(np.int32)

    abs_px = rel2abs(px, params, target_rect)

    _set_bg_450cm(abs_px, label_px)

    partial_ddf = partial(_depth_difference_features, abs_px=abs_px, offsets=offsets)
    with mp.Pool(int(mp.cpu_count() / 4)) as pool:
        results = pool.map(partial_ddf, np.c_[target_pixels, target_pixels].astype(np.int32))
        features = np.array([feature for feature in results])

    end = time.time()
    print("Took %.7f seconds for feature extraction." % (end - start))

    return features.astype(np.float16), target_pixels


def _labeling(np.ndarray[UINT16_t, ndim=1] target_pixel, np.ndarray[INT32_t, ndim=3] px, np.ndarray[INT32_t, ndim=2] part_labels):

    cdef UINT16_t v, h
    cdef INT32_t label, idx, tmp_dist
    cdef np.ndarray[INT32_t, ndim=1] part_label

    v, h = target_pixel
    label = part_labels.shape[0] - 2
    for idx, part_label in enumerate(part_labels):
        tmp_dist = np.sum(np.abs(px[v, h, :] - part_label))
        if 32 > tmp_dist:
            label = min(idx, 31)
            break

    return np.uint8(label)


def make_labels(np.ndarray[UINT8_t, ndim=3] px, np.ndarray[UINT16_t, ndim=2] target_pixels=None):

    cdef DOUBLE_t start, end
    cdef np.ndarray[UINT8_t, ndim=1] labels
    cdef np.ndarray[INT32_t, ndim=2] part_labels

    start = time.time()

    part_labels = \
            np.array([(63,0,0), (0,63,0), (255,0,0), (127,0,63), (127,255,0), (191,255,191), (255,255,191), (127,255,127), (191,191,191), (63,127,0),
                      (0,191,63), (255,255,0), (255,191,0), (0,255,255), (0,191,255), (127,63,0), (0,63,127), (255,63,255), (63,255,255), (255,63,0),
                      (0,63,255), (127,63,255), (127,63,63), (63,127,255), (255,63,63), (63,0,63), (63,0,127), (255,127,127), (63,255,63), (191,127,63),
                      (63,63,0), (0,0,0), (255,255,255)], dtype=np.int32)

    cdef UINT16_t v, h
    if target_pixels is None:
        target_pixels = np.array([[v, h] for v in range(px.shape[0]) for h in range(px.shape[1])], dtype=np.uint16)

    partial_labeling = partial(_labeling, px=px.astype(np.int32), part_labels=part_labels)
    with mp.Pool(int(mp.cpu_count() / 4)) as pool:
        results = pool.map(partial_labeling, target_pixels)
        labels = np.array([label for label in results])

    end = time.time()
    print("Took %.7f seconds for label extraction." % (end - start))

    return np.array(labels, dtype=np.uint8)


def make_features_labels(filename, offsets, n_target_pixels_per_image=2000, random_seed=1, tf=None):

    random.seed(random_seed)

    filename_id = "/".join(filename.split("/")[-2:])

    if tf is None:
        depth_filename = filename + " Z.png"
        label_filename = filename + ".png"
        param_filename = filename + "_param"
        depth_img = cv2.imread(depth_filename) # (424, 512)型
        label_img = cv2.imread(label_filename)
        params = get_parameter(param_filename)

        depth_px = np.asarray(depth_img, dtype=np.uint8)[:, :, 0]
        label_px = np.asarray(label_img, dtype=np.uint8)[:,:,:3][:,:,::-1]  # (424, 512)型
    else:
        depth_filename = filename_id + " Z.png"
        label_filename = filename_id + ".png"
        param_filename = filename_id + "_param"
        depth_img = tf.extractfile(depth_filename)
        label_img = tf.extractfile(label_filename)
        params = get_parameter(param_filename, tf)

        depth_px = cv2.imdecode(np.asarray(bytearray(depth_img.read()), dtype=np.uint8), 0)
        label_px = cv2.imdecode(np.asarray(bytearray(label_img.read()), dtype=np.uint8), 1)

    try:
        min_v, max_v, min_h, max_h = human_rectangle(label_px)
    except (IndexError, ValueError):
        print("figure disappears!: %s" % filename_id)
        return np.empty((0, offsets.shape[0]), dtype=np.float16), np.empty((0, 1), dtype=np.uint8)

    target_pixels = np.array([[random.randint(min_v, max_v), random.randint(min_h, max_h)]
                              for x in range(min((max_h - min_h + 1) * (max_v - min_v + 1), n_target_pixels_per_image))],
                              dtype=np.uint16)

    features, _ = make_features(depth_px, offsets, label_px, params=params, target_pixels=target_pixels,
                                target_rect=np.array([[min_v, min_h], [max_v, max_h]], dtype=np.uint16))
    labels = make_labels(label_px, target_pixels)

    return features, labels

