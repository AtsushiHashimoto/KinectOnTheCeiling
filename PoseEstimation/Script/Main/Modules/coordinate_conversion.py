# -*- coding: utf-8 -*-

import numpy as np
import multiprocessing as mp
from functools import partial
from statistics import mode, StatisticsError
from math import radians, sin, cos, sqrt, tan, atan2, floor, pi, degrees


__all__ = ["make_point_cloud", "rel2abs", "project_point_cloud", "rotation_matrix", "point_cloud2hcc"
           "human_loc2camera_loc"]


def _get_standard(px):

    back = 0
    center = floor(px.shape[0] / 2)
    front = px.shape[0] - 1

    vb = back
    db_ = float(min(px[vb, :]))
    for v in range(px.shape[0]):
        try:
            tmp = float(mode(px[v, :]))
        except StatisticsError:
            print("StatisticsError: no unique mode; found 2 equally common values")
            break
        if db_ < tmp:
            break
        db_ = tmp
        vb = v

    vf = front
    df_ = float(min(px[vf, :]))
    for v in reversed(range(px.shape[0])):
        try:
            tmp = float(mode(px[v, :]))
        except StatisticsError:
            print("StatisticsError: no unique mode; found 2 equally common values")
            break
        if df_ > tmp:
            break
        df_ = tmp
        vf = v

    d0_ = float(min(px[center, :]))
    d1_5_ = float(px[back, -1])
    d450_ = float(px[back, 0])

    return db_, df_, d0_, vb, vf, d1_5_, d450_


def make_point(target_pixel, px, v_center, h_center, v_fov, h_fov, a=None, b=None):

    v, h = target_pixel
    v_angle = atan2(v_center - v, v_center / tan(v_fov))
    h_angle = atan2(h - h_center, h_center / tan(h_fov))
    if a is None or b is None:
        z = px[v, h] * 1000. / 255.
    else:
        z = a * px[v, h] + b

    return z * tan(h_angle), z * tan(v_angle), z


def make_point_cloud(px, params=None, target_rect=None):

    height = px.shape[0]
    width = px.shape[1]
    point_cloud = np.ones((height, width, 3)) * 450.

    if target_rect is None:
        target_rect = np.array([[0, 0], [height-1, width-1]])
    v_min = target_rect[0, 0]
    h_min = target_rect[0, 1]
    v_max = target_rect[1, 0]
    h_max = target_rect[1, 1]

    if params is None or "Camera DollyZ" not in params.keys():
        v_center = (height - 1) / 2
        h_center = (width - 1) / 2
        v_fov = radians(22.5)
        h_fov = radians(29)

        partial_make_point = partial(make_point, px=px, v_center=v_center, h_center=h_center, v_fov=v_fov, h_fov=h_fov)

    else:
        v_center = (height - 1) / 2
        h_center = (width - 1) / 2
        v_fov = radians(30.1077034261)
        h_fov = radians(35)

        db_, df_, d0_, vb, vf, d1_5_, d450_ = _get_standard(px)

        pitch = params['Camera Pitch']
        dollyZ = params['Camera DollyZ']
        dollyY = params['Camera DollyY']

        pitch = radians(pitch)
        d0 = dollyZ + dollyY / tan(pitch)
        vb_angle = atan2(v_center - vb, v_center / tan(v_fov))
        vf_angle = atan2(v_center - vf, v_center / tan(v_fov))
        db = d0 * tan(pitch) / (tan(pitch) - tan(vb_angle))
        df = d0 * tan(pitch) / (tan(pitch) - tan(vf_angle))

        try:
            a = (df - db) / (df_ - db_)
            b = (db * df_ - df * db_) / (df_ - db_)
        except ZeroDivisionError:
            a = (d0 - 1.5) / (d0_ - d1_5_)
            b = (1.5 * d0_ - d0 * d1_5_) / (d0_ - d1_5_)

        partial_make_point = partial(make_point, px=px, a=a, b=b, v_center=v_center, h_center=h_center, v_fov=v_fov, h_fov=h_fov)

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(partial_make_point, ([v, h] for v in range(v_min, v_max+1)
                                                       for h in range(h_min, h_max+1)))
        x = np.array([point for point in results])
        point_cloud[v_min:v_max+1, h_min:h_max+1, :] = x.reshape((v_max-v_min+1, h_max-h_min+1, 3))

    return point_cloud


def rel2abs(px, params=None, target_rect=None):

    return make_point_cloud(px, params, target_rect)[:, :, 2]


def project_point(point, px, joint_label, v_center, h_center, v_fov, h_fov):

    v_angle = -atan2(point[1], point[2])
    h_angle = atan2(point[0], point[2])
    #val = int((point[2] / (cos(v_angle) * cos(h_angle)) - b) / a)
    v = int(v_center + v_center * tan(v_angle) / tan(v_fov))
    h = int(h_center + h_center * tan(h_angle) / tan(h_fov))
    px[v, h] = joint_label
    px[v-1, h] = joint_label
    px[v+1, h] = joint_label
    px[v, h-1] = joint_label
    px[v, h+1] = joint_label

    return v, h


def project_point_cloud(point_cloud, px, visible_joints, device="Kinect"):#, params):

    joint_labels = np.array([(63,0,0), (127,255,0), (191,255,191), (127,255,127), (63,127,0), (0,191,63),
                             (127,63,0), (0,63,127), (255,63,255), (63,255,255), (255,63,0), (0,63,255),
                             (63,0,63), (63,0,127), (255,127,127), (63,255,63), (191,127,63), (63,63,0)])

#    db_, df_, d0_, vb, vf, d1_5_, d450_ = _get_standard(original_px)

    n_joints = point_cloud.shape[0]

#    pitch = params['Camera Pitch']
#    dollyZ = params['Camera DollyZ']
#    dollyY = params['Camera DollyY']

    height = px.shape[0]
    width = px.shape[1]
    v_center = (height - 1) / 2
    h_center = (width - 1) / 2
    if device == "Kinect":
        v_fov = radians(30.1077034261)
        h_fov = radians(35)
    elif device == "Xtion":
        v_fov = radians(22.5)
        h_fov = radians(29)
    else:
        raise ValueError("Invalide device name.")

#    pitch = radians(pitch)
#    d0 = dollyZ + dollyY / tan(pitch)
#    vb_angle = atan2(v_center - vb, v_center / tan(v_fov))
#    vf_angle = atan2(v_center - vf, v_center / tan(v_fov))
#    db = d0 * tan(pitch) / (tan(pitch) - tan(vb_angle))
#    df = d0 * tan(pitch) / (tan(pitch) - tan(vf_angle))

    #px = np.zeros((height, width, 3), dtype=np.uint8)
#    try:
#        a = (df - db) / (df_ - db_)
#        b = (db * df_ - df * db_) / (df_ - db_)
#    except ZeroDivisionError:
#        a = (d0 - 1.5) / (d0_ - d1_5_)
#        b = (1.5 * d0_ - d0 * d1_5_) / (d0_ - d1_5_)

    joint_pos_px_v = np.zeros(n_joints)
    joint_pos_px_h = np.zeros(n_joints)
    for j, point in enumerate(point_cloud):
        if j in visible_joints:
            v, h = project_point(point, px, joint_labels[j], v_center, h_center, v_fov, h_fov)
            joint_pos_px_v[j] = v
            joint_pos_px_h[j] = h

    return px, [joint_pos_px_v, joint_pos_px_h]


def rotation_matrix(theta, axis):

    if type(axis) == str:
        R = np.eye(3)
        c, s = cos(theta), sin(theta)
        if axis == "x":
            R[1,1], R[1,2], R[2,1], R[2,2] = c, -s, s, c
        elif axis == "y":
            R[2,2], R[2,0], R[0,2], R[0,0] = c, -s, s, c
        elif axis == "z":
            R[0,0], R[0,1], R[1,0], R[1,1] = c, -s, s, c
    elif type(axis) == np.ndarray:
        axis /= sqrt(np.dot(axis, axis))
        a = cos(theta / 2.0)
        b, c, d = -axis * sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        R = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                      [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                      [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    else:
        raise ValueError("axis is invalid.")

    return R


# Convert Point Cloud to Human Center Coordinate(HCC)
def point_cloud2hcc(X, params):

    #figure_name = params["Figure Name"]
    pitch = params['Camera Pitch']
    #camera_dollyX = params["Camera DollyX"]
    #camera_dollyY = params["Camera DollyY"]
    #camera_dollyZ = params["Camera DollyZ"]
    #if figure_name == "female":
    #    fig_height = params["Figure Height"] / 170. * 173
    #elif figure_name == "male":
    #    fig_height = params["Figure Height"] / 170. * 176
    #else:
    #    raise ValueError("Invalid figure name.")

    n_points = X.shape[0]
    pitch = radians(pitch)
    #origin_point_from_camera = np.array([-camera_dollyX, -camera_dollyY, camera_dollyZ])
    #X -= np.tile(origin_point_from_camera, (n_points, 1))
    human_center = np.mean(X, axis=0)
    X -= np.tile(human_center, (n_points, 1))
    X = np.dot(X, rotation_matrix(pitch, "x").T)
    X[:, 2] = -X[:, 2]

    return X


def human_loc2camera_loc(human_center, cpitch):

    x, y, z = human_center
    #print("human_center [%f, %f, %f]" % (x, y, z))
    azim = degrees(atan2(x, z))
    polar = degrees(cpitch - atan2(y, z))
    rad = z / (cos(azim) * cos(polar))
    #print("camera_location [%f, %f, %f]" % (rad, azim, polar))

    return [rad, azim, polar]

