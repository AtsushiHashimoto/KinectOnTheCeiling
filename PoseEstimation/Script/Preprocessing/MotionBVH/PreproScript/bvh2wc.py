# coding: utf-8

import glob, argparse
from math import cos, sin, atan2, radians
import numpy as np


def adjust_part_length(offset, joint_name, figure_joint_tree):

    figure_joint_names = ["lThigh", "lShin", "lFoot", "lFoot End Site",
                          "rThigh", "rShin", "rFoot", "rFoot End Site",
                          "hip", "abdomen", "chest", "neck", "head", "head End Site",
                          "lCollar", "lShldr", "lForeArm", "lHand", "lHand End Site",
                          "rCollar", "rShldr", "rForeArm", "rHand", "rHand End Site"]

    bvh_joint_names = ["Left Thigh", "Left Shin", "Left Foot", "lToe End Site",
                       "Right Thigh", "Right Shin", "Right Foot", "rToe End Site",
                       "Hip", "Abdomen", "Chest", "Neck", "Head", "Head End Site",
                       "Left Collar", "Left Shoulder", "Left Forearm", "Left Hand", "LFingers End Site",
                       "Right Collar", "Right Shoulder", "Right Forearm", "Right Hand", "RFingers End Site"]

    if joint_name in bvh_joint_names:
        for figure_joint in figure_joint_tree:
            if figure_joint.name == figure_joint_names[bvh_joint_names.index(joint_name)]:
                figure_part_length = np.sqrt(np.sum(np.array(figure_joint.rel_pos) ** 2))
                break

        bvh_part_length = np.sqrt(np.sum(offset ** 2))
        if bvh_part_length != 0:
            offset = offset * figure_part_length / bvh_part_length
        else:
            offset = np.array([0,0,0])
    else:
        offset = np.array([0,0,0])

    return offset


def make_joint_tree(inFile, figure_joint_tree=None):

    depth = 0
    joint_tree = [JointNode('Hip', 0)]
    joint_tree[-1].set_rel_pos([0, 0, 0])
    for line in open(inFile, 'r'):
        items = line.split()
        if items[0] == '{':
            depth += 1
        elif items[0] == '}':
            depth -= 1
        elif items[0] == 'JOINT':
            joint_tree.append(JointNode(" ".join(items[1:]), depth))
            for joint in joint_tree[::-1]:
                if joint.depth < depth:
                    joint_tree[-1].set_parent(joint)
                    break
        elif items[0] == 'End':
            joint_tree.append(JointNode(joint_tree[-1].name + " " + " ".join(items), depth))
            for joint in joint_tree[::-1]:
                if joint.depth < depth:
                    joint_tree[-1].set_parent(joint)
                    break
        elif items[0] == 'OFFSET':
            if len(joint_tree) > 1:
                offset = [float(p) for p in items[1:]]
                if figure_joint_tree is not None:
                    offset = adjust_part_length(np.array(offset), joint_tree[-1].name, figure_joint_tree)
                joint_tree[-1].set_rel_pos(offset)
        elif 'Frame Time:' in line:
            break

    return joint_tree


def angle2pos(joint_angles, joint_tree):

    joint_tree[0].set_abs_pos(joint_angles[:3])
    joint_tree[0].set_rot_mat(joint_angles[3+1], joint_angles[3+2], joint_angles[3+3])
    positions = [str(joint_angles[i]) for i in range(3)]
    i = 2
    for joint_node in joint_tree[1:]:
        if "End Site" in joint_node.name:
            joint_node.set_rot_mat(0, 0, 0)
            joint_node.convert()
        else:
            joint_node.set_rot_mat(joint_angles[3*i+1], joint_angles[3*i+2], joint_angles[3*i+3])
            joint_node.convert()
            i += 1
        positions.extend([str(abs_pos) for abs_pos in joint_node.abs_pos])

    return positions


def make_wc(in_file, out_file, figure_joint_tree):

    joint_tree = make_joint_tree(in_file, figure_joint_tree)

    with open(out_file, 'w') as fout:
        motion_flag = False
        for line in open(in_file, 'r'):
            if 'Frame Time:' in line:
                motion_flag = True
            elif motion_flag:
                joint_angles = [float(p) for p in line.split()]
                positions = angle2pos(joint_angles, joint_tree)
                fout.write(" ".join(positions)+"\n")


class JointNode:

    def __init__(self, name, depth):
        self.name = name
        self.depth = depth
        self.rel_pos = [0.0,0.0,0.0]
        self.abs_pos = [0.0,0.0,0.0]
        self.rot_mat = None
        self.cum_rot_mat = None
        self.parent = None

    def set_parent(self, parent):
        self.parent = parent
    
    def set_rel_pos(self, offset):
        self.rel_pos = offset

    def set_abs_pos(self, posList):
        self.abs_pos = posList

    def set_rot_mat(self, Zrot, Yrot, Xrot):
        Xrot, Yrot, Zrot = radians(Xrot), radians(Yrot), radians(Zrot)
        Cx, Sx, Cy, Sy, Cz, Sz = cos(Xrot), sin(Xrot), cos(Yrot), sin(Yrot), cos(Zrot), sin(Zrot)
        # 各軸に関する回転行列
        Rz = np.array([[Cz, -Sz, 0], [Sz, Cz, 0], [0, 0, 1]])
        Ry = np.array([[Cy, 0, Sy], [0, 1, 0], [-Sy, 0, Cy]])
        Rx = np.array([[1, 0, 0], [0, Cx, -Sx], [0, Sx, Cx]])
        self.rot_mat = Rz.dot(Ry.dot(Rx))
        if self.parent is None:
            testV = np.dot(self.rot_mat, np.array([[0],[0],[1]]))
            phi = atan2(testV[0], testV[2])
            Cp = cos(-phi)
            Sp = sin(-phi)
            norm_rot_mat = np.array([[Cp, 0, Sp], [0, 1, 0], [-Sp, 0, Cp]])
            self.cum_rot_mat = np.dot(norm_rot_mat, self.rot_mat)
        else:
            self.cum_rot_mat = np.dot(self.parent.cum_rot_mat, self.rot_mat)

    def convert(self):
        p_abs_pos = np.array([[self.parent.abs_pos[0]], [self.parent.abs_pos[1]], [self.parent.abs_pos[2]]])
        s_rel_pos = np.array([[self.rel_pos[0]], [self.rel_pos[1]], [self.rel_pos[2]]])
        self.abs_pos = (p_abs_pos + np.dot(self.parent.cum_rot_mat, s_rel_pos)).flatten().tolist()


if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument("-d", type=str, default="../../../../Data/")
    args = p.parse_args()

    data_path = args.d

    in_path = data_path + "Preprocessing/MotionBVH/t_pose/"
    in_app = ".bvh"
    out_path = data_path + "Preprocessing/MotionBVH/WC/"
    out_app = "_pos"
    figure_bvh_path = data_path + "Preprocessing/MotionBVH/"
    BVHs = glob.glob(in_path+"*/*"+in_app)
    file_names = [x.replace(in_path, "").replace(in_app, "") for x in BVHs]
    figure_names = ["female", "male"]
    
    for figure_name in figure_names:

        figure_joint_tree = make_joint_tree(figure_bvh_path+"labeled_"+figure_name+".bvh")

        for file_name in file_names:
        
            in_file = in_path + file_name + in_app
            wc_file = out_path + file_name + "_" + figure_name + out_app
        
            ## 座標変換
            make_wc(in_file, wc_file, figure_joint_tree)
            print(file_name)

