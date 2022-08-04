import numpy as np
import math
import cv2

body_point = [[0, 15, 17], [0, 16, 18], [0, 1, 8], [1, 2, 3, 4], [1, 5, 6, 7], [8, 12, 13, 14, 19, 20],
              [8, 9, 10, 11, 22, 23], [11, 24], [14, 21]]
# 24 points, 9 lambs
hand_point = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
# 20 points, 5 fingers
part_list = ['pose_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']
pre_pose = None
crop_size = 25


def get_label(img, pose):
    colors = list(range(1, 100))
    colors = iter(colors)
    label = np.zeros(img.shape[0:2], np.uint8)
    pose = update_pose(pose)
    for limb_index in range(len(body_point)):
        joint_list = []
        for point_index in body_point[limb_index]:
            joint_list.append([(pose[part_list[0]][3 * point_index]),
                               (pose[part_list[0]][3 * point_index + 1])])
        for line_index in range(len(joint_list) - 1):
            joint_coords = tuple(joint_list[line_index: line_index + 2])
            coords_center = tuple(np.round(np.mean(joint_coords, 0)).astype(int))
            limb_dir = (joint_coords[0][0] - joint_coords[1][0], joint_coords[0][1] - joint_coords[1][1])
            limb_length = np.linalg.norm(limb_dir)
            angle = math.degrees(math.atan2(limb_dir[1], limb_dir[0]))
            polygon = cv2.ellipse2Poly(coords_center, (int(limb_length / 2), 4), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(label, polygon, color=next(colors))
    for part in part_list[1:]:
        for limb_index in range(len(hand_point)):
            joint_list = []
            for point_index in hand_point[limb_index]:
                joint_list.append((int(pose[part][3 * point_index]),
                                   int(pose[part][3 * point_index + 1])))
            for line_index in range(len(joint_list) - 1):
                joint_coords = tuple(joint_list[line_index: line_index + 2])
                cv2.line(label, joint_coords[0], joint_coords[1], color=next(colors), thickness=2, lineType=8)
    head_index = 0
    hand_index = 9
    lfoot_index = 19
    rfoot_index = 22
    head = [adjust_cord(pose[part_list[0]][head_index * 3]), adjust_cord(pose[part_list[0]][head_index * 3 + 1])]
    lhand = [adjust_cord(pose[part_list[1]][hand_index * 3]), adjust_cord(pose[part_list[1]][hand_index * 3 + 1])]
    rhand = [adjust_cord(pose[part_list[2]][hand_index * 3]), adjust_cord(pose[part_list[2]][hand_index * 3 + 1])]
    lfoot = [adjust_cord(pose[part_list[0]][lfoot_index * 3]), adjust_cord(pose[part_list[0]][lfoot_index * 3 + 1])]
    rfoot = [adjust_cord(pose[part_list[0]][rfoot_index * 3]), adjust_cord(pose[part_list[0]][rfoot_index * 3 + 1])]
    head = np.array(head, dtype=np.float64)
    lhand = np.array(lhand, dtype=np.float64)
    rhand = np.array(rhand, dtype=np.float64)
    lfoot = np.array(lfoot, dtype=np.float64)
    rfoot = np.array(rfoot, dtype=np.float64)

    lear_index = 18
    rear_index = 17
    lear = [pose[part_list[0]][lear_index * 3], pose[part_list[0]][lear_index * 3 + 1]]
    rear = [pose[part_list[0]][rear_index * 3], pose[part_list[0]][rear_index * 3 + 1]]
    head_len = (lear[0] - rear[0]) ** 2 + (lear[1] - rear[1]) ** 2

    leye_index = 16
    reye_index = 15
    leye = [pose[part_list[0]][leye_index * 3], pose[part_list[0]][leye_index * 3 + 1]]
    reye = [pose[part_list[0]][reye_index * 3], pose[part_list[0]][reye_index * 3 + 1]]
    eye_len = (leye[0] - reye[0]) ** 2 + (leye[1] - reye[1]) ** 2

    nose_index = 0
    nose = [pose[part_list[0]][nose_index * 3], pose[part_list[0]][nose_index * 3 + 1]]
    nose_leye = (leye[0] - nose[0]) ** 2 + (leye[1] - nose[1]) ** 2
    nose_reye = (reye[0] - nose[0]) ** 2 + (reye[1] - nose[1]) ** 2
    nose_lear = (lear[0] - nose[0]) ** 2 + (lear[1] - nose[1]) ** 2
    nose_rear = (rear[0] - nose[0]) ** 2 + (rear[1] - nose[1]) ** 2

    return label, head, lhand, rhand, lfoot, rfoot, \
           head_len, eye_len, nose_leye, nose_reye, nose_lear, nose_rear


def update_pose(pose):
    global pre_pose
    pose = pose['people'][0]
    if pre_pose is not None:
        for part in part_list:
            for i in range(0, len(pose[part]), 3):
                if pose[part][i + 2] == 0:
                    pose[part][i:i + 3] = pre_pose[part][i:i + 3]
    pre_pose = pose
    return pose


def adjust_cord(x):
    if x < crop_size:
        return crop_size
    elif x > 512 - crop_size:
        return 512 - crop_size
    else:
        return x
