#!/usr/bin/env python
# coding: utf-8

'''
Read training images based on `valid_images.txt` and then detect skeletons.
    
In each image, there should be only 1 person performing one type of action.
Each image is named as 00001.jpg, 00002.jpg, ...

An example of the content of valid_images.txt is shown below:
    
    jump_03-12-09-18-26-176
    58 680

    jump_03-13-11-27-50-720
    65 393

    kick_03-02-12-36-05-185
    54 62
    75 84

The two indices (such as `56 680` in the first `jump` example)
represents the starting index and ending index of a certain action.

Input:
    SRC_IMAGES_DESCRIPTION_TXT
    SRC_IMAGES_FOLDER
    
Output:
    DST_IMAGES_INFO_TXT
    DST_DETECTED_SKELETONS_FOLDER
    DST_VIZ_IMGS_FOLDER
'''

import cv2
import yaml
import numpy as np
import tensorflow as tf
if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    from utils.lib_openpose import SkeletonDetector
    from utils.lib_tracker import Tracker
    from utils.lib_skeletons_io import ReadValidImagesAndActionTypesByTxt
    import utils.lib_commons as lib_commons
    import posenet

def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != "/") else path

# -- Settings


cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
cfg = cfg_all["s1_get_skeletons_from_training_imgs.py"]

IMG_FILENAME_FORMAT = cfg_all["image_filename_format"]
SKELETON_FILENAME_FORMAT = cfg_all["skeleton_filename_format"]


# Input
if True:
    SRC_IMAGES_DESCRIPTION_TXT = par(cfg["input"]["images_description_txt"])
    SRC_IMAGES_FOLDER = par(cfg["input"]["images_folder"])

# Output
if True:
    # This txt will store image info, such as index, action label, filename, etc.
    # This file is saved but not used.
    DST_IMAGES_INFO_TXT = par(cfg["output"]["images_info_txt"])

    # Each txt will store the skeleton of each image
    DST_DETECTED_SKELETONS_FOLDER = par(
        cfg["output"]["detected_skeletons_folder"])

    # Each image is drawn with the detected skeleton
    DST_VIZ_IMGS_FOLDER = par(cfg["output"]["viz_imgs_folders"])

# Openpose
if True:
    OPENPOSE_MODEL = cfg["openpose"]["model"]
    OPENPOSE_IMG_SIZE = cfg["openpose"]["img_size"]

# -- Functions
def posenet_to_openpose(cv_keypoints):
    openpose_points = np.zeros(36)
    # Nose = 0 (0,1)
    # Neck = 1 (12 + 10 / 2 , 13 + 11 / 2)
    # RShoulder = 2 (12,13)
    # RElbow = 3(16,17)
    # RWrist = 4 (20,21)
    # LShoulder = 5 (10,11)
    # LElbow = 6 (14,15)
    # LWrist = 7 (18,19)
    # RHip = 8 (24,25)
    # RKnee = 9 (28,29)
    # RAnkle = 10 (32,33)
    # LHip = 11 (22,23)
    # LKnee = 12 (26,27)
    # LAnkle = 13 (30,31)
    # REye = 14 (4,5)
    # LEye = 15 (2,3)
    # REar = 16 (8,9)
    # LEar = 17 (6,7)
    openpose_points[0] = cv_keypoints[0]
    openpose_points[1] = cv_keypoints[1]
    if cv_keypoints[10] and cv_keypoints[12]:
        openpose_points[2] = int((cv_keypoints[10] + cv_keypoints[12]) / 2)
    else:
        openpose_points[2]
    if cv_keypoints[11] and cv_keypoints[13]:
        openpose_points[3] = int((cv_keypoints[11] + cv_keypoints[13]) / 2)
    else:
        openpose_points[2]
    openpose_points[4] = cv_keypoints[12]
    openpose_points[5] = cv_keypoints[13]
    openpose_points[6] = cv_keypoints[16]
    openpose_points[7] = cv_keypoints[17]
    openpose_points[8] = cv_keypoints[20]
    openpose_points[9] = cv_keypoints[21]

    openpose_points[10] = cv_keypoints[10]
    openpose_points[11] = cv_keypoints[11]
    openpose_points[12] = cv_keypoints[14]
    openpose_points[13] = cv_keypoints[15]
    openpose_points[14] = cv_keypoints[18]
    openpose_points[15] = cv_keypoints[19]
    openpose_points[16] = cv_keypoints[24]
    openpose_points[17] = cv_keypoints[25]
    openpose_points[18] = cv_keypoints[28]
    openpose_points[19] = cv_keypoints[29]
    openpose_points[20] = cv_keypoints[32]
    openpose_points[21] = cv_keypoints[33]
    openpose_points[22] = cv_keypoints[22]
    openpose_points[23] = cv_keypoints[23]
    openpose_points[24] = cv_keypoints[26]
    openpose_points[25] = cv_keypoints[27]
    openpose_points[26] = cv_keypoints[30]
    openpose_points[27] = cv_keypoints[31]
    openpose_points[28] = cv_keypoints[4]
    openpose_points[29] = cv_keypoints[5]
    openpose_points[30] = cv_keypoints[2]
    openpose_points[31] = cv_keypoints[3]
    openpose_points[32] = cv_keypoints[8]
    openpose_points[33] = cv_keypoints[9]
    openpose_points[34] = cv_keypoints[6]
    openpose_points[35] = cv_keypoints[7]

    return openpose_points.tolist()
def valid_resolution(width, height, output_stride=16):
    target_width = (int(width) // output_stride) * output_stride + 1
    target_height = (int(height) // output_stride) * output_stride + 1
    return target_width, target_height

def _process_input(source_img, scale_factor=1.0, output_stride=16):
    target_width, target_height = valid_resolution(
        source_img.shape[1] * scale_factor, source_img.shape[0] * scale_factor, output_stride=output_stride)
    scale = np.array([source_img.shape[0] / target_height, source_img.shape[1] / target_width])

    input_img = cv2.resize(source_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = input_img * (2.0 / 255.0) - 1.0
    input_img = input_img.reshape(1, target_height, target_width, 3)
    return input_img, source_img, scale

def get_adjacent_keypoints(keypoint_scores, keypoint_coords, min_confidence=0.1):
    results = []
    for left, right in posenet.CONNECTED_PART_INDICES:
        if keypoint_scores[left] < min_confidence or keypoint_scores[right] < min_confidence:
            continue
        results.append(
            np.array([keypoint_coords[left][::-1], keypoint_coords[right][::-1]]).astype(np.int32),
        )
    return results
def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

class ImageDisplayer(object):
    ''' A simple wrapper of using cv2.imshow to display image '''

    def __init__(self):
        self._window_name = "cv2_display_window"
        cv2.namedWindow(self._window_name)

    def display(self, image, wait_key_ms=1):
        cv2.imshow(self._window_name, image)
        cv2.waitKey(wait_key_ms)

    def __del__(self):
        cv2.destroyWindow(self._window_name)


# -- Main
if __name__ == "__main__":

    # -- Openpose
    # skeleton_detector = SkeletonDetector(OPENPOSE_MODEL, OPENPOSE_IMG_SIZE)
    #Posenet
    sess_posenet = tf.Session()
    model_cfg, model_outputs = posenet.load_model(101, sess_posenet)
    output_stride = model_cfg['output_stride']
    
    multiperson_tracker = Tracker()

    # -- Image reader and displayer
    images_loader = ReadValidImagesAndActionTypesByTxt(
        img_folder=SRC_IMAGES_FOLDER,
        valid_imgs_txt=SRC_IMAGES_DESCRIPTION_TXT,
        img_filename_format=IMG_FILENAME_FORMAT)
    # This file is not used.
    images_loader.save_images_info(filepath=DST_IMAGES_INFO_TXT)
    img_displayer = ImageDisplayer()

    # -- Init output path
    os.makedirs(os.path.dirname(DST_IMAGES_INFO_TXT), exist_ok=True)
    os.makedirs(DST_DETECTED_SKELETONS_FOLDER, exist_ok=True)
    os.makedirs(DST_VIZ_IMGS_FOLDER, exist_ok=True)

    # -- Read images and process
    num_total_images = images_loader.num_images
    for ith_img in range(num_total_images):

        # -- Read image
        img, str_action_label, img_info = images_loader.read_image()

        
        # humans = skeleton_detector.detect(img)

        # -- Posenet Detect
        posenet_image, source_img, scale = _process_input(img, 1, output_stride)
        
        print("Posenet Image Shape: ", posenet_image.shape)

        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess_posenet.run(
                model_outputs,feed_dict={'image:0': posenet_image})

        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=5,
                min_pose_score=0.15)
            

        # -- Openpose Draw
        img_disp = img.copy()
        # skeleton_detector.draw(img_disp, humans)
        
        # -- Posenet Draw
        overlay_image = posenet.draw_skel_and_kp(
                img_disp, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)
        img_displayer.display(overlay_image, wait_key_ms=1)

        # -- Posenet Get skeleton data
        adjacent_keypoints = []
        skeletons = []
        for ii, score in enumerate(pose_scores):
            if score < 0.15:
                continue

            new_keypoints = get_adjacent_keypoints(
                keypoint_scores[ii, :], keypoint_coords[ii, :, :], 0.1)
            adjacent_keypoints.extend(new_keypoints)
            cv_keypoints = []
            for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
                if ks < 0.1:
                    cv_keypoints.append(0)
                    cv_keypoints.append(0)
                    continue
                if len(cv_keypoints) == 2:
                    cv_keypoints.append(0)
                    cv_keypoints.append(0)
                cv_keypoints.append(int(kc[1]))
                cv_keypoints.append(int(kc[0]))
            print(len(cv_keypoints)) 
            if cv_keypoints:
                openpose_keypoints = posenet_to_openpose(cv_keypoints)
            else:
                continue

            skeleton = openpose_keypoints
        skeletons.append(skeleton)
        #-- Openpose Get skeleton data
        # skeletons, scale_h = skeleton_detector.humans_to_skels_list(humans)

        dict_id2skeleton = multiperson_tracker.track(
            skeletons)  # dict: (int human id) -> (np.array() skeleton)

        skels_to_save = [img_info + skeleton.tolist()
                         for skeleton in dict_id2skeleton.values()]

        # -- Save result

        # Save skeleton data for training
        filename = SKELETON_FILENAME_FORMAT.format(ith_img)
        lib_commons.save_listlist(
            DST_DETECTED_SKELETONS_FOLDER + filename,
            skels_to_save)

        # Save the visualized image for debug
        filename = IMG_FILENAME_FORMAT.format(ith_img)
        cv2.imwrite(
            DST_VIZ_IMGS_FOLDER + filename,
            img_disp)

        print(f"{ith_img}/{num_total_images} th image "
              f"has {len(skeletons)} people in it")

    print("Program ends")
