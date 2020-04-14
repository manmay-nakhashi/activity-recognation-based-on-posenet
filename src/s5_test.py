#!/usr/bin/env python
# coding: utf-8

'''
Test action recognition on
(1) a video, (2) a folder of images, (3) or web camera.

Input:
    model: model/trained_classifier.pickle

Output:
    result video:    output/${video_name}/video.avi
    result skeleton: output/${video_name}/skeleton_res/XXXXX.txt
    visualization by cv2.imshow() in img_displayer
'''

'''
Example of usage:

(1) Test on video file:
python src/s5_test.py \
    --model_path model/trained_classifier.pickle \
    --data_type video \
    --data_path data_test/exercise.avi \
    --output_folder output
    
(2) Test on a folder of images:
python src/s5_test.py \
    --model_path model/trained_classifier.pickle \
    --data_type folder \
    --data_path data_test/apple/ \
    --output_folder output

(3) Test on web camera:
python3 src/s5_test.py \
    --model_path model/trained_classifier.pickle \
    --data_type webcam \
    --data_path 2 \
    --output_folder output
    
'''


import numpy as np
import cv2
import argparse
import tensorflow as tf
import time

if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    import utils.lib_images_io as lib_images_io
    import utils.lib_plot as lib_plot
    import utils.lib_commons as lib_commons
    from utils.lib_openpose import SkeletonDetector
    from utils.lib_tracker import Tracker
    from utils.lib_tracker import Tracker
    from utils.lib_classifier import ClassifierOnlineTest
    from utils.lib_classifier import *  # Import all sklearn related libraries
    import posenet

def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != "/") else path


# -- Command-line input

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
def numpy_fill(arr):
    df = pd.DataFrame(arr)
    df = df.replace(0, np.nan)
    df.fillna(method='ffill', axis=0, inplace=True)
    df = df.replace(np.nan, 0)
    out = df.as_matrix()
    return out
def get_command_line_arguments():

    def parse_args():
        parser = argparse.ArgumentParser(
            description="Test action recognition on \n"
            "(1) a video, (2) a folder of images, (3) or web camera.")
        parser.add_argument("-m", "--model_path", required=False,
                            default='model/trained_classifier.pickle')
        parser.add_argument("-k", "--keras", required=False,
                            default=True, help="load model True for keras and False for sklearn")
        parser.add_argument("-t", "--data_type", required=False, default='webcam',
                            choices=["video", "folder", "webcam"])
        parser.add_argument("-p", "--data_path", required=False, default="",
                            help="path to a video file, or images folder, or webcam. \n"
                            "For video and folder, the path should be "
                            "absolute or relative to this project's root. "
                            "For webcam, either input an index or device name. ")
        parser.add_argument("-o", "--output_folder", required=False, default='output/',
                            help="Which folder to save result to.")

        args = parser.parse_args()
        return args
    args = parse_args()
    if args.data_type != "webcam" and args.data_path and args.data_path[0] != "/":
        # If the path is not absolute, then its relative to the ROOT.
        args.data_path = ROOT + args.data_path
    return args


def get_dst_folder_name(src_data_type, src_data_path):
    ''' Compute a output folder name based on data_type and data_path.
        The final output of this script looks like this:
            DST_FOLDER/folder_name/vidoe.avi
            DST_FOLDER/folder_name/skeletons/XXXXX.txt
    '''

    assert(src_data_type in ["video", "folder", "webcam"])

    if src_data_type == "video":  # /root/data/video.avi --> video
        folder_name = os.path.basename(src_data_path).split(".")[-2]

    elif src_data_type == "folder":  # /root/data/video/ --> video
        folder_name = src_data_path.rstrip("/").split("/")[-1]

    elif src_data_type == "webcam":
        # month-day-hour-minute-seconds, e.g.: 02-26-15-51-12
        folder_name = lib_commons.get_time_string()

    return folder_name


args = get_command_line_arguments()

SRC_DATA_TYPE = args.data_type
SRC_DATA_PATH = args.data_path
SRC_MODEL_PATH = args.model_path

DST_FOLDER_NAME = get_dst_folder_name(SRC_DATA_TYPE, SRC_DATA_PATH)

# -- Settings

cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
cfg = cfg_all["s5_test.py"]

CLASSES = np.array(cfg_all["classes"])
SKELETON_FILENAME_FORMAT = cfg_all["skeleton_filename_format"]

# Action recognition: number of frames used to extract features.
WINDOW_SIZE = int(cfg_all["features"]["window_size"])

# Output folder
DST_FOLDER = args.output_folder + "/" + DST_FOLDER_NAME + "/"
DST_SKELETON_FOLDER_NAME = cfg["output"]["skeleton_folder_name"]
DST_VIDEO_NAME = cfg["output"]["video_name"]
# framerate of output video.avi
DST_VIDEO_FPS = float(cfg["output"]["video_fps"])


# Video setttings

# If data_type is webcam, set the max frame rate.
SRC_WEBCAM_MAX_FPS = float(cfg["settings"]["source"]
                           ["webcam_max_framerate"])

# If data_type is video, set the sampling interval.
# For example, if it's 3, then the video will be read 3 times faster.
SRC_VIDEO_SAMPLE_INTERVAL = int(cfg["settings"]["source"]
                                ["video_sample_interval"])

# Openpose settings
OPENPOSE_MODEL = cfg["settings"]["openpose"]["model"]
OPENPOSE_IMG_SIZE = cfg["settings"]["openpose"]["img_size"]

# Display settings
img_disp_desired_rows = int(cfg["settings"]["display"]["desired_rows"])


# -- Function


def select_images_loader(src_data_type, src_data_path):
    if src_data_type == "video":
        images_loader = lib_images_io.ReadFromVideo(
            src_data_path,
            sample_interval=SRC_VIDEO_SAMPLE_INTERVAL)

    elif src_data_type == "folder":
        images_loader = lib_images_io.ReadFromFolder(
            folder_path=src_data_path)

    elif src_data_type == "webcam":
        if src_data_path == "":
            webcam_idx = 0
        elif src_data_path.isdigit():
            webcam_idx = int(src_data_path)
        else:
            webcam_idx = src_data_path
        images_loader = lib_images_io.ReadFromWebcam(
            SRC_WEBCAM_MAX_FPS, webcam_idx)
    return images_loader


class MultiPersonClassifier(object):
    ''' This is a wrapper around ClassifierOnlineTest
        for recognizing actions of multiple people.
    '''

    def __init__(self, model_path, classes):

        self.dict_id2clf = {}  # human id -> classifier of this person

        # Define a function for creating classifier for new people.
        self._create_classifier = lambda human_id: ClassifierOnlineTest(
            model_path, classes, WINDOW_SIZE, human_id)

    def classify(self, dict_id2skeleton, pose_classification_model, keras_model):
        ''' Classify the action type of each skeleton in dict_id2skeleton '''

        # Clear people not in view
        old_ids = set(self.dict_id2clf)
        cur_ids = set(dict_id2skeleton)
        humans_not_in_view = list(old_ids - cur_ids)
        for human in humans_not_in_view:
            del self.dict_id2clf[human]

        # Predict each person's action
        id2label = {}
        for id, skeleton in dict_id2skeleton.items():

            if id not in self.dict_id2clf:  # add this new person
                self.dict_id2clf[id] = self._create_classifier(id)

            classifier = self.dict_id2clf[id]
            id2label[id] = classifier.predict(skeleton, pose_classification_model, keras_model)  # predict label
            # print("\n\nPredicting label for human{}".format(id))
            # print("  skeleton: {}".format(skeleton))
            # print("  label: {}".format(id2label[id]))

        return id2label

    def get_classifier(self, id):
        ''' Get the classifier based on the person id.
        Arguments:
            id {int or "min"}
        '''
        if len(self.dict_id2clf) == 0:
            return None
        if id == 'min':
            id = min(self.dict_id2clf.keys())
        return self.dict_id2clf[id]


def remove_skeletons_with_few_joints(skeletons):
    ''' Remove bad skeletons before sending to the tracker '''
    good_skeletons = []
    for skeleton in skeletons:
        px = skeleton[2:2+13*2:2]
        py = skeleton[3:2+13*2:2]
        num_valid_joints = len([x for x in px if x != 0])
        num_leg_joints = len([x for x in px[-6:] if x != 0])
        total_size = max(py) - min(py)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # IF JOINTS ARE MISSING, TRY CHANGING THESE VALUES:
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if num_valid_joints >= 5 and total_size >= 0.1 and num_leg_joints >= 0:
            # add this skeleton only when all requirements are satisfied
            good_skeletons.append(skeleton)
    return good_skeletons


def draw_result_img(img_disp, ith_img, dict_id2skeleton, multiperson_classifier):
    ''' Draw skeletons, labels, and prediction scores onto image for display '''

    # Resize to a proper size for display
    r, c = img_disp.shape[0:2]
    desired_cols = int(1.0 * c * (img_disp_desired_rows / r))
    img_disp = cv2.resize(img_disp,
                          dsize=(desired_cols, img_disp_desired_rows))

    # Draw all people's skeleton
    # skeleton_detector.draw(img_disp, humans)
    scale_h = 1.0
    # Draw bounding box and label of each person
    if len(dict_id2skeleton):
        for id, label in dict_id2label.items():
            skeleton = dict_id2skeleton[id]
            # scale the y data back to original
            skeleton[1::2] = skeleton[1::2] / scale_h
            # print("Drawing skeleton: ", dict_id2skeleton[id], "with label:", label, ".")
            img_disp = lib_plot.draw_action_result(img_disp, id, skeleton, label)

    # Add blank to the left for displaying prediction scores of each class
    img_disp = lib_plot.add_white_region_to_left_of_image(img_disp)

    cv2.putText(img_disp, "Frame:" + str(ith_img),
                (20, 20), fontScale=1.5, fontFace=cv2.FONT_HERSHEY_PLAIN,
                color=(0, 0, 0), thickness=2)

    # Draw predicting score for only 1 person
    if len(dict_id2skeleton):
        classifier_of_a_person = multiperson_classifier.get_classifier(
            id='min')
        classifier_of_a_person.draw_scores_onto_image(img_disp)
    return img_disp


def get_the_skeleton_data_to_save_to_disk(dict_id2skeleton):
    '''
    In each image, for each skeleton, save the:
        human_id, label, and the skeleton positions of length 18*2.
    So the total length per row is 2+36=38
    '''
    skels_to_save = []
    for human_id in dict_id2skeleton.keys():
        label = dict_id2label[human_id]
        skeleton = dict_id2skeleton[human_id]
        skels_to_save.append([[human_id, label] + skeleton.tolist()])
    return skels_to_save


# -- Main
if __name__ == "__main__":

    # -- Detector, tracker, classifier
    sess_posenet = tf.Session()
    model_cfg, model_outputs = posenet.load_model(50, sess_posenet)
    output_stride = model_cfg['output_stride']

    # skeleton_detector = SkeletonDetector(OPENPOSE_MODEL, OPENPOSE_IMG_SIZE)

    multiperson_tracker = Tracker()

    multiperson_classifier = MultiPersonClassifier(SRC_MODEL_PATH, CLASSES)

    # -- Image reader and displayer
    # images_loader = select_images_loader(SRC_DATA_TYPE, SRC_DATA_PATH)
    img_displayer = lib_images_io.ImageDisplayer()
    # logger.debug('cam read+')
    cam = cv2.VideoCapture(0)
    ret_val, image = cam.read()
    # logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    # -- Init output
    if args.keras == True:
        print(args.keras)
        pose_classification_model = load_model(args.model_path)
    else:
        f = open(args.model_path, 'rb')
        pose_classification_model = pickle.load(f)
    if pose_classification_model is None:
        print("my Error: failed to load model")
        assert False
    # output folder
    os.makedirs(DST_FOLDER, exist_ok=True)
    os.makedirs(DST_FOLDER + DST_SKELETON_FOLDER_NAME, exist_ok=True)

    # video writer
    video_writer = lib_images_io.VideoWriter(
        DST_FOLDER + DST_VIDEO_NAME, DST_VIDEO_FPS)

    # -- Read images and process
    try:
        ith_img = -1
        while 1:
            start_time = time.time()
            # -- Read image
            # img = images_loader.read_image()
            ret_val, img = cam.read()
            ith_img += 1
            img_disp = img.copy()
            # print(f"\nProcessing {ith_img}th image ...")
            posenet_image, source_img, scale = _process_input(img, 1, output_stride)
            # -- Openpose Detect skeletons
            # humans = skeleton_detector.detect(img)
            # skeletons, scale_h = skeleton_detector.humans_to_skels_list(humans)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess_posenet.run(
                model_outputs,feed_dict={'image:0': posenet_image})
            #--posenet Detect skeletons
            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=3,
                min_pose_score=0.2)
            
            overlay_image = posenet.draw_skel_and_kp(
                img, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.2, min_part_score=0.1)

            adjacent_keypoints = []
            skeletons = []
            for ii, score in enumerate(pose_scores):
                if score < 0.2:
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
                if cv_keypoints:
                    openpose_keypoints = posenet_to_openpose(cv_keypoints)
                else:
                    continue

                skeleton = openpose_keypoints
                skeletons.append(skeleton)
            # skeletons = remove_skeletons_with_few_joints(skeletons)

            # -- Track people
            dict_id2skeleton = multiperson_tracker.track(
                skeletons)  # int id -> np.array() skeleton
            # print(dict_id2skeleton)
            # -- Recognize action of each person
            if len(dict_id2skeleton):
                dict_id2label = multiperson_classifier.classify(
                    dict_id2skeleton, pose_classification_model, args.keras)
            # print(dict_id2label)
            # -- Draw
            img_disp = draw_result_img(overlay_image, ith_img, dict_id2skeleton, multiperson_classifier)

            # Print label of a person
            if len(dict_id2skeleton):
                min_id = min(dict_id2skeleton.keys())
                print("prediced label is :", dict_id2label[min_id])

            # -- Display image, and write to video.avi
            img_displayer.display(img_disp, wait_key_ms=1)
            video_writer.write(img_disp)
            print("FPS: ", 1.0 / (time.time() - start_time))
            # -- Get skeleton data and save to file
            skels_to_save = get_the_skeleton_data_to_save_to_disk(
                dict_id2skeleton)
            lib_commons.save_listlist(
                DST_FOLDER + DST_SKELETON_FOLDER_NAME +
                SKELETON_FILENAME_FORMAT.format(ith_img),
                skels_to_save)
    finally:
        video_writer.stop()
        print("Program ends")
