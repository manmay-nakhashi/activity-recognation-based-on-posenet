
PART_NAMES = [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
]

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
    # Background = 18

NUM_KEYPOINTS = len(PART_NAMES)

PART_IDS = {pn: pid for pid, pn in enumerate(PART_NAMES)}

CONNECTED_PART_NAMES = [
    ("leftHip", "leftShoulder"), ("leftElbow", "leftShoulder"),
    ("leftElbow", "leftWrist"), ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"), ("rightHip", "rightShoulder"),
    ("rightElbow", "rightShoulder"), ("rightElbow", "rightWrist"),
    ("rightHip", "rightKnee"), ("rightKnee", "rightAnkle"),
    ("leftShoulder", "rightShoulder"), ("leftHip", "rightHip")
]

CONNECTED_PART_INDICES = [(PART_IDS[a], PART_IDS[b]) for a, b in CONNECTED_PART_NAMES]

LOCAL_MAXIMUM_RADIUS = 1

POSE_CHAIN = [
    ("nose", "leftEye"), ("leftEye", "leftEar"), ("nose", "rightEye"),
    ("rightEye", "rightEar"), ("nose", "leftShoulder"),
    ("leftShoulder", "leftElbow"), ("leftElbow", "leftWrist"),
    ("leftShoulder", "leftHip"), ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"), ("nose", "rightShoulder"),
    ("rightShoulder", "rightElbow"), ("rightElbow", "rightWrist"),
    ("rightShoulder", "rightHip"), ("rightHip", "rightKnee"),
    ("rightKnee", "rightAnkle")
]

PARENT_CHILD_TUPLES = [(PART_IDS[parent], PART_IDS[child]) for parent, child in POSE_CHAIN]

PART_CHANNELS = [
  'left_face',
  'right_face',
  'right_upper_leg_front',
  'right_lower_leg_back',
  'right_upper_leg_back',
  'left_lower_leg_front',
  'left_upper_leg_front',
  'left_upper_leg_back',
  'left_lower_leg_back',
  'right_feet',
  'right_lower_leg_front',
  'left_feet',
  'torso_front',
  'torso_back',
  'right_upper_arm_front',
  'right_upper_arm_back',
  'right_lower_arm_back',
  'left_lower_arm_front',
  'left_upper_arm_front',
  'left_upper_arm_back',
  'left_lower_arm_back',
  'right_hand',
  'right_lower_arm_front',
  'left_hand'
]