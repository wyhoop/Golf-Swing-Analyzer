# This program will serve as the main file for implementing my Golf Swing Analyzer
import cv2
import numpy as np
import math
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import os

# MEDIAPIPE JOINT NUMBERINGS
# 0 - nose
# 1 - left eye (inner)
# 2 - left eye
# 3 - left eye (outer)
# 4 - right eye (inner)
# 5 - right eye
# 6 - right eye (outer)
# 7 - left ear
# 8 - right ear
# 9 - mouth (left)
# 10 - mouth (right)
# 11 - left shoulder
# 12 - right shoulder
# 13 - left elbow
# 14 - right elbow
# 15 - left wrist
# 16 - right wrist
# 17 - left pinky
# 18 - right pinky
# 19 - left index
# 20 - right index
# 21 - left thumb
# 22 - right thumb
# 23 - left hip
# 24 - right hip
# 25 - left knee
# 26 - right knee
# 27 - left ankle
# 28 - right ankle
# 29 - left heel
# 30 - right heel
# 31 - left foot index (toe)
# 32 - right foot index (toe)

# Initialize MediaPipe pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode = True)  # Static mode for pre-saved frames

#landmark_list = [11, 12, 15, 16, 23, 24, 27, 28]

# Trying a dictionary so I can use english values for readability
landmark_dictionary = {
    11: "Left Shoulder",
    12: "Right Shoulder",
    13: "Left Elbow",
    14: "Right Elbow",
    15: "Left Wrist", 
    16: "Right Wrist",
    23: "Left Hip",
    24: "Right Hip",
    25: "Left Knee",
    26: "Right Knee",
    27: "Left Ankle",
    28: "Right Ankle"
}

def get_joint_coordinates():
    landmark_coordinate_list = []
    source_folder = "C:/Users/werho/Golf Swing Analyzer/Golf-Swing-Analyzer/output_folder/9-iron with openpose/Good Swing Frames/Swing 2/Super Simple Test"
    
    for folder in os.listdir(source_folder):
        full_path = os.path.join(source_folder, folder)

        if os.path.isfile(full_path):  # Ensure it's a file
            print("File: ", full_path)
            image = cv2.imread(full_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_image)

            if result.pose_landmarks:
                for ids in landmark_dictionary:
                    name = landmark_dictionary[ids]
                    landmark = result.pose_landmarks.landmark[ids]
                    landmark_coordinate_list.append(landmark)
                    print(f'{name}: (x={landmark.x}, y={landmark.y}, z={landmark.z})') 
                    
            else:
                print(f"No pose landmarks detected for {folder}")
                    

    cv2.destroyAllWindows()
    return landmark_coordinate_list

def find_angles_between_joints(landmark_coordinate_list):
    left_shoulder_coords = landmark_coordinate_list[0] 
    right_shoulder_coords = landmark_coordinate_list[1]
    left_wrist_coords = landmark_coordinate_list[2]
    right_wrist_coords = landmark_coordinate_list[3]
    left_elbow_coords = landmark_coordinate_list[4]
    right_elbow_coords = landmark_coordinate_list[5]
    left_hip_coords = landmark_coordinate_list[6]
    right_hip_coords = landmark_coordinate_list[7]
    left_knee_coords = landmark_coordinate_list[8]
    right_knee_coords = landmark_coordinate_list[9]
    left_ankle_coords = landmark_coordinate_list[10]
    right_ankle_coords = landmark_coordinate_list[11]

    print("Right Ankle Coordinates: ", right_ankle_coords)
    
    # I did NOT want to do calculus......
    # Get the vectors which are: 
    #   Upper Arm (Bicep Portion)
    #   Torso (Trunk)

# VECTOR SET 1 Angle between upper arm and torso
    # Vector 1
    UA = (left_elbow_coords.x - left_shoulder_coords.x, 
          left_elbow_coords.y - left_shoulder_coords.y,
          left_elbow_coords.z - left_shoulder_coords.z)
    
    # Vector 2
    torso = (left_shoulder_coords.x - left_hip_coords.x,
             left_shoulder_coords.y - left_hip_coords.y,
             left_shoulder_coords.z - left_hip_coords.z)

    dot_product = UA[0] * torso[0] + UA[1] * torso[1] + UA[2] * torso[2]

    UA_magnitude = math.sqrt(UA[0]**2 + UA[1]**2 + UA[2]**2)
    torso_magnitude = math.sqrt(torso[0]**2 + torso[1]**2 + torso[2]**2)

    cosine_theta = dot_product / (UA_magnitude * torso_magnitude)

    angle_in_radians = math.acos(cosine_theta)
    angle_in_degrees = math.degrees(angle_in_radians)
    

# VECTOR SET 2 Angle between torso and thigh
    # Vector 1
    UA = (left_elbow_coords.x - left_shoulder_coords.x, 
          left_elbow_coords.y - left_shoulder_coords.y,
          left_elbow_coords.z - left_shoulder_coords.z)
    
    # Vector 2
    torso = (left_shoulder_coords.x - left_hip_coords.x,
             left_shoulder_coords.y - left_hip_coords.y,
             left_shoulder_coords.z - left_hip_coords.z)

    dot_product = (UA[0] * torso[0]) + (UA[1] * torso[1]) + (UA[2] * torso[2])

    UA_magnitude = math.sqrt(UA[0]**2 + UA[1]**2 + UA[2]**2)
    torso_magnitude = math.sqrt(torso[0]**2 + torso[1]**2 + torso[2]**2)

    cosine_theta = dot_product / (UA_magnitude * torso_magnitude)

    angle_in_radians = math.acos(cosine_theta)
    angle_in_degrees = math.degrees(angle_in_radians)

# VECTOR SET 3 Angle between thigh and calf

    print("Angle in degrees: ", angle_in_degrees)
    print("Dot Product: ", dot_product)
    # print("left Shoulder: ", left_shoulder_coords.x)
    print("UA: ", UA)
    print("Torso: ", torso)
    print("Radians: ", angle_in_radians)
    print("Magnitudes: ", UA_magnitude, torso_magnitude)


landmark_coordinate_list = get_joint_coordinates()
print(landmark_coordinate_list[11])
find_angles_between_joints(landmark_coordinate_list)
