import mediapipe as mp
import numpy
import cv2
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

folder = "C:/Users/werho/Golf Swing Analyzer/Golf-Swing-Analyzer/output_folder/No OpenPose Yet/Top"

for file in os.listdir(folder):
    file_name = os.path.join(folder, file)
    image = cv2.imread(file_name)

    if image is None:
        print("Image could not be loaded")

    else:
        with mp_pose.Pose(
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5) as pose:
        
            # Process the image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            # Draw landmarks on the image
            image.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            # Get the image dimensions to resize proportionally
            height, width = image.shape[:2]

            # Resize to half the original size
            image_resized = cv2.resize(image, (width // 2, height // 2))

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Pose', cv2.flip(image_resized, 1))
            cv2.imwrite(file_name, image)
            cv2.waitKey(0)  # Wait until a key is pressed

# Clean up
cv2.destroyAllWindows()