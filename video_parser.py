import cv2
import os

def extract_key_frames(video_folder_path, output_folder):

    for video in os.listdir(video_folder_path):
        if video.endswith('.mp4'):
            video_path = os.path.join(video_folder_path, video)
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print("Error opening video file.")
                return

            # Frame storage parameters
            frame_count = 0
            extracted_frames = []
            i = 1

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Perform motion detection, or another method to identify key moments
                # Here, I am extracting frames at specific intervals for simplicity
                if frame_count % 1 == 0:  # Captures every frame of the video clip
                    frame_name = os.path.join(output_folder, f'{os.path.splitext(video)[0]}_frame_{frame_count}.jpg')
                    cv2.imwrite(frame_name, frame)
                    extracted_frames.append(frame_name)

                frame_count += 1
                i += 1

    cap.release()
    return extracted_frames

# Example usage
extract_key_frames('assets/Training Data and Videos/Swing Videos/4-iron clips/Top', 'output_folder/4-iron without openpose/Top')