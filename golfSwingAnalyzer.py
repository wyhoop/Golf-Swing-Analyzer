# This program will serve as the main file for implementing my Golf Swing Analyzer
import cv2

def loadVideo():
    cap = cv2.VideoCapture('assets/IMG_5801.MOV')

    while cap.isOpened():
        ret, frame = cap.read()
        resizedFrame = cv2.resize(frame, (0, 0), fx = 0.5, fy = 0.5)

        # If frame is read correctly, ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting...")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame', resizedFrame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

loadVideo()