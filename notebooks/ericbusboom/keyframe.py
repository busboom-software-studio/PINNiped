#!/usr/bin/env python

import cv2
import sys

def display_middle_frame(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        sys.exit()

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the middle frame number
    middle_frame_number = total_frames // 2

    # Set the current frame position to the middle frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_number)

    # Read the middle frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read the middle frame.")
        sys.exit()

    # Display the middle frame
    cv2.imshow("Middle Frame", frame)

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Release the video capture object
    cap.release()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <video_path>")
        sys.exit()

    video_path = sys.argv[1]
    display_middle_frame(video_path)