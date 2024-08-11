#!/usr/bin/env python

import argparse
from pathlib import Path
import os
#from pytube import YouTube
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time 

import matplotlib.animation as animation
from IPython.display import HTML


def show_frame(frame):
    """Display a frame in the notebook"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Display the frame using matplotlib
    plt.imshow(frame_rgb)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()

def extract_traj_per_frame(frame, centroid_func = None, ex_frame=.5):
    if centroid_func:
        centroid = centroid_func(frame)[0]
        return {
            'x': centroid[0],
            'y': centroid[1]
        }
   
def find_circles_in_mask(mask, original_image):
    # Clone the original image to draw on
    output_image = original_image.copy()

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # List to hold all detected circles
    all_circles = []

    for contour in contours:
        # Draw the contour in red
        cv2.drawContours(output_image, [contour], -1, (0, 0, 255), 2)

        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Expand the region by 50%
        x_expanded = max(0, x - int(0.25 * w))
        y_expanded = max(0, y - int(0.25 * h))
        w_expanded = int(1.5 * w)
        h_expanded = int(1.5 * h)

        # Create a sub-mask for the expanded region
        sub_mask = mask[y_expanded:y_expanded + h_expanded, x_expanded:x_expanded + w_expanded]

        # Detect circles in the sub-mask using HoughCircles
        circles = cv2.HoughCircles(sub_mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)
        
        if circles is not None:
            # Convert the circle parameters (x, y, radius) to integers
            circles = np.round(circles[0, :]).astype("int")

            for (cx, cy, r) in circles:
                # Adjust circle coordinates to the original mask's coordinate system
                cx += x_expanded
                cy += y_expanded
                all_circles.append((cx, cy, r))

                # Draw the circle in green
                cv2.circle(output_image, (cx, cy), r, (0, 255, 0), 2)
                cv2.rectangle(output_image, (cx - 5, cy - 5), (cx + 5, cy + 5), (0, 128, 255), -1)
    
    return all_circles, output_image
def calc_background(frames, num_frames=30):
    cap = cv2.VideoCapture(video_path)
    new_frames = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_frames.append(gray)

    cap.release()
    median_frame = np.mean(frames, axis=0).astype(dtype=np.uint8)
    return median_frame
    
def sub_bg(frames):
    background = calc_background(frames)
    
    
    subtracted_frames = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = cv2.absdiff(gray, background)
        _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
        subtracted_frames.append(mask)

    return subtracted_frames

def mask_color(frames, lower_color, upper_color):
    
    masked_frames = []

    for frame in frames:
        
        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Apply the color mask
        mask = cv2.inRange(hsv, lower_color, upper_color)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Convert the masked frame to grayscale
        gray_masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        
        # Threshold the grayscale masked frame to get a binary (1-bit) image
        _, binary_masked_frame = cv2.threshold(gray_masked_frame, 1, 255, cv2.THRESH_BINARY)
        
        # Append the binary masked frame to the list
        masked_frames.append(binary_masked_frame)

    cap.release()
    return masked_frames

# Define color range for the basketball (shades of orange)
lower_m = np.array([5, 100, 100])
upper_m = np.array([15, 255, 255])

#c_frames = mask_color(vids[0], lower_m, upper_m)

#nbg_frames = sub_bg(vids[0])

def and_frames(list1, list2):
    # Check if both lists have the same number of frames
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same number of frames.")
    
    anded_frames = []
    
    for frame1, frame2 in zip(list1, list2):
        # Perform bitwise AND operation
        anded_frame = cv2.bitwise_and(frame1, frame2)
        anded_frames.append(anded_frame)
    
    return anded_frames

def extract_traj(fp, centroid_func=None, ex_frame=.5):
    """Run extraction function and generate dataframe, for each frame in the movie"""
    # Open the video file
    cap = cv2.VideoCapture(fp)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Iterate over the frames
    frame_count = 0
    rows = []
    frame_shown = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if centroid_func:
            centroid, aframe = centroid_func(frame)
            rows.append({
                'x': centroid[0],
                'y': centroid[1]
            })
        else:
            aframe = frame
    
        if not frame_shown and frame_count > n_frames*ex_frame:
            
            show_frame(aframe)

            frame_shown = True
        
    cap.release()

    if rows:
        df = pd.DataFrame(rows)
        
        #t['x'] = t.x.max() - t.x
        df['y'] = df.y.max() - df.y
    
        return df
def find_cannonball_centroid_color(image):
    """Find the centroid for the canon ball in the canon video. """
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color range for the red cannonball
    # FIXME?> I think this should acrtually be a split range for red in  HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([8, 255, 255])

    # Create a mask for the red color
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours found, handle the case
    if len(contours) == 0:
        return None, image

    # Assume the largest contour is the cannonball
    cannonball_contour = max(contours, key=cv2.contourArea)

    # Compute the centroid of the cannonball
    M = cv2.moments(cannonball_contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    # Draw the contour and centroid on the image
    cv2.drawContours(image, [cannonball_contour], -1, (0, 255, 0), 2)
    cv2.circle(image, (cX, cY), 2, (255, 0, 0), -1)
   
    return (cX, cY), image


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="A program that processes a video file and outputs the results to a CSV file.")

    # Add arguments
    parser.add_argument("-v", "--video", type=str, required=True, help="The name of the video file")
    parser.add_argument("-o", "--output", type=str, required=True, help="The name of the output CSV file")

    parser.add_argument("-O", "--one",required=False,
                        action='store_true' ,
                        help="Just run one frame")
    
    parser.add_argument("-k", "--keyframe",  type=int, 
                    help="Specify which key frame")



    # Parse the arguments
    args = parser.parse_args()

    # Print the arguments
    print(f"Video file: {args.video}")
    print(f"Output CSV: {args.output}")

    # Here you can add your code to process the video and generate the output CSV
    if args.one:
        process_one(args.video, args.keyframe)
    else:
        process_video(args.video, args.output)


def yield_frame(video_file, release=True):

    cap = cv2.VideoCapture(video_file)
    rows = []
    
    # Check if the video file opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_file}")
        return

    print("Video file opened successfully.")
    
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        
        yield frame

    if release==True:
        cap.release()


def process_video(video_file, output_file):
    
    #Define the upper and lower bounds for red in the hsv color space
    lower_bound = np.array([0,100,100])
    upper_bound = np.array([0,100,25])
    frames = list(yield_frame(video_file,release=True))
    rows = []
    
    for frame in yield_frame(video_file):
        
        rows.append(extract_traj_per_frame(frame,find_cannonball_centroid_color))
        
        #f = find_cannonball_centroid_color(frame)[1]
        #cv2.imshow('frame', f)
        #time.sleep(.1)

        if cv2.waitKey(1) == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    
    cv2.destroyAllWindows()
    print("Video processing complete.")

    # Add your video processing code here
    df = pd.DataFrame(rows)
    df['y'] = df.y.max() - df.y
    
    #df.plot(range(len(df.x)),"y",kind="line")
    plt.plot(df.index,df.x)
    plt.show()
def process_one(video_file, keyframe):

    frames = list(yield_frame(video_file,release=True))

    if keyframe is None:
        keyframe = len(frames)//2+5

    frame = frames[keyframe]
   
    cv2.imshow('Foobar',frame)

    centroid = extract_traj_per_frame(frame,find_cannonball_centroid_color)
    print(f"x: {centroid["x"]}, y: {centroid["y"]}")
    
    while True:
        if cv2.waitKey(1) == ord('q'):
            return

    #centroid = find_cannonball_centroid_color(frame)





if __name__ == "__main__":
    main()
