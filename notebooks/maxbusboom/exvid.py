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
    
    #print(f"Processing video file '{video_file}' and generating output '{output_file}'")
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
