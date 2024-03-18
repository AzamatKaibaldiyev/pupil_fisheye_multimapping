import cv2
import numpy as np
import os
from datetime import datetime
import json
import shutil
import sys
import datetime
import math

def milliseconds_to_hms(milliseconds):
    return str(datetime.timedelta(milliseconds=milliseconds))


def find_matches(kp1, des1, frame, sift, flann, img_name):

    # convert the frame to grayscale
    origFrame = frame.copy()
    frame_gray = cv2.cvtColor(origFrame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute descriptors for the current frame
    kp2, des2 = sift.detectAndCompute(frame_gray, None)

    # Use FLANN matcher to find matches between descriptors
    #print('---------------')
    #print(f"Ref img {img_name} feautures: {len(kp1)}")
    #print("Frame feautures: ", len(kp2))

    try: 
        matches = flann.knnMatch(des1, des2, k=2)
        #outt_path = '/Users/azamatkaibaldiyev/GREYC_project/Tobii2_myvideos/grc73it/recordings/zs7is5l/segments/1/Result_outputs/Preprocessed_divided/'
        #cv2.imwrite(os.path.join(outt_path, 'last_frame.jpeg'), frame_gray)

        # Apply ratio test to get good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:   # 0-1 lower values more conservative
                good_matches.append(m)

        #print('{img_name} Number of good matches: ', len(good_matches))
    except Exception as e:
        #print(f"An error occurred, on number of good matches: {e}")
        good_matches = [0]

    return len(good_matches)


def save_video_portion(video_path, reference_image_path, output_folder, img_name, start_frame, end_frame, start_timestamp, end_timestamp):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    output_frames = []

    # Set the video capture position to the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

    # Check if the video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        exit()


    # Create a folder for the current portion
    output_path = os.path.join(output_folder, f"portion_{img_name}")
    os.makedirs(output_path, exist_ok=True)

    # Save frames as a new video in the portion folder
    output_video_path = os.path.join(output_path, f"portion_{img_name}.mp4")

    # Get video properties for the writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed  *'divx' *'mp4v'
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Read frames until the end frame is reached
    while cap.get(cv2.CAP_PROP_POS_FRAMES) <= int(end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        output_frames.append(frame)

    cap.release()
    writer.release()

    print(f"""
          Frames saved from {start_frame} to {end_frame}"
          Timestamp from {start_timestamp} ms to {end_timestamp} ms
          Total number of frames: {len(output_frames)}
          TotalTime: {milliseconds_to_hms(end_timestamp-start_timestamp)}
          """)

    # Save frame information to a JSON file
    frame_info = {
        "ReferenceImageIndex": img_name,
        "StartFrame": start_frame,
        "EndFrame": end_frame,
        "TotalFrames": len(output_frames),
        "StartTimestamp": start_timestamp,
        "EndTimestamp": end_timestamp,
        "StartTime": milliseconds_to_hms(start_timestamp),
        "EndTime": milliseconds_to_hms(end_timestamp),
        "TotalTime": milliseconds_to_hms(end_timestamp-start_timestamp)
    }

    with open(os.path.join(output_path, "frame_info.json"), "w") as info_file:
        json.dump(frame_info, info_file, indent=4)

    # Copy the reference image into the current folder
    shutil.copy(reference_image_path, output_path)



def process_video(video_path, reference_image_path, output_folder, start_frame=None):

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    sift = cv2.SIFT_create()

    # Set start frame position if provided
    if start_frame is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Set up FLANN parameters
    FLANN_INDEX_KDTREE = 0    # before 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5) 
    search_params = dict(checks=100)  # Higher values of checks lead to more accurate results but also result in longer search times
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Initialize variables for tracking portions
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # number of total frames in the video
    img_name = reference_image_path.split('/')[-1].split('.')[0]
    consecutive_matches_count = 0
    consecutive_matches_threshold = 4  # Adjust this threshold as needed
    padding_frames = 0 # Extending purposes: number of frames to subtract from start of the video and add to end of the video
    check_matches_every = 30 #check every number of frames
    num_matches_threshold = 16
    start_frame = None
    end_frame = None
    start_timestamp = 0
    end_timestamp = 0

    # convert the reference image to grayscale
    reference_image =cv2.imread(reference_image_path)
    refImgColor = reference_image.copy()      # store a color copy of the image
    refImg = cv2.cvtColor(refImgColor, cv2.COLOR_BGR2GRAY)  # convert the orig to bw
    # Detect keypoints and compute descriptors for the reference image
    ref_kp1, ref_des1 = sift.detectAndCompute(refImg, None)

    while True:
        # Read the next frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Check matches every N frames
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % check_matches_every == 0:
            # Find matches
            num_matches = find_matches(ref_kp1, ref_des1, frame, sift, flann, img_name)

            # Get the current frame number
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Print the current frame number and total number of frames
            # print(f"""
            #       **************
            #       {img_name}
            #       Frame: {current_frame}/{total_frames}
            #       Number of matches: {num_matches}""")

            # Adjust the threshold as needed
            if num_matches > num_matches_threshold:
                # Count consecutive matches
                consecutive_matches_count += 1
                # print(f"""
                #       ___________________________
                #       {img_name} 
                #       Frame: {current_frame}/{total_frames}
                #       Number of good matches: {num_matches})
                #       Consecutive matches: {consecutive_matches_count}""")

                # If consecutive matches exceed the threshold, consider it a valid transition
                if consecutive_matches_count >= consecutive_matches_threshold:
                    # Update only once
                    if start_frame is None:
                        if current_frame>padding_frames:
                            offset = check_matches_every * consecutive_matches_threshold
                            start_frame = current_frame - offset #- padding_frames
                            start_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) - 1000*offset/math.ceil(cap.get(cv2.CAP_PROP_FPS))#- padding_frames / cap.get(cv2.CAP_PROP_FPS)
                            print(f"Offset time: {offset/cap.get(cv2.CAP_PROP_FPS)}")

                        else:
                            start_frame = 0 
                            start_timestamp = 0

                    end_frame = current_frame #+ padding_frames 
                    end_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) #+ padding_frames / cap.get(cv2.CAP_PROP_FPS)

            else:
                consecutive_matches_count = 0


    # Save the video between start frame to end frame
    print(f"Saving the video for reference image {img_name}")
    save_video_portion(video_path, reference_image_path, output_folder, img_name, 
                        start_frame, end_frame, start_timestamp, end_timestamp)

    # Release the video capture object
    cap.release()

if __name__ == "__main__":

    if len(sys.argv)==4:

        video_path = sys.argv[1]
        reference_image = sys.argv[2]
        output_folder = sys.argv[3]
        # Change the start frame or None
        start_frame = None

        process_video(video_path, reference_image, output_folder, start_frame)
        
    else:
        print("Problem with arguments")


