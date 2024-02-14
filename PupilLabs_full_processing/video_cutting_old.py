import cv2
import numpy as np
import os
from datetime import datetime
import argparse
from PIL import Image
import json
import shutil
import time

def find_matches(reference_image, frame, sift, flann, img_name):


    # convert the reference image to grayscale
    refImgColor = reference_image.copy()      # store a color copy of the image
    refImg = cv2.cvtColor(refImgColor, cv2.COLOR_BGR2GRAY)  # convert the orig to bw


    # convert the frame to grayscale
    origFrame = frame.copy()
    frame_gray = cv2.cvtColor(origFrame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute descriptors for the reference image and the current frame
    kp1, des1 = sift.detectAndCompute(refImg, None)
    kp2, des2 = sift.detectAndCompute(frame_gray, None)

    # Use FLANN matcher to find matches between descriptors
    print('---------------')
    print('Features: ')
    print(f"Ref img {img_name} ftrs: {len(kp1)}")
    print("Frame ftrs: ", len(kp2))

    try: 
        matches = flann.knnMatch(des1, des2, k=2)
        #outt_path = '/Users/azamatkaibaldiyev/GREYC_project/Tobii2_myvideos/grc73it/recordings/zs7is5l/segments/1/Result_outputs/Preprocessed_divided/'
        #cv2.imwrite(os.path.join(outt_path, 'last_frame.jpeg'), frame_gray)

        # Apply ratio test to get good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:   # 0-1 lower values more conservative
                good_matches.append(m)

        print('Number of good matches: ', len(good_matches))
    except Exception as e:
        print(f"An error occurred, on number of good matches: {e}")
        good_matches = [0]

    return len(good_matches)


def save_video_portion(video_path, reference_image_path, output_folder, reference_index, start_frame, end_frame, start_timestamp, end_timestamp):
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
    output_path = os.path.join(output_folder, f"portion_{reference_index}")
    os.makedirs(output_path, exist_ok=True)

    # Save frames as a new video in the portion folder
    output_video_path = os.path.join(output_path, f"portion_{reference_index}.mp4")

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

    print(f"Frames saved from {start_frame} to {end_frame}")
    print(f"Timestamp from {start_timestamp} ms to {end_timestamp} ms")
    print("Total number of frames: ", len(output_frames))

    # print(frame.shape)
    # cv2.imwrite(os.path.join(output_path, 'last_frame.jpeg'), frame)

    # Save frame information to a JSON file
    frame_info = {
        "ReferenceImageIndex": reference_index,
        "StartFrame": start_frame,
        "EndFrame": end_frame,
        "StartTimestamp": start_timestamp,
        "EndTimestamp": end_timestamp
    }

    with open(os.path.join(output_path, "frame_info.json"), "w") as info_file:
        json.dump(frame_info, info_file, indent=4)

    # Copy the reference image into the current folder
    shutil.copy(reference_image_path, output_path)



def process_video(video_path, reference_folder, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    sift = cv2.SIFT_create()

    # Set up FLANN parameters
    FLANN_INDEX_KDTREE = 0    # before 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5) 
    search_params = dict(checks=100)  # Higher values of checks lead to more accurate results but also result in longer search times
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Load reference images from the specified folder
    valid_image_formats = ['.jpeg', '.jpg', '.png']
    reference_images = [ (cv2.imread(os.path.join(reference_folder, img)), os.path.join(reference_folder, img), img)  
                         for img in sorted(os.listdir(reference_folder)) 
                         if any(img.lower().endswith(format) for format in valid_image_formats)]

    # print("Reference images: ", len(reference_images))
    # print(reference_images)
    # for i in reference_images:
    #     print(i.shape)

    # Initialize variables for tracking portions
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # number of total frames in the video
    current_reference_index = None
    img_name = None
    consecutive_matches_counts = {i: 0 for i in range(len(reference_images))} # For keeping track of consecutive matches between frame and ref images
    consecutive_matches_threshold = 3  # Adjust this threshold as needed
    current_start_frame = {i: [None, None] for i in range(len(reference_images))}  # [Frame number, Frame timestamp] for each reference image
    last_occurrence_frame = {i: [None, None] for i in range(len(reference_images))}  # [Frame number, Frame timestamp] for each reference image
    padding_frames = 30 # Extending purposes: number of frames to subtract from start of the video and add to end of the video
    check_matches_every = 10

    while True:
        # Read the next frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Check matches every 30 frames
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % check_matches_every == 0:
            new_match_found = False

            # Iterate through reference images and find matches
            for i, (reference_image, reference_image_path, img_name) in enumerate(reference_images):
                num_matches = find_matches(reference_image, frame, sift, flann, img_name)

                # Get the current frame number
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                # Print the current frame number and total number of frames
                print(f"Frame: {current_frame}/{total_frames}")

                print(f"First frame occurences dict: {current_start_frame}")
                print(f"Last frame occurences dict: {last_occurrence_frame}")

                # Adjust the threshold as needed
                if num_matches > 50:

                    consecutive_matches_counts[i] += 1
                    print(f"Image {img_name} consecutive matches: {consecutive_matches_counts[i]}")
                    # If consecutive matches exceed the threshold, consider it a valid transition
                    if consecutive_matches_counts[i] >= consecutive_matches_threshold:
                        # Reset the consecutive match count for others except i
                        previous_consec_for_i = consecutive_matches_threshold-2
                        consecutive_matches_counts = {idx: 0 for idx in range(len(reference_images))}
                        consecutive_matches_counts[i] = previous_consec_for_i

                        if current_reference_index is None or i != current_reference_index:
                            # If a new match is found, save the current portion and update tracking variables
                            if current_reference_index is not None:
                                
                                    current_end_frame = last_occurrence_frame[current_reference_index][0]
                                    current_start_timestamp = current_start_frame[current_reference_index][1]
                                    current_end_timestamp = last_occurrence_frame[current_reference_index][1]

                                    #Check if folder and video were already created
                                    directory_path = os.path.join(output_folder, f"portion_{current_reference_index}")
                                    if not os.path.exists(directory_path):
                                        previous_img_name = reference_images[current_reference_index][2]
                                        print(f"Saving the video for reference image {current_reference_index}, image name {previous_img_name}")
                                        previous_reference_image_path = reference_images[current_reference_index][1]
                                        save_video_portion(video_path, previous_reference_image_path, 
                                                           output_folder, current_reference_index,
                                                           current_start_frame[current_reference_index][0],
                                                           current_end_frame,
                                                           current_start_timestamp, current_end_timestamp)

                                    # Reset the consecutive match count for all
                                    consecutive_matches_counts = {i: 0 for i in range(len(reference_images))}


                            current_reference_index = i
                            current_start_frame[current_reference_index][0] = cap.get(cv2.CAP_PROP_POS_FRAMES) #- consecutive_matches_threshold - padding_frames 
                            current_start_frame[current_reference_index][1] = cap.get(cv2.CAP_PROP_POS_MSEC) #- (consecutive_matches_threshold + padding_frames) / cap.get(cv2.CAP_PROP_FPS
                            
                            # Check if out of boundary
                            if current_start_frame[current_reference_index][0] < 0:
                                current_start_frame[current_reference_index][0] = cap.get(cv2.CAP_PROP_POS_FRAMES)
                                current_start_frame[current_reference_index][1] = cap.get(cv2.CAP_PROP_POS_MSEC)

                            last_occurrence_frame[current_reference_index][0] = cap.get(cv2.CAP_PROP_POS_FRAMES) #+ padding_frames 
                            last_occurrence_frame[current_reference_index][1] = cap.get(cv2.CAP_PROP_POS_MSEC) #+ padding_frames / cap.get(cv2.CAP_PROP_FPS
                            new_match_found = True
                            break  # Exit the loop to start with the next reference image

                        # Update the last occurrence frame while detecting num matches > threshold and onsecutive_matches_counts >consecutive matches threshold
                        last_occurrence_frame[i][0] = cap.get(cv2.CAP_PROP_POS_FRAMES) #+ padding_frames 
                        last_occurrence_frame[i][1] = cap.get(cv2.CAP_PROP_POS_MSEC) #+ padding_frames / cap.get(cv2.CAP_PROP_FPS

    # Handle the last occurrence of match
    if current_reference_index is not None:
        print("ENTERING Handling last occurence")
        print(current_start_frame)
        print(last_occurrence_frame)
        current_end_frame = last_occurrence_frame[current_reference_index][0]
        current_start_timestamp = current_start_frame[current_reference_index][1]
        current_end_timestamp = last_occurrence_frame[current_reference_index][1]

        #Check if folder and video were already cerated
        directory_path = os.path.join(output_folder, f"portion_{current_reference_index}")
        if not os.path.exists(directory_path):
            print(f"Saving the video for reference image {current_reference_index}, image name {img_name}")
            save_video_portion(video_path, reference_image_path, output_folder, current_reference_index,
                               current_start_frame[current_reference_index][0], current_end_frame, current_start_timestamp, current_end_timestamp)

    # Release the video capture object
    cap.release()

if __name__ == "__main__":

    # # Parse arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--outputRoot',help= 'Path to result_outputs directory')
    # parser.add_argument('--referenceDir', help = 'path to the reference images directory')
    # args = parser.parse_args()

    # # Paths for files
    # folder_path = os.path.join(args.outputRoot, 'Preprocessed')
    # # Get to a folder with files
    # main_subfolders = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    # subfolder_path = main_subfolders[0]
    # sub_subfolders = [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if os.path.isdir(os.path.join(subfolder_path, f))]
    # last_subfolder_path = sub_subfolders[0]


    # # Get paths of files
    # path_worldCameraVid = os.path.join(last_subfolder_path, 'worldCamera.mp4')

    # # Set output directory
    # output_directory = os.path.join(args.outputRoot, 'Preprocessed_divided')
    # os.makedirs(output_directory, exist_ok=True)

    # # Replace these paths with your actual paths
    # video_path = path_worldCameraVid
    # reference_folder = args.referenceDir
    # output_folder = output_directory 



    # Replace these paths with your actual paths
    video_path = "/home/kaibald231/work_pupil/pupil_mobile/Test_recording/Test_video/exports/000/iMotions_12_02_2024_09_54_24/scene.mp4"
    reference_folder = "/home/kaibald231/work_pupil/pupil_mobile/Test_recording/reference_imgs"
    output_folder = "/home/kaibald231/work_pupil/pupil_mobile/Test_recording/Test_video/exports/000/iMotions_12_02_2024_09_54_24/Results_divided_sep"


    start_time = time.time()
    process_video(video_path, reference_folder, output_folder)
    end_time = time.time()

    print(f"Video cutting took: {end_time - start_time:.2f} seconds")