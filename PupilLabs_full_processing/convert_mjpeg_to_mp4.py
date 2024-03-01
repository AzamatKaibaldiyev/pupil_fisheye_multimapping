import subprocess
import json
import cv2
import numpy as np
import os
import sys


def merge_npy_files(input_folder):
        
    # Get a list of all files in the folder
    world_files_in_folder = os.listdir(input_folder)
    matching_npy_files = [file for file in world_files_in_folder if file.startswith('world') and file.endswith('timestamps.npy')]

    # Load each .npy file
    arrays = [np.load(os.path.join(input_folder, file)) for file in matching_npy_files]

    # Concatenate the arrays along the specified axis (change axis according to your data)
    merged_array = np.concatenate(arrays, axis=0)

    # Save the merged array to a new .npy file
    path_merged_npy_output = os.path.join(path_merged_files, 'merged_world_timestamps.npy')
    np.save(path_merged_npy_output, merged_array)

    return merged_array


#  Prints timestamps of the input video
def get_video_timestamps(path_world_mjpeg):
    # Run ffprobe command to get video information
    ffprobe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', path_world_mjpeg]
    result = subprocess.run(ffprobe_cmd, stdout=subprocess.PIPE)

    # Parse JSON output
    output_json = json.loads(result.stdout)

    # Extract frame timestamps
    timestamps = [float(frame['best_effort_timestamp_time']) for frame in output_json['frames']]

    # Calculate time differences between consecutive frames
    time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]

    # Print frame rates
    print("Number of total frames:", len(timestamps))

    if path_world_mjpeg.split('.')[-1]!='mjpeg':
        #print("Frame Rates (fps):", frame_rates)
        print("Timestamps start: ", timestamps[:7])
        print("Timestamps end: ", timestamps[-7:])

    # Calculate the number of frames in each consecutive second
    fps = 0  # Initialize frames counter
    lower_bound = 0

    counter_list = []
    for timestamp in timestamps:
        if timestamp>=lower_bound and timestamp<lower_bound+1:
            fps+=1
            # print(fps)
            # print(timestamp)
            if timestamp==timestamps[-1]:
                counter_list.append(fps)
                break
        else:    
            counter_list.append(fps)
            fps = 1
            lower_bound+=1
            # print(fps)
            # print(timestamp)

    # Print frames count per second
    print("Frames count per second:", counter_list)

    return timestamps


def concat_convert_videos(video_paths, timestamps, output_folder_path):

    # Get the parameters from the first video
    cap = cv2.VideoCapture(video_paths[0])

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print("Error: Could not open video file")
        exit()
    # Get frame width and height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Determine frame rate dynamically
    frame_rate = 1 / np.mean(np.diff(timestamps))
    print("Frame rate for writing np.mean (choosing this): ", frame_rate)
    frame_rate =  float(f'{frame_rate:.{4}f}')

    # Define output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(output_folder_path , 'converted_world.mp4')
    out = cv2.VideoWriter(out_path , fourcc, frame_rate, (width, height))

    for file_path in video_paths:

        cap = cv2.VideoCapture(file_path)
        # Check if the video file was successfully opened
        if not cap.isOpened():
            print("Error: Could not open video file")
            exit()
        # Get total number of frames
        #frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #print(f"Number of frames for {file_path.split('/')[-1]}: {frame_count}")

        # Read and collect frames until the end of the video
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Write frame to video writer
            for _ in range(1):  # Repeat frame to match specified timestamp
                out.write(frame)

        # Release the VideoCapture object
        cap.release()

    # Release resources
    out.release()



if __name__ == '__main__':

    if len(sys.argv)==2:

            input_folder= sys.argv[1]

            #Check if you have one or several world mjpeg videos
            world_files_in_folder = os.listdir(input_folder)
            matching_mjpeg_files = [file for file in world_files_in_folder if file.startswith('world') and file.endswith('.mjpeg')]
            multiple_videos = False
            if len(matching_mjpeg_files)>1:
                multiple_videos = True

            # Create a folder for merged and converted data
            path_merged_files = os.path.join(input_folder, 'Converted_files')
            os.makedirs(path_merged_files, exist_ok=True)
            if multiple_videos:
                world_timestamps = merge_npy_files(input_folder)
            else:
                data = np.load(os.path.join(input_folder , 'world_timestamps.npy'))
                world_timestamps = data - data[0]

            # Get all mjpeg video file paths
            video_paths  = [os.path.join(input_folder, file) for file in world_files_in_folder if file.startswith('world') and file.endswith('.mjpeg')]

            print("Converting the MJPEG into MP4 format video...")
            concat_convert_videos(video_paths, world_timestamps, path_merged_files)
            print("Converted")
            
    else:
        print("Problem with arguments")





