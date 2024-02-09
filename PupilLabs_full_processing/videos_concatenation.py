import cv2
import numpy as np
import argparse
import os


def concatenate_videos(path_paintingVideo, path_glassesVideo, path_output_video):

    # Load the two video files
    video1 = cv2.VideoCapture(path_paintingVideo)  # Replace with the path to your first video
    video2 = cv2.VideoCapture(path_glassesVideo)  # Replace with the path to your second video


    # Check if the videos opened successfully
    if not video1.isOpened() or not video2.isOpened():
        print("Error: Could not open videos.")
        exit()

    # Get video properties
    fps1 = video1.get(cv2.CAP_PROP_FPS)
    fps2 = video2.get(cv2.CAP_PROP_FPS)

    # Get the original widths and heights of both videos
    width1, height1 = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2, height2 = int(video2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Determine the common height for resizing
    target_height = min(height1, height2)

    # Determine the scaling factors for resizing
    scale_factor1 = target_height / height1
    scale_factor2 = target_height / height2

    # Create a VideoWriter object for the output
    output_width = int(width1 * scale_factor1)+ int(width2 * scale_factor2)
    output_height = target_height
    output_fps = min(fps1, fps2)  # Use the minimum fps of the two videos
    output_video = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'mp4v'), output_fps, (output_width, output_height))

    while True:
        # Read frames from both videos
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()

        # Break the loop if either video reaches the end
        if not ret1 or not ret2:
            break

        # Resize frames using scaling factors
        resized_frame1 = cv2.resize(frame1, (int(width1 * scale_factor1), target_height))
        resized_frame2 = cv2.resize(frame2, (int(width2 * scale_factor2), target_height))

        # Concatenate the frames horizontally
        combined_frame = np.concatenate((resized_frame1, resized_frame2), axis=1)

        # Write the combined frame to the output video
        output_video.write(combined_frame)

    # Release resources
    video1.release()
    video2.release()
    output_video.release()

    cv2.destroyAllWindows()



if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputRoot', help='Path to result_outputs directory')
    args = parser.parse_args()

    # Paths for files
    folder_path = os.path.join(args.outputRoot , 'Processed_mapped')

    #Iterate through folders
    for folder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, folder_name)

        if os.path.isdir(subfolder_path):

            path_paintingVideo = os.path.join(subfolder_path , 'ref_gaze.mp4')
            path_glassesVideo = os.path.join(subfolder_path , 'world_gaze.mp4')

            path_output = os.path.join(args.outputRoot, 'Final_results', folder_name)
            os.makedirs(path_output, exist_ok=True)
            path_output_video = os.path.join(path_output, f'Final_video_{folder_name}.mp4')

            print(f"Concatenating videos from folder {folder_name}")

            concatenate_videos(path_paintingVideo, path_glassesVideo, path_output_video)



