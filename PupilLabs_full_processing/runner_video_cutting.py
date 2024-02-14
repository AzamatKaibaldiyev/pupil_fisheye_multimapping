import multiprocessing
import subprocess
import os
import pandas as pd
import argparse
import time


def run_video_cutting(folder_name, script_path, video_path, reference_image_path, output_folder):
    """Function to execute the script with the provided arguments."""
    start_time = time.time()
    command = ['python', script_path] + [video_path, reference_image_path, output_folder]
    subprocess.run(command)
    end_time = time.time()
    print(f">>>>>>>>>>>>>>>>>>Video cutting for {folder_name} took: {end_time - start_time:.2f} seconds<<<<<<<<<<<<<<<<<<<<")



if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputDir', help='Path to the exported iMotion recording directory')
    parser.add_argument('--outputRoot',help= 'Path to result_outputs directory')
    parser.add_argument('--referenceDir', help = 'Path to the reference images directory')
    parser.add_argument('--folder')
    args = parser.parse_args()

    # Paths for files
    script_path = os.path.join(args.folder, 'video_cutting.py')
    print(args.inputDir)
    video_path = os.path.join(args.inputDir, 'scene.mp4')
    script_args_video_cutting = []

    # Set output directory
    output_folder = os.path.join(args.outputRoot, 'Preprocessed_divided')
    os.makedirs(output_folder, exist_ok=True)    

    # Load reference images from the specified folder
    for ref_img in os.listdir(args.referenceDir):
        reference_image_path = os.path.join(args.referenceDir, ref_img)
        img_name = ref_img.split('.')[0]
        script_args_video_cutting.append((img_name, (img_name, script_path, video_path, reference_image_path, output_folder)))

    # Create and start processes for video_cutting.py for each argument
    processes = []
    for folder_name, folder_args in script_args_video_cutting :
        print(f'video_cutting.py processing the ref image {folder_name}')       
        single_process = multiprocessing.Process(target=run_video_cutting, args=folder_args)
        processes.append(single_process)
        single_process.start()
    
    # Wait for all processes to complete
    for process in processes:
        process.join()
    