import multiprocessing
import subprocess
import os
import pandas as pd
import argparse
import time
import sys
import warnings
warnings.filterwarnings("ignore")


def run_video_cutting(folder_name, script_path, video_path, reference_image_path, output_folder):
    """Function to execute the script with the provided arguments."""
    start_time = time.time()
    command = ['python', script_path] + [video_path, reference_image_path, output_folder]
    subprocess.run(command)
    end_time = time.time()
    print(f">>>>>>>>>>>>>>>>>>Video cutting for {folder_name} took: {end_time - start_time:.2f} seconds<<<<<<<<<<<<<<<<<<<<")

def run_mapGaze(folder_name, script_path, path_gazeData, path_worldCameraVid, path_referenceImage, output_directory):
    """Function to execute the script with the provided arguments."""
    start_time = time.time()
    command = ['python', script_path] + [path_gazeData, path_worldCameraVid, path_referenceImage, output_directory]
    subprocess.run(command)
    end_time = time.time()
    print(f">>>>>>>>>>>>>>>>>>Gaze mapping for {folder_name} took: {end_time - start_time:.2f} seconds<<<<<<<<<<<<<<<<<<<<")

def run_concatenation(folder_name, script_path, path_paintingVideo, path_glassesVideo, path_output_video):
    """Function to execute the script with the provided arguments."""
    start_time = time.time()
    command = ['python', script_path] + [path_paintingVideo, path_glassesVideo, path_output_video]
    subprocess.run(command)
    end_time = time.time()
    print(f">>>>>>>>>>>>>>>>>>Concatenation for {folder_name} took: {end_time - start_time:.2f} seconds<<<<<<<<<<<<<<<<<<<<")

def run_heatmap(folder_name, script_path, tsv_file_path, output_path_density_map, path_referenceImage, output_path_expansion_map):
    """Function to execute the script with the provided arguments."""
    start_time = time.time()
    command = ['python', script_path] + [tsv_file_path, output_path_density_map, path_referenceImage, output_path_expansion_map]
    subprocess.run(command)
    end_time = time.time()
    print(f">>>>>>>>>>>>>>>>>>Heatmap for {folder_name} took: {end_time - start_time:.2f} seconds<<<<<<<<<<<<<<<<<<<<")



if __name__ == '__main__':

    run_video_cut = False
    run_data_cut = False
    run_gaze_map = False
    run_video_concatenation = False
    run_heatmap_expansion = True

    # DEFAULT PATHS
    # 1) 3 posters
    input_default = '/home/kaibald231/work_pupil/pupil_mobile/Test_recording/Test_video/exports/000/iMotions_12_02_2024_09_54_24/'
    reference_default = '/home/kaibald231/work_pupil/pupil_mobile/Test_recording/reference_imgs/'
    script_folder_path = os.getcwd()

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputDir', help = 'Input directory of pupil recordings and data', default = input_default)
    parser.add_argument('--referenceDir', help = 'Path to folder with the reference images', default  = reference_default)
    parser.add_argument('--outputRoot', help = 'Path to where output data is saved to')
    parser.add_argument('--scripts_folder',help = 'Path to scripts folder',  default = script_folder_path)
    args = parser.parse_args()

    # Check if input directory is provided
    if args.inputDir is not None:
        # Check if the directory is valid
        if not os.path.isdir(args.inputDir):
            print('Invalid input dir: {}'.format(args.inputDir))
            sys.exit()
        else:
            print(f'User provided input : {args.inputDir}')
    else:
        print('Please provide input directory path')
        sys.exit()

    # Check if reference images directory is provided
    if args.referenceDir is not None:
        # Check if the directory is valid
        if not os.path.isdir(args.referenceDir):
            print('Invalid reference dir: {}'.format(args.referenceDir))
            sys.exit()
        else:
            print(f'User provided input : {args.referenceDir}')
    else:
        print('Please provide reference images directory path')
        sys.exit()

    # Check if the user provided the output directory argument
    if args.outputRoot is not None:
        print(f'User provided path for output directory: {args.outputRoot}')
    else:
        print(f'User did not provide path for output directory. Created folder in user provided input directory instead: {args.inputDir}')
        args.outputRoot = args.inputDir
    output_results_directory = os.path.join(args.outputRoot, 'Result_outputs')
    os.makedirs(output_results_directory, exist_ok=True)


    total_start_time = time.time()
    ############################################################
    ### VIDEO CUTTING
    # Paths for files
    script_path = os.path.join(args.scripts_folder, 'video_cutting.py')
    video_path = os.path.join(args.inputDir, 'scene.mp4')
    script_args_video_cutting = []

    # Set output directory
    output_folder = os.path.join(output_results_directory, 'Preprocessed_divided')
    os.makedirs(output_folder, exist_ok=True)    

    # Load reference images from the specified folder
    for ref_img in os.listdir(args.referenceDir):
        reference_image_path = os.path.join(args.referenceDir, ref_img)
        img_name = ref_img.split('.')[0]
        script_args_video_cutting.append((img_name, (img_name, script_path, video_path, reference_image_path, output_folder)))

    if run_video_cut:
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


    ############################################################
    ### GAZE DATA FILE CUTTING
    # Run the script and check the return code
    script_gazedata = 'gazeData_file_cutting.py'
    script_gazedata_path = os.path.join(args.scripts_folder, script_gazedata)
    command = ['python', script_gazedata] + ['--outputRoot', output_results_directory]
    
    if run_data_cut:
        print(f"Launching script {script_gazedata}.__________________________________________________________________________________________")
        result = subprocess.run(command)

        # Check if the script failed (return code other than 0)
        if result.returncode != 0:
            print(f"Script {script_gazedata} failed. Stopping further execution.")
            sys.exit();
        else:
            print(f"Script {script_gazedata} completed successfully._____________________________________________________________________________")


    ############################################################
    ### GAZE MAPPING
    # Paths for files
    folder_path = os.path.join(output_results_directory, 'Preprocessed_divided')
    script_path = os.path.join(args.scripts_folder, 'mapGaze.py')
    script_args_mapgaze = []

    for folder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, folder_name) 

        if os.path.isdir(subfolder_path):
            # List all files in the folder
            all_files = os.listdir(subfolder_path)

            # Get paths of files
            gazeData_file = [file for file in all_files if file.endswith('.tsv')]
            path_gazeData = os.path.join(subfolder_path, gazeData_file[0])
            worldCameraVid_file = [file for file in all_files if file.endswith('.mp4')]
            path_worldCameraVid = os.path.join(subfolder_path, worldCameraVid_file[0])
            referenceImage_file = [file for file in all_files if (file.endswith('.jpeg') or file.endswith('.jpg') or file.endswith('.png'))]
            path_referenceImage = os.path.join(subfolder_path, referenceImage_file[0])

            # Set output directory
            output_directory = os.path.join(output_results_directory, 'Processed_mapped', folder_name)
            os.makedirs(output_directory, exist_ok=True)

            #Packing all the arguments
            script_args_mapgaze.append((folder_name, (folder_name, script_path, path_gazeData, path_worldCameraVid, path_referenceImage,output_directory),output_directory))

            # Input error checking
            badInputs = []
            for arg in [path_gazeData, path_worldCameraVid, path_referenceImage]:
                if not os.path.exists(arg):
                    badInputs.append(arg)
            if len(badInputs) > 0:
                raise ValueError('{} does not exist! Check your input file path'.format(badInputs))
                #sys.exit()

    if run_gaze_map:
        # Create and start processes for mapGaze.py for each argument
        processes = []
        for folder_name, folder_args,output_directory in script_args_mapgaze:
            print(f'processing the folder {folder_name}')
            print('Output saved in: {}'.format(output_directory))        
            single_process = multiprocessing.Process(target=run_mapGaze, args=folder_args)
            processes.append(single_process)
            single_process.start()
            
        # Wait for all processes to complete
        for process in processes:
            process.join()


    ############################################################
    ### CONCATENATION AND HEATMAP
    # Paths for files
    folder_path = os.path.join(output_results_directory , 'Processed_mapped')
    script_args_concat = []
    script_args_heatmap = []

    #Iterate through folders
    for folder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, folder_name)

        if os.path.isdir(subfolder_path):

            # Preparing arguments for videos_concatentation.py
            script_path = os.path.join(args.scripts_folder, 'videos_concatenation.py')
            path_paintingVideo = os.path.join(subfolder_path , 'ref_gaze.mp4')
            path_glassesVideo = os.path.join(subfolder_path , 'world_gaze.mp4')
            path_output = os.path.join(output_results_directory, 'Final_results', folder_name)
            os.makedirs(path_output, exist_ok=True)
            path_output_video = os.path.join(path_output, f'Final_video_{folder_name}.mp4')
            script_args_concat .append((folder_name, (folder_name, script_path, path_paintingVideo, path_glassesVideo, path_output_video)))

            # Preparing arguments for density_map.py
            script_path = os.path.join(args.scripts_folder, 'density_map.py')
            tsv_file_path  = os.path.join(subfolder_path, 'gazeData_mapped.tsv')
            all_files = os.listdir(subfolder_path)
            referenceImage_file = [os.path.join(subfolder_path, file) for file in all_files if (file.endswith('.jpeg') or file.endswith('.jpg') or file.endswith('.png'))]
            path_referenceImage = referenceImage_file[0]
            path_output = os.path.join(output_results_directory, 'Final_results', folder_name)
            output_path_density_map = os.path.join(path_output,  f'density_map_{folder_name}.jpeg')
            output_path_expansion_map = os.path.join(path_output,  f'expansion_map_{folder_name}.jpeg')
            script_args_heatmap.append((folder_name, (folder_name, script_path, tsv_file_path, output_path_density_map, path_referenceImage, output_path_expansion_map)))

    if run_video_concatenation:
        # Create and start processes for videos_concatenation.py for each argument
        processes = []
        for folder_name, folder_args in script_args_concat :
            print(f'videos_concatenation.py processing the folder {folder_name}')       
            single_process = multiprocessing.Process(target=run_concatenation, args=folder_args)
            processes.append(single_process)
            single_process.start()
        
        # Wait for all processes to complete
        for process in processes:
            process.join()
    
    if run_heatmap_expansion:
        # Create and start processes for density_map.py for each argument
        processes = []
        for folder_name, folder_args in script_args_heatmap:
            print(f'density_map.py processing the folder {folder_name}')       
            single_process = multiprocessing.Process(target=run_heatmap, args=folder_args)
            processes.append(single_process)
            single_process.start()

        # Wait for all processes to complete
        for process in processes:
            process.join()

    total_end_time = time.time()
    print("Total execution time: {} seconds", total_end_time-total_start_time)

