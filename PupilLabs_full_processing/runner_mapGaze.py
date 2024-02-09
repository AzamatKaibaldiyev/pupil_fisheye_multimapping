import subprocess
import os
import pandas as pd
import argparse

if __name__ == '__main__':


    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputRoot',help='Path to result_outputs directory')
    parser.add_argument('--folder')
    args = parser.parse_args()


    # Paths for files
    folder_path = os.path.join(args.outputRoot, 'Preprocessed_divided')

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
            output_directory = os.path.join(args.outputRoot, 'Processed_mapped', folder_name)
            os.makedirs(output_directory, exist_ok=True)


            # Input error checking
            badInputs = []
            for arg in [path_gazeData, path_worldCameraVid, path_referenceImage]:
                if not os.path.exists(arg):
                    badInputs.append(arg)
            if len(badInputs) > 0:
                #[print('{} does not exist! Check your input file path'.format(x)) for x in badInputs]
                raise ValueError('{} does not exist! Check your input file path'.format(badInputs))
                #sys.exit()


            ## process the recording
            print(f'processing the folder {folder_name}')
            print('Output saved in: {}'.format(output_directory))

            script_folder = args.folder

            script_path = os.path.join(script_folder, 'mapGaze.py')

            command = ['python', script_path] + [path_gazeData, path_worldCameraVid, path_referenceImage, output_directory]
            result = subprocess.run(command)

