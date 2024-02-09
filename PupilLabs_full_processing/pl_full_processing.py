"""
1) tobii_preprocessing.py
needs paths to the folder with content (video does not have a sound), and to the reference_image
outputs 3 files, one video(why does it have a sound after processing?)
if output folder path is  provided, ouput folder -> "Provided_directory_path"/Result_outputs/Preprocessed"
else default ouput folder -> "inputDir/Result_outputs/Preprocessed"

2) video_cutting.py
needs path to tobii_preprocessing.py outputs
outputs folders with corresponding video, reference image, and timestamp data file
to ".../Result_outputs/Preprocessed_divided"

3) gazeData_file_cutting.py
needs path to tobii_preprocessing.py and video_cutting.py
outputs cutted gazeData files corresponding to cutted videos 
to ".../Result_outputs/Preprocessed_divided"

4) processData.py
needs path to ".../Result_outputs/Preprocessed_divided"
outputs 3 videos and file for each of the folders
to ".../Result_outputs/Processed"

5) video_concatenation.py
needs 2 videos from each folder of processData.py  ".../Result_outputs/Processed"
outputs concatenated videos
to ".../Result_outputs/Final_result"

6) density_map.py
needs outputs of processData.py   ".../Result_outputs/Processed"
outputs density maps 
to ".../Result_outputs/Final_result"
"""


import concurrent.futures
import subprocess
import argparse
import sys
import os
import time

# INPUT PATHS
# 5) 3 posters
#input_default = '/Users/azamatkaibaldiyev/GREYC_project/Tobii2_myvideos/uffnpti_6/recordings/qvxghya/segments/1'
# 4)3 posters
input_default = '/Users/azamatkaibaldiyev/GREYC_project/Tobii2_myvideos/ggrko5d_5/recordings/4uirm2q/segments/1'
# 3) 3 posters
#input_default = '/Users/azamatkaibaldiyev/GREYC_project/Tobii2_myvideos/57dz5g3_4recordings/wlgk2rn/segments/1'
# 2) 4 posters
#input_default = '/Users/azamatkaibaldiyev/GREYC_project/Tobii2_myvideos/grc73it/recordings/zs7is5l/segments/1'
# 1) one poster
#input_default = '/Users/azamatkaibaldiyev/GREYC_project/Tobii2_myvideos/bxuvruh/recordings/bvxjhgo/segments/1'

#If the reference images folder is inside the folder with scripts
script_folder_path= os.path.dirname(os.path.abspath(__file__))
reference_default  = os.path.join(script_folder_path, 'reference_images')


parser = argparse.ArgumentParser()
parser.add_argument('--inputDir', help='Input directory of tobii2 raw files', default = input_default)
parser.add_argument('--referenceDir', help='Path to folder with the images of paintings', default  = reference_default)
parser.add_argument('--outputRoot', help='Path to where output data is saved to')
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

output_directory = os.path.join(args.outputRoot, 'Result_outputs')
os.makedirs(output_directory, exist_ok=True)


#Create a path for the script to run it
def create_path(script_name):
    return os.path.join(script_folder_path, script_name)


#List of your script files
script_files = [
				#'tobii_preprocessing.py', 
				#'video_cutting.py', 
				#'gazeData_file_cutting.py',
				'runner_mapGaze.py', 
				'mapGaze.py', 
				#'videos_concatenation.py', 
				#'density_map.py'
				]

# Arguments for each script
script_arguments =[
  				  #['--inputDir', args.inputDir, '--outputRoot', output_directory],     			#tobii_preprocessing.py
   				  #['--outputRoot', output_directory, '--referenceDir', args.referenceDir], 		#video_cutting.py
  				  #['--outputRoot', output_directory], 											#gazeData_file_cutting.py
  				  ['--outputRoot', output_directory, '--folder', script_folder_path],			#runner mapgaze
   				  [""],																			#mapGaze.py
  				  #['--outputRoot', output_directory],											#videos_concatenation.py
   				  #['--outputRoot', output_directory]											#density_map.py
				  ]


# #Testing single script
# script_files = ['density_map.py']
# script_arguments = [
#     ['--outputRoot', output_directory, '--referenceDir', args.referenceDir]
# ]

start_time = time.time()
# Run scripts sequentially
for script, args in zip(script_files, script_arguments):

    script_path = create_path(script)
    
    # Run the script and check the return code
    print(f"Launching script {script}.________________________________________________________________________________")
    command = ['python', script_path] + [str(arg) for arg in args]
    result = subprocess.run(command)

    # Check if the script failed (return code other than 0)
    if result.returncode != 0:
        print(f"Script {script} failed. Stopping further execution.")
        break
    else:
        print(f"Script {script} completed successfully._____________________________________________________________________________")

end_time = time.time()
print("Total execution time: {} seconds", end_time-start_time)
# def run_script(args):
#     script, script_args = args
#     subprocess.run(['python', script] + script_args)

# # Run scripts in parallel
# with concurrent.futures.ThreadPoolExecutor() as executor:
#     executor.map(run_script, zip(script_files, script_arguments))

