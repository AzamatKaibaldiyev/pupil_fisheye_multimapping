import concurrent.futures
import subprocess
import argparse
import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")

# INPUT PATHS
# 1) 3 posters
input_default = '/home/kaibald231/work_pupil/pupil_mobile/Test_recording/Test_video/exports/000/iMotions_12_02_2024_09_54_24/'

#If the reference images folder is inside the folder with scripts
# script_folder_path= os.path.dirname(os.path.abspath(__file__))
# reference_default  = os.path.join(script_folder_path, 'reference_images')
reference_default = '/home/kaibald231/work_pupil/pupil_mobile/Test_recording/reference_imgs/'


parser = argparse.ArgumentParser()
parser.add_argument('--inputDir', help='Input directory of pupil recordings and data', default = input_default)
parser.add_argument('--referenceDir', help='Path to folder with the reference images', default  = reference_default)
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
script_folder_path = os.getcwd()
def create_path(script_name):
    return os.path.join(script_folder_path, script_name)


#List of your script files
script_files = [
				# future batch exporter
				 'runner_video_cutting.py',
				#'video_cutting.py', 
				#'gazeData_file_cutting.py',
				#'runner_mapGaze.py', 
				#'mapGaze.py', 
				#'videos_concatenation.py', 
				#'density_map.py'
				]

# Arguments for each script
script_arguments =[
  				  #['--inputDir', args.inputDir, '--outputRoot', output_directory],     									#future batch exporter
      			  ['--inputDir', args.inputDir, '--outputRoot', output_directory, '--referenceDir', args.referenceDir, '--folder', script_folder_path], 	#runner_video_cutting.py
   				  #['--outputRoot', output_directory, '--referenceDir', args.referenceDir], 								#video_cutting.py
  				  #['--outputRoot', output_directory], 																		#gazeData_file_cutting.py
  				  #['--outputRoot', output_directory, '--folder', script_folder_path],										#runner mapgaze
   				  #[""],																									#mapGaze.py
  				  #['--outputRoot', output_directory],																		#videos_concatenation.py
   				  #['--outputRoot', output_directory]																		#density_map.py
				  ]


total_start_time = time.time()
# Run scripts sequentially
for script, args in zip(script_files, script_arguments):

    script_path = create_path(script)
    
    # Run the script and check the return code
    print(f"Launching script {script}.__________________________________________________________________________________________")
    command = ['python', script_path] + [str(arg) for arg in args]
    result = subprocess.run(command)

    # Check if the script failed (return code other than 0)
    if result.returncode != 0:
        print(f"Script {script} failed. Stopping further execution.")
        break
    else:
        print(f"Script {script} completed successfully._____________________________________________________________________________")

total_end_time = time.time()
print("Total execution time: {} seconds", total_end_time-total_start_time)

