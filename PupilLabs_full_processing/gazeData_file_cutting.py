# cutting gazeData_world.tsv: gaze data into corresponding portions
import pandas as pd
import os
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

def extract_and_save_gaze_data(gaze_file_path, json_folders):
	# Load gaze data from the tsv file
	gaze_data = pd.read_csv(gaze_file_path, delimiter='\t')

	# Iterate through subfolders in the specified gaze folder
	for subfolder_name in os.listdir(json_folders):
		json_folder = os.path.join(json_folders, subfolder_name)

		if os.path.isdir(json_folder):

			# Iterate through JSON files in the specified folder
			json_files = [file for file in os.listdir(json_folder) if file.endswith('.json')]

			for json_file_name in json_files:
				# Load JSON file to get start_frame and end_frame values
				json_file_path = os.path.join(json_folder, json_file_name)
				with open(json_file_path, 'r') as json_file:
					json_data = json.load(json_file)
					start_frame = json_data['StartFrame']
					end_frame = json_data['EndFrame']

				# Filter gaze data based on frame_idx and frame range
				filtered_gaze_data = gaze_data[(gaze_data['MediaFrameIndex'] >= start_frame) & (gaze_data['MediaFrameIndex'] <= end_frame)]
				first_row_value = filtered_gaze_data['MediaFrameIndex'].iloc[0]
				filtered_gaze_data['MediaFrameIndex'] = filtered_gaze_data['MediaFrameIndex'] - first_row_value

				# Save the filtered gaze data to a new tsv file
				output_file_name = f"gazeData_{os.path.splitext(json_file_name)[0]}.tsv"
				output_file_path = os.path.join(json_folder, output_file_name)
				filtered_gaze_data.to_csv(output_file_path, sep='\t', index=False)


if __name__ == "__main__":

	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--outputRoot',help='Path to result_outputs directory')
	args = parser.parse_args()

	# Paths for files
	folder_path = os.path.join(args.outputRoot, 'Preprocessed')
	json_folders_path = os.path.join(args.outputRoot, 'Preprocessed_divided')

	gaze_file_path = os.path.abspath(os.path.join(args.outputRoot, '..', 'gaze.tlv'))
	print(gaze_file_path)

	# Replace these paths with your actual paths
	json_folders = json_folders_path
	print(json_folders)

	extract_and_save_gaze_data(gaze_file_path, json_folders)

