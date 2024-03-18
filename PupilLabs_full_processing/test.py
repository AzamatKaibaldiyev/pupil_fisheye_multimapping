# import sys
# import os

# # Get the current script's directory
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # Add the parent directory to the sys.path
# parent_dir = os.path.join(current_dir, '..')
# sys.path.append(parent_dir)
# print(parent_dir)

# # Add the subfolder to the sys.path
# subfolder_dir = os.path.join(parent_dir, 'pupil',  'pupil_src', 'shared_modules')
# sys.path.append(subfolder_dir)

# print(subfolder_dir)

# import file_methods

# # file_methods.PLData_Writer 
# # file_methods.load_pldata_file() 
# inputDir = '/Users/azamatkaibaldiyev/GREYC_project/Pupil_labs/pupil/recordings/2024_01_16/001'

# #pupil_data_path = os.path.join(inputDir, 'pupil_data')

# data, data_ts, topics = file_methods.load_pldata_file(inputDir, 'gaze') 

# print(len(data))
# print(len(data_ts))
# print(len(topics))

# print(data[0])


# data, data_ts, topics = file_methods.load_pldata_file(inputDir, 'pupil') 

# print(len(data))
# print(len(data_ts))
# print(len(topics))

# print(data[0])
# print("#")
# print(data[1])
# print("#")
# print(data[2])
# print("#")
# print(data[3])

########################################################################################################################################

# import os
# import collections

# import msgpack
# import numpy as np


# PLData = collections.namedtuple("PLData", ["data", "timestamps", "topics"])


# def serialized_dict_from_msgpack_bytes(data):
#     return msgpack.unpackb(
#         data, raw=False, use_list=False, ext_hook=msgpack_unpacking_ext_hook,
#     )


# def msgpack_unpacking_ext_hook(self, code, data):
#     SERIALIZED_DICT_MSGPACK_EXT_CODE = 13
#     if code == SERIALIZED_DICT_MSGPACK_EXT_CODE:
#         return serialized_dict_from_msgpack_bytes(data)
#     return msgpack.ExtType(code, data)


# def load_pldata_file(directory, topic):
#     ts_file = os.path.join(directory, topic + "_timestamps.npy")
#     msgpack_file = os.path.join(directory, topic + ".pldata")
#     try:
#         data = []
#         topics = []
#         data_ts = np.load(ts_file)
#         with open(msgpack_file, "rb") as fh:
#             for topic, payload in msgpack.Unpacker(fh, raw=False, use_list=False):
#                 datum = serialized_dict_from_msgpack_bytes(payload)
#                 data.append(datum)
#                 topics.append(topic)
#     except FileNotFoundError:
#         data = []
#         data_ts = []
#         topics = []

#     return PLData(data, data_ts, topics)


# if __name__ == "__main__":

#     # edit `path` s.t. it points to your recording
#     #path = "/Users/me/recordings/2020_06_19/001"
#     path = '/Users/azamatkaibaldiyev/GREYC_project/Pupil_labs/pupil/recordings/2024_01_16/001'

#     # Read "gaze.pldata" and "gaze_timestamps.npy" data
#     get_data = load_pldata_file(path, "gaze")

#     data_gaze = get_data.data
#     data_ts = get_data.timestamps
#     topics = get_data.topics

#     import pprint

#     p = pprint.PrettyPrinter(indent=4)

#     print(">>> FIRST GAZE TIMESTAMP:")
#     p.pprint(data_ts[0])
#     print()

#     print(">>> FIRST GAZE DATUM:")
#     p.pprint(data_gaze[0])
#     print()

# import time

# import pandas as pd

# def read_tlv_file(file_path):
#     df = pd.read_csv(file_path, header = 0, delimiter='\t')
#     #print(df.head())
#     print(df['Gaze2dX'][:5])

# # Example usage
# start_time = time.time()
# read_tlv_file("/home/kaibald231/work_pupil/pupil_mobile/Test_recording/Test_video/exports/000/iMotions_12_02_2024_09_54_24/gaze.tlv")
# end_time = time.time()

# print(f"Video cutting took: {end_time - start_time:.2f} seconds")


# import os

# # Get the current working directory
# current_dir = os.getcwd()
# file_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'file.txt'))


# ### ACTIVATE PUPIL_VENV
# # source ~/work_pupil/pupil/pupil_env/bin/activate
# source ~/work_pupil/pupil/pupil_env_2/bin/activate


###########################################################################################
# from moviepy.editor import VideoFileClip, AudioFileClip
# from pydub import AudioSegment

# def get_audio_length(audio_file_path):
#     # Load the audio file
#     audio = AudioSegment.from_file(audio_file_path)
    
#     # Get the length of the audio in milliseconds
#     length_ms = len(audio)
    
#     # Convert milliseconds to seconds
#     length_sec = length_ms / 1000
    
#     return length_sec



# # Add audio portions to video portions
# def merge_audio_with_video(video_path, audio_path, output_path, audio_adjusted):
#     # Load video and audio clips
#     video_clip = VideoFileClip(video_path)
#     audio_clip = AudioFileClip(audio_path)
#     audio_adjusted_clip = AudioFileClip(audio_adjusted)

#     print('pupil audio duration method ', audio_clip.duration)
#     print('adjusted audio duration method ', audio_adjusted_clip.duration)
#     print('pupil video duration method ', video_clip.duration)
#     # Set audio of the video clip to the loaded audio clip
#     video_clip = video_clip.set_audio(audio_clip)

#     # Write the new video file with the merged audio
#     video_clip.write_videofile(output_path)


# import subprocess

# def get_audio_duration(audio_file):
#     command = [
#         "ffprobe",
#         "-i", audio_file,
#         "-show_entries", "format=duration",
#         "-v", "quiet",
#         "-of", "csv=p=0"
#     ]
#     result = subprocess.run(command, capture_output=True, text=True)
#     duration = float(result.stdout.strip())
#     return duration




# video_path = '/home/kaibald231/work_pupil/pupil_mobile/s20_museum_test_2/20240308144312223/Converted_files/converted_world.mp4'
# audio_path = '/home/kaibald231/work_pupil/pupil_mobile/s20_museum_test_2/20240308144312223/audio_00010000.mp4'
# audio_adjusted = '/home/kaibald231/work_pupil/pupil_mobile/s20_museum_test_2/20240308144312223/Converted_files/synced_audio_adjusted.mp3'
# output_path = '/home/kaibald231/work_pupil/pupil_mobile/s20_museum_test_2/20240308144312223/Converted_files/test_video.mp4'

# print('#######################')
# duration = get_audio_duration(audio_path)
# print("Duration of the pupil audio file:", duration, "seconds")
# duration = get_audio_duration(audio_adjusted)
# print("Duration of the adjusted audio file:", duration, "seconds")

# print('#######################')
# print('pupil audio lenght funtcion', get_audio_length(audio_path))
# print('adjusted micro audio lenght funtcion', get_audio_length(audio_adjusted))
# print('#######################')
# merge_audio_with_video(video_path, audio_path, output_path, audio_adjusted)


############################################33


# from datetime import datetime, timedelta

# def milliseconds_to_hh_mm_ss_ms(milliseconds):
#     # Create a timedelta object with milliseconds
#     delta = timedelta(milliseconds=milliseconds)

#     # Use datetime.utcfromtimestamp to get a datetime object
#     time_obj = datetime.utcfromtimestamp(0) + delta

#     # Format the datetime object to hh:mm:ss:ms
#     formatted_time = time_obj.strftime('%H:%M:%S.%f')[:-3]

#     return formatted_time

# # Example usage
# milliseconds = 2330.257700*1000  # Example milliseconds
# formatted_time = milliseconds_to_hh_mm_ss_ms(milliseconds)
# print("Formatted time:", formatted_time)

#############################

# import subprocess

# def get_audio_duration(input_audio_path):
#     # Run ffprobe command to get audio duration
#     command = ["ffprobe", "-i", input_audio_path, "-show_entries", "format=duration", "-v", "quiet", "-of", "csv=p=0"]
#     result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
#     # Check if ffprobe command was successful
#     if result.returncode == 0:
#         # Extract audio duration from ffprobe output
#         audio_duration = float(result.stdout.decode().strip())
#         return audio_duration
#     else:
#         # Handle error if ffprobe command failed
#         print("Error:", result.stderr.decode().strip())
#         return None

# def get_audio_duration(audio_file):
#     command = [
#         "ffprobe",
#         "-i", audio_file,
#         "-show_entries", "format=duration",
#         "-v", "quiet",
#         "-of", "csv=p=0"
#     ]
#     result = subprocess.run(command, capture_output=True, text=True)
#     duration = float(result.stdout.strip())
#     return duration


# def adjust_audio_offset(input_audio_path, output_audio_path, offset_beginning, offset_end):
#     # Get audio duration using ffprobe
#     audio_duration = get_audio_duration(input_audio_path)
    
#     if audio_duration is None:
#         return  # Exit function if unable to get audio duration
    
#     # Calculate start and end positions for trimming
#     start_time = offset_beginning
#     end_time = audio_duration - offset_end
    
#     # Adjust end_time if it's negative and add silent duration at the end
#     if offset_end < 0:
#         # Calculate the required silent duration
#         silence_duration = abs(offset_end)
        
#         # Create a silent audio segment
#         command = ["ffmpeg", "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100", "-t", str(silence_duration), "-acodec", "pcm_s16le", "silent.wav"]
#         subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
#         # Concatenate the silent audio segment with the original audio
#         command = ["ffmpeg", "-i", "concat:" + input_audio_path + "|silent.wav", "-c", "copy", "concatenated_audio.wav"]
#         subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
#         # Update the input audio path to the concatenated audio file
#         input_audio_path = "concatenated_audio.wav"
        
#         # Update end_time to the adjusted duration
#         end_time = audio_duration
    
#     # Run ffmpeg command to trim audio
#     command = ["ffmpeg", "-i", input_audio_path, "-ss", str(start_time), "-to", str(end_time), "-c", "copy", output_audio_path]
#     result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
#     # Check if ffmpeg command was successful
#     if result.returncode == 0:
#         print("Audio trimmed successfully")
#     else:
#         print("Error:", result.stderr.decode().strip())






# # Example usage:
# input_audio_path='/home/kaibald231/work_pupil/pupil_mobile/s20_museum_test_2/20240308144312223/microphone_audio/200115_001.WAV'
# output_audio_path='/home/kaibald231/work_pupil/pupil_mobile/s20_museum_test_2/20240308144312223/microphone_audio/adjusted_micro_test.mp3'
# offset_beginning=2066.9025
# offset_end=-17.000200000000405

# print('Starting adjusting')
# adjust_audio_offset(input_audio_path, output_audio_path, offset_beginning, offset_end)  # Trim 2 seconds from the beginning and add 2 seconds of silence at the end

# def get_audio_duration(audio_file):
#     command = [
#         "ffprobe",
#         "-i", audio_file,
#         "-show_entries", "format=duration",
#         "-v", "quiet",
#         "-of", "csv=p=0"
#     ]
#     result = subprocess.run(command, capture_output=True, text=True)
#     duration = float(result.stdout.strip())
#     return duration


# print(get_audio_duration(output_audio_path))
# print(get_audio_duration('/home/kaibald231/work_pupil/pupil_mobile/s20_museum_test_2/20240308144312223/audio_00010000.mp4'))


##########################################################################

# import subprocess




# def get_audio_duration(audio_file):
#     command = [
#         "ffprobe",
#         "-i", audio_file,
#         "-show_entries", "format=duration",
#         "-v", "quiet",
#         "-of", "csv=p=0"
#     ]
#     result = subprocess.run(command, capture_output=True, text=True)
#     duration = float(result.stdout.strip())
#     return duration

# pupil_video_path = '/home/kaibald231/work_pupil/pupil_mobile/s20_museum_test_2/20240308144312223/Converted_files/converted_world.mp4'
# sync_audio_path = '/home/kaibald231/work_pupil/pupil_mobile/s20_museum_test_2/20240308144312223/Converted_files/synced_audio_adjusted.mp3'
# pupil_duration = get_audio_duration(pupil_video_path)
# sync_duration = get_audio_duration(sync_audio_path)
# print('pupil audio ', pupil_duration)
# print('sync audio ', sync_duration)




# def trim_audio(input_audio_path, output_audio_path, offset_beginning):
#     # Construct the ffmpeg command to trim the audio
#     ffmpeg_command = [
#         "ffmpeg",
#         "-y",  # Overwrite output files without asking
#         "-i", input_audio_path,  # Input audio file
#         "-ss", str(offset_beginning),  # Start trimming from the specified offset
#         "-c", "copy",  # Copy audio codec without re-encoding
#         output_audio_path  # Output audio file
#     ]

#     # Execute the ffmpeg command
#     try:
#         subprocess.run(ffmpeg_command, check=True)
#         print("Audio trimmed successfully.")
#     except subprocess.CalledProcessError as e:
#         print("Error trimming audio:", e)

# # Example usage:
 
# input_audio_path = sync_audio_path
# output_audio_path = '/home/kaibald231/work_pupil/pupil_mobile/s20_museum_test_2/20240308144312223/Converted_files/trimmed_synced_audio_adjusted.mp3'

# offset_beginning = sync_duration - pupil_duration
# print(offset_beginning)
# if offset_beginning>0:
#     print('Start trimming')
#     trim_audio(input_audio_path, output_audio_path, offset_beginning)


# sync_duration = get_audio_duration(output_audio_path)
# print('sync audio ', sync_duration)
#################################################################
# import json
# text = "This is a sample text. It has more than 10 words, so it needs to be split. This part has more, than 10 words too, so we split it again. This is a shorter part."
# #text ='This part has more, than 10 words too, so we split it again.'
# subtitles_json = '/home/kaibald231/work_pupil/pupil_mobile/s20_museum_test_2/20240308144312223/Converted_files/Preprocessed_divided/portion_BRUEGHEL LE JEUNE Pierre, Le paiement de la dîme, Inv 22 /transcriptions.json'
# with open(subtitles_json, 'r', encoding='utf-8') as file:
#     subtitles_data = json.load(file)
    
# import re
# import math
# full_text = text
# full_text = subtitles_data["text"]
# full_text_splitted = [sentence+'.' for sentence in re.split(r'(?<!\.)\.(?!\.)', full_text)]
# if full_text_splitted[-1]=='.':
#     full_text_splitted = full_text_splitted[:-1]
# sentence_chunks = []
# next_idx = -1
# for sentence in full_text_splitted:
#     #print(sentence)
#     if len(sentence.split())>10:
#         splitted_sentence = [sentence_part+',' if i+1!=len(sentence.split(',')) else sentence_part for i, sentence_part in enumerate(sentence.split(','))]
#         splits_len = len(splitted_sentence)
#         #print(splitted_sentence)
#         if splits_len!=1:
#             for idx in range(len(splitted_sentence)):
#                 if idx>next_idx:
#                     if idx+1<splits_len:
#                         added_1 = splitted_sentence[idx] + splitted_sentence[idx+1]
#                         #print('added_1 ', added_1)
#                         if len(added_1.split())<10:
#                             if idx+2<splits_len:
#                                 added_2 = added_1 + splitted_sentence[idx+2]
#                                 #print('added_2 ', added_2)
#                                 next_idx = idx+1
#                                 if len(added_2.split())<10 and idx+2<splits_len:
#                                     sentence_chunks.append(added_2) 
#                                     next_idx=idx+2
#                                 else:
#                                     sentence_chunks.append(added_1) 
#                             else:
#                                 sentence_chunks.append(added_1) 
#                         else:
#                             sentence_chunks.append(splitted_sentence[idx])   
#                     else:
#                         sentence_chunks.append(splitted_sentence[idx]) 
#         else:
#             words = sentence.split()
#             sentence_length = 7
#             rest = len(words)%sentence_length
#             split_parts = math.floor(len(words)/sentence_length)
#             if rest<4:
#                 for i in range(split_parts-1):
#                     sentence_chunks.append(' '.join(words[i*sentence_length:(i+1)*sentence_length]))
#                 sentence_chunks.append(' '.join(words[(i+1)*sentence_length:]))
#             else:
#                 for i in range(split_parts):
#                     sentence_chunks.append(' '.join(words[i*sentence_length:(i+1)*sentence_length]))
#                 sentence_chunks.append(' '.join(words[(i+1)*sentence_length:]))

#     else:
#         sentence_chunks.append(sentence)

# print('----------------------------------------')
# #print(sentence_chunks)

# # Example usage:
# #text = "This is a sample text. It has more than 10 words, so it needs to be split. This part has more, than 10 words too, so we split it again. This is a shorter part."
# #result_text = split_text(text)
# #print(result_text)
# print('----------------------------------------')
# print('----------------------------------------')

# def split_text(text, max_words_per_sentence=7, max_words_per_chunk=7):
#     full_text_splitted = [sentence+'.' for sentence in re.split(r'(?<!\.)\.(?!\.)', text)]
#     if full_text_splitted[-1]=='.':
#         full_text_splitted = full_text_splitted[:-1]
#     sentence_chunks = []
#     next_idx = -1
#     for sentence in full_text_splitted:
#         if len(sentence.split()) > max_words_per_sentence:
#             splitted_sentence = [sentence_part+',' if i+1!=len(sentence.split(',')) else sentence_part for i, sentence_part in enumerate(sentence.split(','))]
#             splits_len = len(splitted_sentence)
            
#             if splits_len!=1:
#                 for idx in range(len(splitted_sentence)):
#                     current_sentence = splitted_sentence[idx]
#                     if len(current_sentence.split())>max_words_per_sentence:
#                         words = sentence.split()
#                         rest = len(words) % max_words_per_chunk
#                         split_parts = math.floor(len(words) / max_words_per_chunk)
#                         if rest < 4:
#                             for i in range(split_parts-1):
#                                 sentence_chunks.append(' '.join(words[i*max_words_per_chunk:(i+1)*max_words_per_chunk]))
#                             sentence_chunks.append(' '.join(words[(i+1)*max_words_per_chunk:]))
#                         else:
#                             for i in range(split_parts):
#                                 sentence_chunks.append(' '.join(words[i*max_words_per_chunk:(i+1)*max_words_per_chunk]))
#                             sentence_chunks.append(' '.join(words[(i+1)*max_words_per_chunk:]))
#                     else:
#                         if idx > next_idx:
#                             if idx+1 < splits_len:
#                                 added_1 = splitted_sentence[idx] + splitted_sentence[idx+1]
#                                 if len(added_1.split()) < max_words_per_sentence:
#                                     if idx+2 < splits_len:
#                                         added_2 = added_1 + splitted_sentence[idx+2]
#                                         next_idx = idx+1
#                                         if len(added_2.split()) < max_words_per_sentence and idx+2 < splits_len:
#                                             sentence_chunks.append(added_2)
#                                             #added_3 = added_2 + splitted_sentence[idx+3]
#                                             next_idx = idx+2
#                                             # if len(added_3.split()) < max_words_per_sentence and idx+3 < splits_len:
#                                             #     sentence_chunks.append(added_3)
#                                             #     next_idx = idx+3
#                                             # else:
#                                             #     sentence_chunks.append(added_2)
#                                         else:
#                                             sentence_chunks.append(added_1)
#                                     else:
#                                         sentence_chunks.append(added_1)
#                                 else:
#                                     sentence_chunks.append(splitted_sentence[idx])   
#                             else:
#                                 sentence_chunks.append(splitted_sentence[idx]) 
#             else:
#                 words = sentence.split()
#                 rest = len(words) % max_words_per_chunk
#                 split_parts = math.floor(len(words) / max_words_per_chunk)
#                 if rest < 4:
#                     for i in range(split_parts-1):
#                         sentence_chunks.append(' '.join(words[i*max_words_per_chunk:(i+1)*max_words_per_chunk]))
#                     sentence_chunks.append(' '.join(words[(i+1)*max_words_per_chunk:]))
#                 else:
#                     for i in range(split_parts):
#                         sentence_chunks.append(' '.join(words[i*max_words_per_chunk:(i+1)*max_words_per_chunk]))
#                     sentence_chunks.append(' '.join(words[(i+1)*max_words_per_chunk:]))
#         else:
#             sentence_chunks.append(sentence)
#     return sentence_chunks

# # Example usage
# #text = "Ah oui, non, t'inquiète, t'inquiète. Tu dis, Dèborah, dès que c'est parti. Donc là, il faut que je fasse la même chose, Dèborah, c'est ça ? Oui."
# chunks = split_text(full_text)
# print('******************************************')
# print(chunks)


# words_dict = subtitles_data["chunks"]
# print(words_dict)

# # chunks_words = [len(chunk.split()) for chunk in chunks]
# # print(sum(chunks_words))
# # print(len(words_dict))


# def attach_timestamps_to_sentences(chunks, word_timestamps):
#     # Initialize an empty list to store the sentences with timestamps
#     sentences_with_timestamps = []
    
#     # Initialize variables to keep track of the current sentence and its timestamps
#     current_sentence = ''
#     current_start_timestamp = None
#     current_end_timestamp = None
#     chunk_idx = 0
    
#     # Iterate over each word in the word_timestamps dictionary
#     for word_info in word_timestamps:
#         if chunk_idx!=len(chunks):
#             word = word_info['text']
#             if current_start_timestamp is None:
#                 current_start_timestamp = word_info['timestamp']
#             current_chunk = chunks[chunk_idx]
#             last_timestamp = word_info['timestamp']
#             current_sentence += word
            
#             if current_sentence in ' '+current_chunk+' ':
#                 current_end_timestamp = last_timestamp
#             else:
#                 if chunk_idx==len(chunks)-1:
#                     current_end_timestamp = last_timestamp
#                 sentences_with_timestamps.append({
#                 'sentence': current_chunk,
#                 'start_timestamp': current_start_timestamp,
#                 'end_timestamp': current_end_timestamp})
#                 chunk_idx+=1
#                 current_sentence = ''
#                 current_start_timestamp = None

#     return sentences_with_timestamps


# print('__________________________________________________________')
# sentences_with_timestamps = attach_timestamps_to_sentences(chunks, words_dict)
# print(sentences_with_timestamps)

# print('__________________________________________________________')
# print('__________________________________________________________')
# current_sentence = ''
# for word_info in words_dict:
#             word = word_info['text']
#             current_sentence += word
# print(current_sentence)
# all_ch = ''
# for chunk in chunks:
#     all_ch+= (' ' + chunk)
# print(all_ch)

# print(len(chunk.split()))
# print(len(current_sentence.split()))
# print(len(words_dict))





######################################################################
import json
subtitles_json = '/home/kaibald231/work_pupil/pupil_mobile/s20_museum_test_2/20240308144312223/Converted_files/Preprocessed_divided/portion_BRUEGHEL LE JEUNE Pierre, Le paiement de la dîme, Inv 22 /transcriptions.json'
with open(subtitles_json, 'r', encoding='utf-8') as file:
    subtitles_data_word = json.load(file)
    
subtitles_json = '/home/kaibald231/work_pupil/pupil_mobile/s20_museum_test_2/20240308144312223/Converted_files/Preprocessed_divided/portion_BRUEGHEL LE JEUNE Pierre, Le paiement de la dîme, Inv 22 /transcriptions_sentence.json'
with open(subtitles_json, 'r', encoding='utf-8') as file:
    subtitles_data_sentence = json.load(file)
    


# draw the whole sentece in white if we entere the ts_start and <ts_end
# loop through the dict, if we in ts_color, draw 
    
white = [000]
red = [111]
sentence_color_list = {'sentences': [],  #ts, color
                       'sentence_time': []} 
sentence_color_list = {}
subtitles_sentence = subtitles_data_sentence['chunks']
subtitles_word = subtitles_data_word['chunks']
words_add = ''
for sentence in subtitles_sentence:
    sentence_text = sentence['text']
    (sentence_start, sentence_end) = sentence['timestamp']
    if sentence_end is None:
        sentence_end = sentence_start+20
    sentence_color_list[(sentence_start, sentence_end)] = []
    sentence_color_list[(sentence_start, sentence_end)].append([sentence_text, sentence_start, white])
    for word in subtitles_word:
        word_text = word['text']
        word_start, word_end = word['timestamp']
        if sentence_start<=word_start and word_end<=sentence_end:
            words_add += word_text
            sentence_color_list[(sentence_start, sentence_end)].append([words_add, word_start, red])
        else:
            words_add  = ''
            


print(sentence_color_list)
print('_______________________________________')
print(subtitles_word[-40:])
print('_______________________________________')
print(subtitles_sentence[-14:])



def generate_sentence_color_list(subtitles_data_sentence, subtitles_data_word):
    white =  (255, 255, 255)  # White color code
    red = (255, 0, 0)  # Red color code

    sentence_color_list = {}

    subtitles_sentence = subtitles_data_sentence['chunks']
    subtitles_word = subtitles_data_word['chunks']

    for sentence in subtitles_sentence:
        sentence_text = sentence['text']
        sentence_start, sentence_end = sentence['timestamp']
        if sentence_end is None:
            sentence_end = sentence_start + 20
        sentence_color_list[(sentence_start, sentence_end)] = []
        sentence_color_list[(sentence_start, sentence_end)].append([sentence_text, sentence_start, white])
        words_add = ''
        for word in subtitles_word:
            word_text = word['text']
            word_start, word_end = word['timestamp']
            if sentence_start <= word_start and word_end <= sentence_end:
                words_add += word_text
                sentence_color_list[(sentence_start, sentence_end)].append([words_add, word_start, red])
            else:
                words_add = ''

    return sentence_color_list

print('*******************************************************')

res = generate_sentence_color_list(subtitles_data_sentence, subtitles_data_word)
print(res)

