import os
import sys
from pydub import AudioSegment
import json
from moviepy.editor import VideoFileClip, AudioFileClip
import audalign as ad
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from PIL import Image, ImageFont, ImageDraw
import cv2
import numpy as np
import multiprocessing
import subprocess




# Aligning audios
def sync_audios(pupil_audio, microph_audio, destination_path):
    correlation_rec = ad.CorrelationRecognizer()
    cor_spec_rec = ad.CorrelationSpectrogramRecognizer()

    results = ad.align_files(
        pupil_audio,
        microph_audio,
        destination_path = destination_path,
        recognizer=correlation_rec
    )

    # results can then be sent to fine_align
    fine_results = ad.fine_align(
        results,
        recognizer=cor_spec_rec,
    )
    pupil_audio_name = pupil_audio.split('/')[-1]
    offset_pupil = fine_results[pupil_audio_name]
    if offset_pupil==0:
        micr_audio_name = microph_audio.split('/')[-1]
        offset_pupil = -fine_results[micr_audio_name]

    return offset_pupil 


# Get audio length
def get_audio_duration(audio_file):
    command = [
        "ffprobe",
        "-i", audio_file,
        "-show_entries", "format=duration",
        "-v", "quiet",
        "-of", "csv=p=0"
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    duration = float(result.stdout.strip())
    return duration


# Adjusting microphone audio to pupil world video
def adjust_audio_offset(input_audio_path, output_audio_path, offset_beginning, offset_end):
    # Load the input audio file
    audio = AudioSegment.from_file(input_audio_path)
    
    # Calculate the offset_beginning in milliseconds
    offset_ms = int(offset_beginning * 1000)  # Convert offset_beginning from seconds to milliseconds
    
    if offset_ms > 0:
        # If offset_beginning is positive, trim the audio from the beginning
        adjusted_audio = audio[offset_ms:]
        print(f"Trimming the beginning by {offset_ms/1000} seconds")
    else:
        # If offset_beginning is negative, add silence duration at the beginning
        offset_ms = abs(offset_ms)
        silence_duration = AudioSegment.silent(duration=offset_ms)
        adjusted_audio = silence_duration + audio
        print(f"Adding duration to the beginning by {offset_ms/1000} seconds")

    # Calculate the offset_end in milliseconds
    offset_ms = int(offset_end * 1000)  # Convert offset_beginning from seconds to milliseconds
    
    if offset_ms > 0:
        # If offset_end is positive, trim the audio from the end
        adjusted_audio = adjusted_audio[:len(adjusted_audio)-offset_ms]
        print(f"Trimming the end by {offset_ms/1000} seconds")
    else:
        # If offset_end is negative, add silence duration at the end
        offset_ms = abs(offset_ms)
        silence_duration = AudioSegment.silent(duration=offset_ms)
        adjusted_audio = adjusted_audio + silence_duration
        print(f"Adding duration to the end by {offset_ms/1000} seconds")
    
    # Export the adjusted audio to the output file
    adjusted_audio.export(output_audio_path, format="mp3")


# Cutting audios
def cut_audio(input_audio_path, timestamps, output_path):
    # Load the audio file
    audio = AudioSegment.from_mp3(input_audio_path)

    # Iterate over timestamps
    start_time, end_time  = timestamps
    # Extract audio segment
    segment = audio[start_time : end_time]  
    
    # Export audio segment
    segment.export(output_path, format="mp3")


# Generate transcriptions and save it
def generate_and_save_transcriptions(audio_json_folder_name):
    
    # Define model and processor
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Define the pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    for (sample_file_path, json_file_path, json_file_path_sentence, folder_name) in audio_json_folder_name:
        # Check if the audio file exists
        if not os.path.exists(sample_file_path):
            print("Error: Sample file does not exist.")
            return
        
        # Process the audio file
        result = pipe(sample_file_path, return_timestamps="word", generate_kwargs={"language": "french"})
        # Save the result as JSON
        with open(json_file_path, 'w') as json_file:
            json.dump(result, json_file)
        
        print(f"Transcriptions for {folder_name} are saved to: {json_file_path}")

        result = pipe(sample_file_path, return_timestamps=True, generate_kwargs={"language": "french"})
        # Save the result as JSON
        with open(json_file_path_sentence, 'w') as json_file:
            json.dump(result, json_file)
        
        print(f"Transcriptions for {folder_name} are saved to: {json_file_path_sentence}")

# def generate_subtitled_video(input_video_path, subtitles_json, output_video_path, folder):
#     # Load the input video
#     video = cv2.VideoCapture(input_video_path)
#     fps = video.get(cv2.CAP_PROP_FPS)

#     # Parse the subtitles JSON with proper encoding
#     with open(subtitles_json, 'r', encoding='utf-8') as file:
#         subtitles_data = json.load(file)
#     #text = subtitles_data["text"]
#     chunks = subtitles_data["chunks"]

#     # Initialize video writer
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
#     print('Saving ', output_video_path)

#     # Load a font supporting accented characters
#     font_path = 'AbhayaLibre-Regular.ttf'
#     #font_path = 'adventpro-regular.ttf'
#     font_size = 35
#     font = ImageFont.truetype(font_path, font_size)

#     # Iterate over video frames
#     frame_idx = 0

#     while True:
#         ret, frame = video.read()
#         if not ret:
#             break

#         # Check if current frame falls within any subtitle chunk
#         current_time = frame_idx / fps
#         for chunk in chunks:
#             start_time, end_time = chunk["timestamp"]
#             if start_time <= current_time <= end_time:
#                 # Decode the text
#                 text = chunk["text"]
                
#                 # Render text on the frame using PIL
#                 pil_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert OpenCV frame to PIL format
#                 pil_img = Image.fromarray(pil_frame)
#                 draw = ImageDraw.Draw(pil_img)
#                 draw.text((int(width*0.45), int(height*0.9)), text, font=font, fill=(255, 255, 255))  # Adjust coordinates as needed
#                 frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)  # Convert back to OpenCV format

#         # Write the frame with subtitles to output video
#         out.write(frame)
#         frame_idx += 1

#     # Release video capture and writer
#     video.release()
#     out.release()
#     print(f'Finished adding subtitles to video for: {folder}')

def generate_subtitled_video(input_video_path, subtitles_json, output_video_path, folder):
    # Load the input video
    video = cv2.VideoCapture(input_video_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    # Parse the subtitles JSON with proper encoding
    with open(subtitles_json, 'r', encoding='utf-8') as file:
        subtitles_data = json.load(file)
    full_text = subtitles_data["text"]
    full_text_splitted = [sentence+['.'] for sentence in full_text.split('.')]
    sentence_chunks = []
    for sentence in full_text_splitted:
        if len(sentence.split())<10:
            splitted_sentence = [sentence+['.'] for sentence in splitted_sentence.split(',')]
            splits_len = len(splitted_sentence)
            for idx in range(len(splitted_sentence)):
                if idx<splits_len:
                    splitted_sentence[idx] + splitted_sentence[idx]     
        else:
            sentence_chunks.append(sentence)

    chunks = subtitles_data["chunks"]

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    print('Saving ', output_video_path)

    # Load a font supporting accented characters
    font_path = 'AbhayaLibre-Regular.ttf'
    #font_path = 'adventpro-regular.ttf'
    font_size = 35
    font = ImageFont.truetype(font_path, font_size)

    # Iterate over video frames
    frame_idx = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Check if current frame falls within any subtitle chunk
        current_time = frame_idx / fps
        for chunk in chunks:
            start_time, end_time = chunk["timestamp"]
            if start_time <= current_time <= end_time:
                # Decode the text
                text = chunk["text"]
                
                # Render text on the frame using PIL
                pil_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert OpenCV frame to PIL format
                pil_img = Image.fromarray(pil_frame)
                draw = ImageDraw.Draw(pil_img)
                draw.text((int(width*0.45), int(height*0.9)), text, font=font, fill=(255, 255, 255))  # Adjust coordinates as needed
                frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)  # Convert back to OpenCV format

        # Write the frame with subtitles to output video
        out.write(frame)
        frame_idx += 1

    # Release video capture and writer
    video.release()
    out.release()

    print(f'Finished adding subtitles to video for: {folder}')






# Add audio portions to video portions
def merge_audio_with_video(video_path, audio_path, output_path):
    # Load video and audio clips
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)
    
    # Set audio of the video clip to the loaded audio clip
    video_clip = video_clip.set_audio(audio_clip)

    # Write the new video file with the merged audio
    video_clip.write_videofile(output_path)





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


if __name__ == "__main__":
    if len(sys.argv)==4:

        exported_data_folder = sys.argv[1]
        general_folder = sys.argv[2]
        microph_audio = sys.argv[3]
        pupil_audio = [os.path.join(general_folder,audio_name) for audio_name in os.listdir(general_folder) if audio_name.endswith('.mp4')][0]
        preprocessed_divided_folder = os.path.join(general_folder, 'Converted_files', 'Preprocessed_divided')


        # # AUDIO SYNC
        # print("STARTING AUDIO SYNC ##############################")
        # destination_path=os.path.join(general_folder, 'Converted_files', 'audio_syncs')
        # os.makedirs(destination_path,exist_ok=True)
        # offset_pupil = sync_audios(pupil_audio, microph_audio, destination_path)                                   # Function execution
        # print('Audio aligning is complete')

        # length_pupil_audio= get_audio_duration(pupil_audio)
        # length_microphone_audio= get_audio_duration(microph_audio)
        # offset_end = length_microphone_audio - (length_pupil_audio + offset_pupil)
        adjusted_audio_path = os.path.join(general_folder, 'Converted_files', 'synced_audio_adjusted.mp3')
        # # Adjust the audio with offsets
        # adjust_audio_offset(microph_audio, adjusted_audio_path, offset_pupil, offset_end)                          # Function execution
        # print('Audio adjsuting is complete')

        # print(f"""
        #     Length of microphone audio file {microph_audio}: {length_microphone_audio} seconds
        #     Length of pupil audio file {pupil_audio}: {length_pupil_audio} seconds
        #     Offset beginning is: {offset_pupil} 
        #     Offset end is: ", {offset_end} 
        #     Final length of the microphone audio is {get_audio_duration(adjusted_audio_path)}
        #     """)
        

        # # AUDIO CUT
        # print("STARTING AUDIO CUT ##############################")
        # for folder in os.listdir(preprocessed_divided_folder):
        #     folder_path = os.path.join(preprocessed_divided_folder, folder)
        #     timestamp_file = [file for file in os.listdir(folder_path) if file.endswith('.json')][0]
        #     with open(os.path.join(folder_path, timestamp_file), 'r') as file:
        #         # Read the JSON data
        #         data = json.load(file)
        #         start_ts = data['StartTimestamp']
        #         end_ts = data['EndTimestamp']

        #     output_path = os.path.join(folder_path, 'corresponding_audio.mp3')
        #     cut_audio(adjusted_audio_path, (start_ts, end_ts), output_path)                                        # Function execution
        #     print(f'Finished trimming audio for {folder}')


        # # AUDIO TRANSCRIPTIONS
        # print("STARTING AUDIO TRANSCRIPTION ##############################")
        # sample_file_paths = []
        # for folder in os.listdir(preprocessed_divided_folder):
        #     folder_path = os.path.join(preprocessed_divided_folder, folder)
        #     ref_audio_file = [os.path.join(folder_path,file) for file in os.listdir(folder_path) if file.endswith('.mp3')][0]
        #     json_file_path = os.path.join(folder_path,'transcriptions.json')
        #     json_file_path_sentence = os.path.join(folder_path,'transcriptions_sentence.json')
        #     sample_file_paths.append((ref_audio_file, json_file_path, json_file_path_sentence,  folder))

        # generate_and_save_transcriptions(sample_file_paths)                                                        # Function execution

        
        # # ADDING SUBTITLES AND AUDIO
        print("STARTING SUBTITLES AND AUDIO ADDITION ##############################")
        final_results_folder = os.path.join(exported_data_folder, 'Result_outputs', 'Final_results')

        # # Subtitles
        # processes = []
        # for folder in os.listdir(preprocessed_divided_folder):
        #     folder_path = os.path.join(preprocessed_divided_folder, folder)
        #     folder_final_results_path = os.path.join(final_results_folder, folder)
        #     input_video_path = [os.path.join(folder_final_results_path, file) for file in os.listdir(folder_final_results_path) if file.endswith('.mp4')][0]
        #     output_video_path = os.path.join(folder_final_results_path,  'video_with_subs_and_audio.mp4')
        #     subtitles_json = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('transcriptions.json')][0]
        #     single_process = multiprocessing.Process(target=generate_subtitled_video, args=(input_video_path, subtitles_json, output_video_path, folder)) # Function execution
        #     processes.append(single_process)
        #     single_process.start()
        # for process in processes:
        #     process.join()

        # Audio
        # processes = []
        # for folder in os.listdir(preprocessed_divided_folder):
        #     folder_path = os.path.join(preprocessed_divided_folder, folder)
        #     video_path = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('video_with_subs_and_audio.mp4')][0]
        #     audio_path = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.mp3')][0]
        #     print(audio_path)
        #     output_path = os.path.join(folder_path, 'video_with_subs_and_audio.mp4')
        #     single_process = multiprocessing.Process(target=merge_audio_with_video, args=(video_path, audio_path, output_path))         # Function execution
        #     processes.append(single_process)
        #     single_process.start()
        #     print(f'Finished adding audio to video for: {folder}')
        # for process in processes:
        #     process.join()


        for folder in os.listdir(preprocessed_divided_folder):
            folder_path = os.path.join(preprocessed_divided_folder, folder)
            folder_final_results_path = os.path.join(final_results_folder, folder)
            video_path = [os.path.join(folder_final_results_path, file) for file in os.listdir(folder_final_results_path) if file.endswith('video_with_subs_and_audio.mp4')][0]
            output_path = os.path.join(folder_final_results_path, 'video_with_subs_and_audio.mp4')
            audio_path = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.mp3')][0]
            merge_audio_with_video(video_path, audio_path, output_path)                 # Function execution
            print(f'Finished adding audio to video for: {folder}')












