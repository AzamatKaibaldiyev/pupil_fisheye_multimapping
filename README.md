# pupil_multimapping
gaze_multimapping for pupil glasses


source /home/kaibald231/work_pupil/pupil/pupil_env/bin/activate



# 2 choices

## Preprocessingpath
Take the folder with recordings and put them into Pupil player

*OR*

Try to replace it with batch_exporter  (later)

## Process Distorted video

- runner map gaze
- mapgaze (OR mapping gaze into reference only)
- videos concat
- heat_map + expansion (for the expansion retry with normal opencv method from gpt)
from moviepy.editor import VideoFileClip, concatenate_videoclips

# List of filenames of your MJPEG files in order
file_names = ['001.mjpeg', '002.mjpeg', '003.mjpeg', ...]

# Load each MJPEG file as a VideoFileClip
video_clips = [VideoFileClip(file) for file in file_names]

# Concatenate the video clips
final_clip = concatenate_videoclips(video_clips)

# Write the concatenated clip to an MP4 file
final_clip.write_videofile("output.mp4", codec="libx264")

## Process Undistorted video

- iMotion_exporter.py
- runner map gaze  (mapping to world video and reference)
- mapGaze
- videos concat
- heat_map + expansion
path



## OR

Run FT on distorted, but output on undistorted video and ref image

Test if FT work better on distorted or undistorted





## TODO

### Calibration
- Try screen calib on a projecctor in room F200, and compare it to the single marker, make a video of 5 mins (dots in room. walk around a hall, and again dots in the room)
- Check other calibrations from the museum
- Try the script on the single video (until gaze data file cutting part)
- IDEA: make a multiprocessing of converting several videos into mp4, then concatenate them (if it is faster (regular one takes 5 mins with 15 min video))
- Check the audio from microphone, compare its syncronisation with pupil audio
- Check the pupil audio and video offset?
- Record with recorder(other person) and you(pupil glasses, start later), make a double clap( at the begginning and the end), try to syncronise
- Ask discord about calibration at different distances, and merging calibrations
- Run the whole script until the end (some needs higher num_matches, some needs lower (for cutting and gaze mapping), also artifacts are present in the )
- Try apriltags?


### For  mapgaze.py
 - remove video creation part for ref2worldmapping

### For density map
- Try different method of expansion (from gpt for ex)
- Check the heatmap created (the focus time would not correspond exactly to the number of points in one frame, try to do heatmap according to the time spend on points or area)

### For concatentation (actually for mapgaze)
 - make the circle smaller for video mapping, and a bit bigger for reference image mapping
