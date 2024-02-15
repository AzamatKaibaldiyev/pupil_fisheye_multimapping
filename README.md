# pupil_multimapping
gaze_multimapping for pupil glasses


source /home/kaibald231/work_pupil/pupil/pupil_env/bin/activate



# 2 choices

## Preprocessing
Take the folder with recordings and put them into Pupil player

*OR*

Try to replace it with batch_exporter  (later)

## Process Distorted video

- runner map gaze
- mapgaze (OR mapping gaze into reference only)
- videos concat
- heat_map + expansion (for the expansion retry with normal opencv method from gpt)


## Process Undistorted video

- iMotion_exporter.py
- runner map gaze  (mapping to world video and reference)
- mapGaze
- videos concat
- heat_map + expansion




## OR

Run FT on distorted, but output on undistorted video and ref image

Test if FT work better on distorted or undistorted





## TODO

Collect and rename the scripts, push to the github
Test batch exporter

### For video cutting.py
- try to reduce steps in video cutting

- Create a dict with frame number and time  as kay value, go back 3 iterations(30 frames), and replace start frame and start time, same with end frame and end time (around 1 second from both sides) 


### For  mapgaze.py
 - remove video creation part for ref2worldmapping

### For density map
- Try different method of exppansion (from gpt for ex)

### For concatentation (actually for mapgaze)
 - make the circle smaller for video mapping, and a bit bigger for reference image mapping
