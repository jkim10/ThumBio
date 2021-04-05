import cv2
import numpy as np
import sys
from pydub import AudioSegment

from scenedetect import VideoManager
from scenedetect import SceneManager
from tqdm import tqdm
from scenedetect.detectors import ContentDetector
from PIL import Image


if(len(sys.argv) < 2):
    print(f"Usage: {sys.argv[0]} <Video File> <Performance Level (Empty: Slow 1: Fast 2: Faster 3: Fastest)>")
    quit()

file_name = sys.argv[1]
performance = 0
if(len(sys.argv) >= 3):
    performance = int(sys.argv[2])

color_weight = .4
saturation_weight = .35
motion_weight = .2
audio_weight = .1




# DFT Blur detection
def detect_blur(image):
    size = 60
    (h,w) = image.shape
    (cX, cY) = (int(w / 2.0) , int(h / 2.0))
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)
    fftShift[(cY - size):(cY + size), cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    return (mean)
    
# Min Max Normalization on Numpy Array
def normalize(arr):
    return (arr - arr.min())/ (arr.max() - arr.min())

# Find Scene Boundaries
def find_scenes(video_path, threshold=30.0):
    # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))

    # Improve processing speed by downscaling before processing.
    video_manager.set_downscale_factor()

    # Start the video manager and perform the scene detection.
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    # Each returned scene is a tuple of the (start, end) timecode.
    return scene_manager.get_scene_list()


# Hasler and Suesstrunk
def image_colorfulness(image):
	(B, G, R) = cv2.split(image.astype("float"))
	rg = np.absolute(R - G)
	yb = np.absolute(0.5 * (R + G) - B)
	(rbMean, rbStd) = (np.mean(rg), np.std(rg))
	(ybMean, ybStd) = (np.mean(yb), np.std(yb))
	stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
	meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
	return stdRoot + (0.3 * meanRoot)

def save_frame(frame):
    cap = cv2.VideoCapture(file_name)
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame))
    ret,frame = cap.read()
    cv2.imwrite("debug.jpg",frame)
# Optical Flow Calculation
def analyze_motion(frame,prvs):
    next = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    return next, mag

def find_most_rep(frameTuple):
    start_frame = frameTuple[0]
    end_frame = frameTuple[1]
    color_values = []
    dft_values = []
    sat_values = []
    cap = cv2.VideoCapture(file_name)
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(start_frame))
    while(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != int(end_frame)):
        ret, frame = cap.read()
        color_value = image_colorfulness(frame)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mean = detect_blur(gray)
        color_values.append(color_value)
        dft_values.append(mean)
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = img_hsv[:, :, 1].mean()
        sat_values.append(saturation)


    color_values = np.array(color_values)
    dft_values = np.array(dft_values)
    norm_saturation_values = normalize(np.array(sat_values))
    norm_color_values = normalize(color_values)
    norm_dft_values = normalize(dft_values)
    final_scene_scores = (.2 * norm_color_values) + (5 * norm_dft_values) + (.3*norm_saturation_values)
    best_frame = int(start_frame) + np.argmax(final_scene_scores)
    return best_frame

cap = cv2.VideoCapture(file_name)
sound = AudioSegment.from_file(file_name)
print("==========================")
print("Detecting scene boundaries")
print("==========================")
scene_tuples = find_scenes(file_name)

def save_scene(scene_num):
    save_frame(find_most_rep(scene_tuples[scene_num]))

scenes = {}
prvs = None
print("==========================")
print("Analyzing Scenes")
print("==========================")
for index,scene in enumerate(tqdm(scene_tuples)):
    try:
        scenes[index] = {'color': [], 'sat': [],'motion': [], 'audio': 0}

        ret, frame1 = cap.read()
        if(performance == 3):
            frame1 = cv2.resize(frame1, (0,0), fx=0.5, fy=0.5)
        color_counts = image_colorfulness(frame1)
        scenes[index]['color'].append(color_counts) 
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255
        
        for f in range(int(scene[0])+1,int(scene[1])+1):
            flag, frame2 = cap.read()
            if not flag:
                break
            if(performance >= 2):
                frame2 = cv2.resize(frame2, (0,0), fx=0.5, fy=0.5)
            # Color Analysis
            color_counts = image_colorfulness(frame2)
            img_hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
            saturation = img_hsv[:, :, 1].mean()
            scenes[index]['color'].append(color_counts)
            scenes[index]['sat'].append(saturation)
            # Motion Analysis
            if (performance >= 1):
                if((f%3)==0):
                    prvs, mag = analyze_motion(frame2,prvs)
                    scenes[index]['motion'].append(np.mean(mag))
            else:
                prvs, mag = analyze_motion(frame2,prvs)
                scenes[index]['motion'].append(np.mean(mag))
        start_ms = scene[0].get_seconds()*1000
        end_ms = scene[1].get_seconds()*1000
        s_segment = sound[start_ms:end_ms]

        loudness = s_segment.dBFS
        scenes[index]['audio'] = loudness
    except Exception as e:
        print(f"Skipping frame {index}: {e}")

# Calculate Most
color_stats = []
saturation_stats = []
motion_stats = []
audio_stats = []

for scene in scenes.keys():
    if(len(np.array(scenes[scene]['color'])) == 0):
        color_stats.append(0.0)
    else:
        color_stats.append(np.nan_to_num(np.array(scenes[scene]['color']),posinf=0, neginf=0).mean())
    if(len(np.array(scenes[scene]['motion'])) == 0):
        motion_stats.append(0.0)
    else:
        motion_stats.append(np.nan_to_num(np.array(scenes[scene]['motion']),posinf=0, neginf=0).mean())
    if(len(np.array(scenes[scene]['sat'])) == 0):
        saturation_stats.append(0.0)
    else:
        saturation_stats.append(np.nan_to_num(np.array(scenes[scene]['sat']),posinf=0, neginf=0).mean())
    audio_stats.append(scenes[scene]['audio'])


color_stats = np.array(color_stats)
saturation_stats = np.array(saturation_stats)
motion_stats = np.array(motion_stats)
audio_stats = np.array(audio_stats)

norm_color_stats = normalize(color_stats)
norm_motion_stats = normalize(motion_stats)
norm_saturation_stats = normalize(saturation_stats)
norm_audio_stats = normalize(audio_stats)
final_scores = (color_weight * norm_color_stats) - (motion_weight * norm_motion_stats) + (saturation_weight * norm_saturation_stats) + (audio_weight * norm_audio_stats)

print(f"Scene {np.argmax(norm_color_stats)} is the most colorful")
print(f"Scene {np.argmax(norm_saturation_stats)} is the most saturated")
print(f"Scene {np.argmax(norm_motion_stats)} has the most motion (negative factor)")
print(f"Scene {np.argmax(norm_audio_stats)} is the loudest")


chosen = np.argmax(final_scores)
print(f"Scene {chosen} chosen")

# Start Finding the Most Representative Frame in the chosen





best_frame = find_most_rep(scene_tuples[chosen])
cap.set(cv2.CAP_PROP_POS_FRAMES,int(best_frame))
ret,frame = cap.read()
print(f"Chose frame {best_frame} in Scene {chosen} as the most interesting. Downloading as output.jpg")
cv2.imwrite(f"output.jpg",frame)
import code; code.interact(local=dict(globals(), **locals()))
