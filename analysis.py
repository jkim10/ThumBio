import cv2
from scenedetect import VideoManager
from scenedetect import SceneManager
from tqdm import tqdm

import numpy as np
import sys
from PIL import Image
from scenedetect.detectors import ContentDetector

if(len(sys.argv) < 2):
    print(f"Usage: {sys.argv[0]} <Video File> <Performance Level (Empty: Slow 1: Fast 2: Faster 3: Fastest)>")
    quit()

file_name = sys.argv[1]
performance = 0
if(len(sys.argv) >= 3):
    performance = int(sys.argv[2])

color_weight = 0.85
motion_weight = 0.15
audio_weight = 0.0

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

def image_colorfulness(image):
	(B, G, R) = cv2.split(image.astype("float"))
	rg = np.absolute(R - G)
	yb = np.absolute(0.5 * (R + G) - B)
	(rbMean, rbStd) = (np.mean(rg), np.std(rg))
	(ybMean, ybStd) = (np.mean(yb), np.std(yb))
	stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
	meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
	return stdRoot + (0.3 * meanRoot)
def analyze_motion(frame,prvs):
    next = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    return next, mag

cap = cv2.VideoCapture(file_name)


scene_tuples = find_scenes(file_name)
scenes = {}
prvs = None
for index,scene in enumerate(tqdm(scene_tuples)):
    try:
        scenes[index] = {'color': [],'motion': [], 'audio': []}
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
            if(performance > 2):
                frame2 = cv2.resize(frame2, (0,0), fx=0.5, fy=0.5)
            # Color Analysis
            color_counts = image_colorfulness(frame2)
            scenes[index]['color'].append(color_counts)

            # Motion Analysis
            if (performance > 1):
                if((f%3)==0):
                    prvs, mag = analyze_motion(frame2,prvs)
                    scenes[index]['motion'].append(np.median(mag))
                    prvs = next
            else:
                prvs, mag = analyze_motion(frame2,prvs)
                scenes[index]['motion'].append(np.median(mag))

    except Exception as e:
        print(f"Skipping frame {index}: {e}")

# Calculate Most
color_stats = []
motion_stats = []

for scene in scenes.keys():
    if(len(np.array(scenes[scene]['color'])) == 0):
        color_stats.append(0.0)
    else:
        color_stats.append(np.array(scenes[scene]['color']).mean())
    if(len(np.array(scenes[scene]['motion'])) == 0):
        motion_stats.append(0.0)
    else:
        motion_stats.append(np.array(scenes[scene]['motion']).mean())
color_stats = np.array(color_stats)
motion_stats = np.array(motion_stats)

norm_color_stats = color_stats / np.linalg.norm(color_stats)
norm_motion_stats = motion_stats / np.linalg.norm(motion_stats)

final_scores = (color_weight * norm_color_stats) + (motion_weight * norm_motion_stats)

print(f"Scene {np.argmax(norm_color_stats)} is the most colorful")
print(f"Scene {np.argmax(norm_motion_stats)} has the most motion")


chosen = np.argmax(final_scores)
print(f"Scene {chosen} chosen")

# Start Finding the Most Representative Frame in the chosen
start_frame = scene_tuples[chosen][0]
end_frame = scene_tuples[chosen][1]
cap = cv2.VideoCapture(file_name)
cap.set(cv2.CAP_PROP_POS_FRAMES,int(start_frame))
lap_values = []
color_values = []
while(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != int(end_frame)):
    ret, frame = cap.read()
    color_value = image_colorfulness(frame)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    lap_values.append(lap)
    color_values.append(color_value)

lap_values = np.array(lap_values)
color_values = np.array(color_values)

norm_lap_values = lap_values / np.linalg.norm(lap_values)
norm_color_values = color_values / np.linalg.norm(color_values)

final_scene_scores = (.3*norm_lap_values) + (.7*norm_color_values)

best_frame = int(start_frame) + np.argmax(final_scene_scores)

cap.set(cv2.CAP_PROP_POS_FRAMES,int(best_frame))
ret,frame = cap.read()

print(f"Chose frame {best_frame} in Scene {chosen} as the most interesting. Downloading as Final.jpg")
cv2.imwrite("Final.jpg",frame)
