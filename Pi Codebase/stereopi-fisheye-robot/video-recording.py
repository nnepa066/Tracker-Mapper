import picamera
from picamera import PiCamera
import time
import cv2
import numpy as np
import os
from datetime import datetime

# User quit method message 
print("You can press 'Q' to quit this script.")

# File for captured image
filename = './scenes/photo.png'

# Camera settimgs
cam_width = 1280
cam_height = 480

# Final image capture settings
scale_ratio = 0.5

# Camera resolution height must be dividable by 16, and width by 32
cam_width = int((cam_width+31)/32)*32
cam_height = int((cam_height+15)/16)*16
print ("Camera resolution: "+str(cam_width)+" x "+str(cam_height))

# Buffer for captured image settings
img_width = int (cam_width * scale_ratio)
img_height = int (cam_height * scale_ratio)
capture = np.zeros((img_height, img_width, 4), dtype=np.uint8)
print ("Scaled image resolution: "+str(img_width)+" x "+str(img_height))

# Initialize the camera
camera = PiCamera(stereo_mode='side-by-side',stereo_decimate=False)
camera.resolution=(cam_width, cam_height)
camera.framerate = 20
#camera.hflip = True

t0 = datetime.now()
counter = 0
avgtime = 0
# Capture frames from the camera
for frame in camera.capture_continuous(capture, format="bgra", use_video_port=True, resize=(img_width,img_height)):
    counter+=1
    cv2.imshow("pair", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop and save last image
    if key == ord("q") :
        camera.stop_recording()
        exit(0)
        break
    elif key == ord("r") :
        camera.start_recording('./cam_record/video_record.h264')
        # time.sleep(30)
