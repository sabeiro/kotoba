import numpy as np
import cv2
import glob
import time
import os

# termination criteria


curD = "/home/sabeiro/lav/viudi/no-key-board/"
fileN = curD + "test_image.png"
cascPath =  curD + "conf/" + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

ramp_frames = 30
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

import pygame.camera
pygame.camera.init()
cam = pygame.camera.Camera(pygame.camera.list_cameras()[0])
cam.start()
img = cam.get_image()
import pygame.image
pygame.image.save(img,fileN)
pygame.camera.quit()
del(cam)


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,0)
        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()


if cap.isOpened(): # try to get the first frame
    rval, frame = cap.read()
else:
    rval = False

while rval:
    rval, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cap.release()
cv2.destroyAllWindows()





def get_image():
 retval, im = cap.read()
 return im
 
for i in xrange(ramp_frames):
 temp = get_image()
print("Taking image...")
camera_capture = get_image()
cv2.imwrite(file, camera_capture)
del(camera)


# Change the camera setting using the set() function
# cap.set(cv2.CAP_PROP_EXPOSURE, -6.0)
# cap.set(cv2.CAP_PROP_GAIN, 4.0)
# cap.set(cv2.CAP_PROP_BRIGHTNESS, 144.0)
# cap.set(cv2.CAP_PROP_CONTRAST, 27.0)
# cap.set(cv2.CAP_PROP_HUE, 13.0) # 13.0
# cap.set(cv2.CAP_PROP_SATURATION, 28.0)
# Read the current setting from the camera
test = cap.get(cv2.CAP_PROP_POS_MSEC)
ratio = cap.get(cv2.CAP_PROP_POS_AVI_RATIO)
frame_rate = cap.get(cv2.CAP_PROP_FPS)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
contrast = cap.get(cv2.CAP_PROP_CONTRAST)
saturation = cap.get(cv2.CAP_PROP_SATURATION)
hue = cap.get(cv2.CAP_PROP_HUE)
gain = cap.get(cv2.CAP_PROP_GAIN)
exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
print("Test: ", test)
print("Ratio: ", ratio)
print("Frame Rate: ", frame_rate)
print("Height: ", height)
print("Width: ", width)
print("Brightness: ", brightness)
print("Contrast: ", contrast)
print("Saturation: ", saturation)
print("Hue: ", hue)
print("Gain: ", gain)
print("Exposure: ", exposure)
while True:
    ret, img = cap.read()
    cv2.imshow("input", img)
    
    key = cv2.waitKey(10)
    if key == 27:
        break

    cv2.destroyAllWindows()
    cv2.VideoCapture(0).release()


#capture from camera at location 0

cv2.destroyAllWindows() 
cap.release()


