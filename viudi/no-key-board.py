import picamera
from time import sleep

camera = picamera.PiCamera()
camera.brightness = 70
camera.sharpness = 0
camera.contrast = 0
camera.brightness = 50
camera.saturation = 0
camera.ISO = 0
camera.video_stabilization = False
camera.exposure_compensation = 0
camera.exposure_mode = 'auto'
camera.meter_mode = 'average'
camera.awb_mode = 'auto'
camera.image_effect = 'none'
camera.color_effects = None
camera.rotation = 0
camera.hflip = False
camera.vflip = False
camera.crop = (0.0, 0.0, 1.0, 1.0)


camera.start_preview()
sleep(10)
camera.stop_preview()

for i in range(100):
    camera.brightness = i
    sleep(0.2)



# camera.start_recording('video.h264')
# sleep(5)
# camera.stop_recording()



