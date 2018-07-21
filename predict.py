import numpy as np
import cv2 as cv
from keras import models

import gui
import kstrack

FRAME_SHAPE = (480, 640, 1)

tracker = kstrack.KerasTracker(FRAME_SHAPE, model='model.h5')

cap = cv.VideoCapture(0)
window = gui.Window()
aim = window.create_circle((0, 0), 300, fill="#999")

def on_update():
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.imshow('frame', gray)
    cv.waitKey(1)

    pos = tracker.predict(np.expand_dims(gray.reshape(FRAME_SHAPE), 0)) * window.get_size()	
    print('pos:', pos)

    aim.pos = pos

window.on_update = on_update
window.start()

