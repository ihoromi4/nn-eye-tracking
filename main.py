import time
import math
import numpy as np
import cv2 as cv

import gui
import kstrack

FRAME_SHAPE = (480, 640, 3)


def main():
    cap = cv.VideoCapture(0)
    window = gui.Window()
    aim = window.create_circle((0, 0), 300, fill="#999")
    target = window.create_circle((500, 500), 50, fill="black")

    tracker = kstrack.KerasTracker(FRAME_SHAPE)

    def on_update():
        size = np.array([window.canvas.winfo_width(), window.canvas.winfo_height()])
        
        ret, frame = cap.read()
        #cv.imshow('frame', self.frame)
        #cv.waitKey(1)

        target_pos = target.pos / size
        print('Target pos:', target_pos)
        tracker.add_sample(frame, target_pos)
        tracker.train()

        pos = tracker.predict(np.expand_dims(frame, 0)) * size	
        print('pos:', pos)

        aim.pos = pos

        target.pos = np.array([900 + 200 * math.sin(time.time()), 500])

    window.on_update = on_update
    window.start()

if __name__ == '__main__':
    main()

