import h5py
import cv2 as cv

import gui

FRAME_SHAPE = (480, 640, 3)


class Preparer:
    def __init__(self):
        self.window = gui.Window()
        self.cap = cv.VideoCapture(0)

        self.frame_buffer = 32
        self.frame_index = -1

    def prepare(self):
        hdf5 = h5py.File('dataset.h5', 'w')
        self.frames = hdf5.create_dataset(
            "frames", 
            (self.frame_buffer,) + FRAME_SHAPE, 
            maxshape=(None,) + FRAME_SHAPE,
            compression="gzip", 
            compression_opts=3)
        self.positions = hdf5.create_dataset("positions", (self.frame_buffer, 2), maxshape=(None, 2))


        self.window.on_update = self.on_update

        try:
            self.window.start()
        except:
            pass

        self.frames.resize((self.frame_index + 1,) + FRAME_SHAPE)
        print('Dataset contain frames:', self.frame_index + 1)

        hdf5.close()
        self.cap.release()

    def on_update(self):
        ret, frame = self.cap.read()
        self.frame_index += 1
        self.frames[self.frame_index] = frame

        print('Frame', self.frame_index)

        if self.frame_index + 1 >= self.frame_buffer:
            self.frame_buffer *= 2
            self.frames.resize((self.frame_buffer,) + FRAME_SHAPE)
            print('Extend dataset to size:', self.frame_buffer)

if __name__ == '__main__':
    p = Preparer()
    p.prepare()

