import numpy as np
import h5py
import cv2 as cv

import gui

DATASET_DEFAUL_FILE_PATH = 'dataset.h5'
FRAME_SHAPE = (480, 640)


class Preparer:
    def __init__(self, filepath=DATASET_DEFAUL_FILE_PATH, debug=False):
        self.filepath = filepath
        self.debug = debug

        self.window = gui.Window()
        self.aim = self.window.create_circle((500, 500), 50, 'black')

        self.cap = cv.VideoCapture(0)

        self.frame_buffer = 32  # internal size of dataset (buffer)
        self.frame_index = -1

        self.speed = np.zeros((2,))

    def prepare(self):
        """Start recording dataset from web-camera"""

        hdf5 = h5py.File(self.filepath, 'w')

        # dataset of video frames
        self.frames = hdf5.create_dataset(
            "frames", 
            (self.frame_buffer,) + FRAME_SHAPE, 
            dtype='uint8',
            maxshape=(None,) + FRAME_SHAPE,
            compression="gzip", 
            compression_opts=3)

        # dataset of target positions
        self.positions = hdf5.create_dataset(
            "positions",
            (self.frame_buffer, 2), 
            dtype='f',
            maxshape=(None, 2),
            compression="gzip", 
            compression_opts=3)

        self.window.on_update = self.on_update

        try:
            self.window.start()
        except:
            pass

        self.frames.resize((self.frame_index + 1,) + FRAME_SHAPE)
        self.positions.resize((self.frame_index + 1, 2))

        if self.debug:
            print('Dataset contain frames:', self.frame_index + 1)

        hdf5.close()
        self.cap.release()

    def update_aim_position(self):
        pos = self.aim.pos + self.speed.astype('int64')
        self.aim.pos = np.maximum(np.minimum(pos, self.window.get_size()), np.zeros(2))

        q = self.window.get_size() / 2 - self.aim.pos
        self.speed += q / np.linalg.norm(q) * 2
        self.speed += (np.random.random(2) - 0.5) * 15
        self.speed *= 0.98

    def on_update(self):
        """
        Add one gray frame to hdf5 dataset
        Scale dataset by 2 if full
        """

        self.update_aim_position()

        ret, frame = self.cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # convert frame to gray scale

        if self.debug:
            cv.imshow('frame', gray)
            cv.waitKey(1)

            print('Frame', self.frame_index)

        self.frame_index += 1
        self.frames[self.frame_index] = gray
        pos_norm = self.aim.pos / self.window.get_size()
        self.positions[self.frame_index] = pos_norm

        if self.debug:
            print('Position:', pos_norm)

        if self.frame_index + 1 >= self.frame_buffer:
            self.frame_buffer *= 2  # scale buffer size by 2

            self.frames.resize((self.frame_buffer,) + FRAME_SHAPE)  # scale dataset by 2
            self.positions.resize((self.frame_buffer, 2))

            if self.debug:
                print('Extend dataset to size:', self.frame_buffer)


def play_dataset(filepath):
    hdf5 = h5py.File(filepath, 'r')
    frames = hdf5['frames']

    for frame in frames:
        cv.imshow('frame', frame)
        if cv.waitKey(30) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()
    hdf5.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Prepare eye tracking dataset from your web cam.')

    parser.add_argument('command', type=str, choices=['record', 'play'], help='main command')
    parser.add_argument('--debug', action='store_true', help='output debug information')
    parser.add_argument('-d', '--dataset', type=str, default=DATASET_DEFAUL_FILE_PATH, metavar='DATASET', help='dataset file path')

    args = parser.parse_args()

    if args.command == 'record':
        preparer = Preparer(args.dataset, args.debug)
        preparer.prepare()
    elif args.command == 'play':
        play_dataset(args.dataset)

