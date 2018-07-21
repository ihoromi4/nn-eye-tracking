import h5py
from keras import models

import kstrack

dataset = h5py.File('dataset.h5', 'r')
frames = dataset['frames']
positions = dataset['positions']

shape = frames.shape[1:]
model = kstrack.create_model(shape)
kstrack.compile_model(model)

model.summary()

model.fit(
    frames,
    positions,
    batch_size=32,
    epochs=10,
    shuffle="batch"
)

models.save_model(model, 'model.h5')

