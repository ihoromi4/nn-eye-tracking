import numpy as np
import keras
from keras import layers
from keras import models
from keras import optimizers


def create_model(shape):
    input_layer = x = layers.Input(shape)
    x = layers.MaxPooling2D(pool_size=(4, 4))(x)
    x = layers.Conv2D(32, (3, 3), use_bias=False)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, (3, 3), use_bias=False)(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(32, (5, 5), use_bias=False)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, (7, 7), use_bias=False)(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(32, (7, 7), use_bias=False)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(16, (9, 9), use_bias=False)(x)
    #x = layers.Activation('relu')(x)
    #x = layers.GlobalMaxPooling2D()(x)
    #x = layers.GlobalAveragePooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, use_bias=False)(x)
    x = layers.Activation('tanh')(x)
    x = layers.Dense(2, use_bias=False)(x)
    x = layers.Activation('sigmoid')(x)

    model = models.Model(input_layer, x)

    return model


def compile_model(model):
    optimizer = optimizers.Adam()

    model.compile(
        loss='mean_squared_error',
        optimizer=optimizer
    )


class KerasTracker:
    def __init__(self, shape, model=None):
        self.frames = np.empty((0,) + shape)
        self.positions = np.empty((0, 2))

        if model:
            self.model = models.load_model(model)
        else:
            self.model = create_model(shape)
            compile_model(self.model)

    def add_sample(self, frame, pos):
        self.frames = np.concatenate([self.frames, np.expand_dims(frame, 0)])
        self.positions = np.concatenate([self.positions, np.expand_dims(pos, 0)])

    def train(self):
        self.model.fit(
            self.frames,
            self.positions,
            batch_size=32,
            epochs=1
        )

    def predict(self, frame):
        return self.model.predict(frame)[0]

