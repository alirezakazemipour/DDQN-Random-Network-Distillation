from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
import numpy as np


class model:
    def __init__(self, n_inputs, n_outputs, lr, do_compile):
        self.n_inputs =  n_inputs
        self.inputs = Input(shape = (self.n_inputs, ), name = "Input_layer")
        self.lr = lr
        self.n_outputs = n_outputs
        self.opt = Adam(self.lr)

        x = self.inputs

        x = Dense(units = 256,
                  activation="relu",
                  kernel_initializer="he_normal")(x)
        x = Dense(units=256,
                  activation="relu",
                  kernel_initializer="he_normal")(x)
        self.outputs = Dense(self.n_outputs)(x)

        self.model = Model(self.inputs, self.outputs)

        if do_compile:
            self.model.compile(self.opt,
                               loss = self.loss,
                               metrics = ["accuracy"])
            print(self.model.metrics_names)
            self.model.summary()

    @staticmethod
    def loss(y_true, y_pred):
        IS = K.reshape(y_true[:, -1], (-1, 1))
        y_true = y_true[:, :-1]
        return K.mean( IS * K.square(y_true - y_pred))

    def predict(self, x):

        return self.model.predict(x)

    def train_on_batch(self, x, y):

        return self.model.train_on_batch(x, y)

    def set_weights(self,x):
        return self.model.set_weights(x)

    def get_weights(self):
        return self.model.get_weights()