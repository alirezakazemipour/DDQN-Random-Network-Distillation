from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam



class Model:
    def __init__(self, n_inputs, n_outputs, lr, do_compile):
        self.n_inputs =  n_inputs
        self.inputs = Input(shape = (self.n_inputs, ))
        self.lr = lr
        self.n_outputs = n_outputs
        self.opt = Adam(self.lr)

        x = self.inputs

        x = Dense(units = 256,
                  activation="relu",
                  kernel_initializer="re_normal")(x)
        x = Dense(units=256,
                  activation="relu",
                  kernel_initializer="re_normal")(x)
        self.outputs = Dense(self.n_outputs)(x)

        self.model = Model(self.inputs, self.outputs)

        if do_compile:
            self.model.compile(self.opt,
                               loss = "mse",
                               metrics = ["accuracy"])
