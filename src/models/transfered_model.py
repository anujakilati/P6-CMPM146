from models.model import Model
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

MODEL_PATH = 'results/72.keras'

class TransferedModel(Model):
    def _define_model(self, input_shape, categories_count):
        loaded_model = load_model(MODEL_PATH)

        # Build the sequential model so input/output tensors are available
        loaded_model.predict(np.zeros((1, *input_shape)), verbose=0)

        # Freeze all layers so their weights are not updated during training
        for layer in loaded_model.layers:
            layer.trainable = False

        # remove last softmax
        base_output = loaded_model.layers[-2].output
        # new classification head
        x = layers.Dense(64, activation='relu', name='transfer_dense')(base_output)
        output = layers.Dense(categories_count, activation='softmax', name='transfer_output')(x)

        self.model = tf.keras.Model(inputs=loaded_model.inputs[0], outputs=output)

    def _compile_model(self):
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
