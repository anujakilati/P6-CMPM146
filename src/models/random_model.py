from models.model import Model
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

MODEL_PATH = 'results/72.keras'

class RandomModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Load the same pre-trained model architecture
        loaded_model = load_model(MODEL_PATH)

        # Build the sequential model so input/output tensors are available
        loaded_model.predict(np.zeros((1, *input_shape)), verbose=0)

        # Randomize all weights before freezing (baseline — no learned features)
        RandomModel._randomize_layers(loaded_model)

        # Freeze all layers
        for layer in loaded_model.layers:
            layer.trainable = False

        # Remove the final softmax layer and add a new head
        base_output = loaded_model.layers[-2].output

        x = layers.Dense(64, activation='relu', name='random_dense')(base_output)
        output = layers.Dense(categories_count, activation='softmax', name='random_output')(x)

        self.model = tf.keras.Model(inputs=loaded_model.inputs[0], outputs=output)

    def _compile_model(self):
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    @staticmethod
    def _randomize_layers(model):
        for layer in model.layers:
            weights = layer.get_weights()
            if weights:
                new_weights = [np.random.normal(0, 0.05, w.shape) for w in weights]
                layer.set_weights(new_weights)
