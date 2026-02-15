from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Your code goes here
        # you have to initialize self.model to a keras model
        self.model = Sequential([
            Rescaling(1/255, input_shape = input_shape),

            layers.Conv2D(32, (3,3), activation = 'relu', padding = 'same'),
            layers.MaxPooling2D(),

            layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same'),
            layers.MaxPooling2D(),

            layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same'),
            layers.MaxPooling2D(),

            layers.Flatten(),

            layers.Dense(128, activation = 'relu'),
            layers.Dropout(0.3),

            layers.Dense(categories_count, activation = 'softmax')
        ])
    
    def _compile_model(self):
        # Your code goes here
        # you have to compile the keras model, similar to the example in the writeup
        self.model.compile(
            optimizer = RMSprop(learning_rate = 0.001),
            loss = 'categorical_crossentropy',
            metrics = ['accuracy']
        )