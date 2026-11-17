import os
import warnings

# Suppress TensorFlow logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='keras')

import tensorflow as tf # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import numpy as np # type: ignore

# Preparing the Dataset and Training the Model
def train_model():
    # Using ImageDataGenerator for data loading and augmentation
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        'images/',  
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(
        'images/',  
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    # Build the Model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')  # Number of classes
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the Model
    model.fit(train_generator, epochs=10, validation_data=validation_generator)

    # Save the Trained Model
    model.save('trained_model.h5')
    print("Model trained and saved as 'trained_model.h5'.")



if __name__ == "__main__":
    # Train the model before opening the camera
    train_model()
