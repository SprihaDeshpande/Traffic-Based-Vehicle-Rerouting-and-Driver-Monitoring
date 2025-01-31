import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np


class DriverEyeStatusModel:
    def __init__(self, train_dir, validation_dir, model_path='eye_status_model.h5'):
        self.train_dir = train_dir
        self.validation_dir = validation_dir
        self.model_path = model_path
        self.model = self.build_model()
        self.train_generator, self.validation_generator = self.create_data_generators()

    def create_data_generators(self):
        # Set up image data generator for loading and augmenting images
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # For validation images, only rescaling is done (no augmentation)
        validation_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary'
        )

        validation_generator = validation_datagen.flow_from_directory(
            self.validation_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary'
        )

        return train_generator, validation_generator

    def build_model(self):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # For binary classification (open/closed)
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, epochs=20):
        history = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.train_generator.samples // 32,  # More accurate steps per epoch calculation
            epochs=epochs,  # Increased epochs
            validation_data=self.validation_generator,
            validation_steps=self.validation_generator.samples // 32  # More accurate steps per validation
        )
        self.model.save(self.model_path)
        return history

    def predict(self, img_path):
        # Load and process the image
        img = load_img(img_path, target_size=(224, 224))  # Resize image to match the model's input size
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize the image to [0, 1]

        # Make the prediction using the model
        prediction = self.model.predict(img_array)

        # Interpret the result (assuming 0 = sleeping, 1 = awake)
        return "Drive Awake" if prediction >= 0.5 else "Driver Sleeping"


# Example usage:
train_dir = '/Users/sprihad/Desktop/essay4/dataset/train'  # Change this to your training data path
validation_dir = '/Users/sprihad/Desktop/essay4/dataset/validation'  # Change this to your validation data path

# Initialize the model
model = DriverEyeStatusModel(train_dir, validation_dir)

# Train the model (if not already trained)
model.train_model(epochs=20)

# Predict on a new image (driver.png)
img_path = '/Users/sprihad/Desktop/essay4/driver.png'  # Change this to the path of the image you want to classify
status = model.predict(img_path)
print(f"The driver is {status}.")
