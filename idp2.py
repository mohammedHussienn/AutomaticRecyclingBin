import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.models import load_model
import os
from PIL import Image


# Function to validate and remove corrupted images
def validate_and_remove_corrupted_images(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(root, file)
                try:
                    img = Image.open(file_path)
                    img.verify()  # Verify that it is an image
                except (IOError, SyntaxError) as e:
                    print(f"Removing bad file: {file_path} - {e}")
                    os.remove(file_path)

# Validate and remove corrupted images from your training and validation sets
validate_and_remove_corrupted_images('/Users/macbook/Desktop/TrashBox-main/TrashBox_train_set')
validate_and_remove_corrupted_images('/Users/macbook/Desktop/TrashBox-testandvalid-main/TrashBox_testandvalid_set')

# Load the pre-trained EfficientNet model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = tf.keras.layers.Dense(7, activation='softmax')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Load the TrashBox dataset and fine-tune the model
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory(
    '/Users/macbook/Desktop/TrashBox-main/TrashBox_train_set', 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='categorical',  # Ensure class_mode is 'categorical'
    shuffle=True  # Shuffle training data
)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)
validation_generator = validation_datagen.flow_from_directory(
    '/Users/macbook/Desktop/TrashBox-testandvalid-main/TrashBox_testandvalid_set/val',  # Point to 'val' subdirectory
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',  # Ensure class_mode is 'categorical'
    shuffle=False  # Don't shuffle validation data to keep it consistent
)

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=20
)

# Save the model after training
model.save('trash_classification_model.h5')


# Test the model using the 'test' dataset
test_generator = validation_datagen.flow_from_directory(
    '/Users/macbook/Desktop/TrashBox-testandvalid-main/TrashBox_testandvalid_set/test',  # Point to 'test' subdirectory
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f"Test Accuracy: {test_accuracy:.2f}, Test Loss: {test_loss:.2f}")

