import tensorflow as tf
import os
import matplotlib.pyplot as plt

# Path to the folder containing images
folder_path = r"folder path"

import pandas as pd

excel_path = r"C:\Users\imsha\Downloads\ML_TRIAL_2\ML count_real_all.xlsx"
# Read the Excel file
df = pd.read_excel(excel_path)

# Create a dictionary to store image names and their colony counts
image_colony_dict = {}

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    # Get the image name and colony count from the respective columns
    image_name = row['image name']
    colony_count = row['colony count']
    
    # Add the image name and colony count to the dictionary
    colony_counts_dict[image_name] = colony_count

# Now, image_colony_dict contains the mapping of image names to colony counts
print(colony_counts_dict)




# List to store image file names (without extensions)
image_paths = list(colony_counts_dict.keys())

def load_and_preprocess_image(image_path, augment=False, target_size=(256, 256)):
    # Construct the full path including the file extension
    full_path = os.path.join(folder_path, image_path)
    
    # Load image and resize
    img = tf.io.read_file(full_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, target_size)
    
    # Augmentation
    if augment:
        # Apply random transformations for data augmentation
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
    
    # Normalize pixel values to range [0, 1]
    img = tf.cast(img, tf.float32) / 255.0
    return img

# Function to generate augmented images from an original image
def generate_augmented_images(image_path, colony_count):
    augmented_images = []
    for _ in range(5):  # Reduce the number of augmented images
        # Load and preprocess the original image with augmentation
        img = load_and_preprocess_image(image_path, augment=True, target_size=(256, 256))
        augmented_images.append((img, colony_count))
    return augmented_images

# Data structure to store augmented images and their labels
augmented_data = []

# Generate augmented images for each original image
for image_path, colony_count in colony_counts_dict.items():
    augmented_data.extend(generate_augmented_images(image_path, colony_count))

# Shuffle the augmented data
import random
random.shuffle(augmented_data)

# Separate images and labels
images = [pair[0] for pair in augmented_data]
labels = [pair[1] for pair in augmented_data]

# Convert images and labels to TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices((images, labels))

# Example usage:
# Iterate over the dataset
for image, label in dataset.take(1):
    plt.imshow(image)
    plt.title(f'Colony Count: {label}')
    plt.axis('off')
    plt.show()
    
    
    
    
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Define the CNN model
def create_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1)  # Output layer with single neuron for regression
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

# Define input image dimensions
input_shape = (256, 256, 3)

# Create the CNN model
model = create_model(input_shape)

# Print model summary
model.summary()

# Split the dataset into training and validation sets
train_size = int(0.8 * len(images))
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

# Configure dataset for performance
train_dataset = train_dataset.batch(8).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(8).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Define early stopping callback
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Train the model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=100, callbacks=[early_stopping])

# Evaluate the model
loss = model.evaluate(val_dataset)
print("Validation Loss:", loss)




# Save the trained model
model.save("colony_count_model.h5")



import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("colony_count_model.h5")




def preprocess_image(image_path):
    img = load_and_preprocess_image(image_path, target_size=(256, 256))
    return img





# Preprocess the image
image_path = r"C:\Users\imsha\Downloads\trialml2.jpg"
preprocessed_image = preprocess_image(image_path)

# Make predictions
prediction = model.predict(tf.expand_dims(preprocessed_image, axis=0))[0][0]
print("Predicted Colony Count:", prediction)

