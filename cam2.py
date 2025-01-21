import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
import time
import random

# Load the pre-trained model
model = load_model('/Users/macbook/Desktop/university shit/YEAR 4/y4 s3/IDP/trash_classification_model.h5')

# Class labels 
class_labels = {
    0: 'cardboard',
    1: 'e-waste',
    2: 'glass',
    3: 'medical',
    4: 'metal',
    5: 'paper',
    6: 'plastic'
}

# Function to determine final prediction 
def get_final_prediction(predictions, random_value):
    metal_prob = predictions[4]  # Probability for metal (index 4 in class_labels)
    highest_prob = np.max(predictions) # get the highest probability
    highest_class = np.argmax(predictions) # get the class of the highest probability

    if random_value:  # if inductive reading is metal
        if metal_prob >= 0.65: # camera probability of metal is >= 0.65
            return "METAL" # returned class is metal
        elif highest_prob >= 0.85: # highest probability is >= 0.85 
            return class_labels[highest_class] # returned class is class of highest probability
        else:
            return "GENERAL WASTE" 
    else:  # inductive reading is non-metal 
        if highest_class == 4:  # Highest probability class is metal
            if highest_prob >= 0.85:
                return "METAL"
            else:
                return "GENERAL WASTE"
        else:
            if highest_prob >= 0.8:
                return class_labels[highest_class]
            else:
                return "GENERAL WASTE"

# Initialize webcam for real-time classification
cap = cv2.VideoCapture(0)
last_prediction_time = time.time()
last_label = ""  # Store the last prediction label
last_probabilities = None  # Store the last prediction probabilities
random_value = None  # Store the random True/False value
final_prediction = ""  # Store the final prediction based on the logic

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Check if 10 seconds have passed since the last prediction
    current_time = time.time()
    if current_time - last_prediction_time >= 10:
        # Preprocess the frame for EfficientNet
        img = cv2.resize(frame, (224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict the class
        predictions = model.predict(img_array)[0]
        predicted_class = np.argmax(predictions)
        last_label = class_labels[predicted_class]  # Update the last prediction label
        last_probabilities = predictions  # Update the last prediction probabilities
        random_value = random.choice([True, False])  # Generate random True/False value
        final_prediction = get_final_prediction(last_probabilities, random_value)  # final prediction value to give to arduino. !!!!!!!!!
        last_prediction_time = current_time  # Update the last prediction time

    # Display the last label on the frame
    if last_label:
        cv2.putText(frame, f'Prediction: {last_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the probabilistic output on the frame
    if last_probabilities is not None:
        for i, (label, prob) in enumerate(zip(class_labels.values(), last_probabilities)):
            text = f'{label}: {prob:.2f}'
            cv2.putText(frame, text, (10, 60 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    # Display the random True/False value under the probabilities
    if random_value is not None:
        cv2.putText(frame, f'Random Value: {random_value}', (10, 60 + len(class_labels) * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the final prediction on the top right of the screen
    if final_prediction: 
        frame_height, frame_width, _ = frame.shape
        cv2.putText(frame, f'Final Prediction: {final_prediction}', (frame_width - 350, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the frame with the label, probabilities, random value, and final prediction
    cv2.imshow('Trash Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
