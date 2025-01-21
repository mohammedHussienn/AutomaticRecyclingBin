from typing import List, Any, TypedDict, Optional, Literal
import cv2 
import numpy as np
import tensorflow as tf  

# State class
class State(TypedDict):
    camera_on: bool
    camera_probability: dict[str, float]
    inductive_reading: bool
    final_prediction: str

class Readings:
    def __init__(self, state: State):
        self.state = state

    def open_camera(self):

        cap = cv2.VideoCapture(0) # to open camera, 0 is default 

        if cap.isOpened():
            print("Camera successfully opened.") # debug
            self.state["camera_on"] = True 
            return cap  # Return the camera object to be used later in other functions 
        
        else:
            print("Failed to open the camera.") # debug
            self.state["camera_on"] = False  # should add later in main --> general waste if camera_on is false
            return None 

    def get_inductive_reading(self):
        # insert inductive reading code here
        # return state with inductive reading
        pass

    def get_camera_prediction(self, cap, model_path: str):
        
        if not self.state["camera_on"]: 
            print("Camera is not on. Cannot capture images.") # debug
            return self.state # again, should add later in main --> general waste if camera_on is false

        if cap is None: 
            print("Invalid camera object. Cannot capture images.") # debug
            return self.state 

        photos = [] # to store the 10 photos

        for i in range(10): # get 10 photos
            ret, frame = cap.read() # to take the photos
            if not ret: 
                print(f"Failed to capture image {i+1}.") # debug
                continue # if 1 or 2 frames are missing, just continue with what we have

            # Resize and preprocess the frame 
            frame = cv2.resize(frame, (224, 224))  # resize for our model needs
            frame = frame / 255.0  # Normalize pixel values to 0-1 
            photos.append(frame) # add processed frames

        cap.release() # close camera
        print("Camera closed after capturing images.") # debug
 
        if not photos: 
            print("No images captured. Returning unchanged state.") # debug
            return self.state

        # Load the model
        print("Loading AI model...") # debug
        model = tf.keras.models.load_model(model_path) # in main, pass the model path to the function 

        # Get predictions for all photos
        photos = np.array(photos)
        predictions = model.predict(photos)  

        # Calculate average probabilities
        avg_probabilities = np.mean(predictions, axis=0)

        # Update the camera_probability dictionary
        classes = list(self.state["camera_probability"].keys())
        self.state["camera_probability"] = {cls: avg_probabilities[i] for i, cls in enumerate(classes)}

        return self.state

class Algorithm:
    def __init__(self, state: State):
        self.state = state

    def calculate_final_prediction(self):
        
        # load needed variables for logic handling
        predictions = self.state["camera_probability"]
        inductive_reading = self.state["inductive_reading"]

        # Identify the class with the highest probability
        highest_class = max(predictions, key=predictions.get)
        highest_prob = predictions[highest_class]
        metal_prob = predictions["metal"]

        # logic handling, The Algorithm
        if inductive_reading and metal_prob >= 0.65:
            self.state["final_prediction"] = "metal"
        elif not inductive_reading and highest_class == "metal" and highest_prob < 0.80:
            self.state["final_prediction"] = "general waste"
        elif highest_prob >= (0.80 if inductive_reading else 0.75):
            self.state["final_prediction"] = highest_class
        else:
            self.state["final_prediction"] = "general waste"

        return self.state

    def send_to_arduino(self):
        # insert arduino sending code here
        pass

if __name__ == "__main__":
    classes = ["cardboard", "e-waste", "glass", "medical", "metal", "paper", "plastic"]
    camera_probability = {cls: 0.0 for cls in classes}
    
    state = State(
        camera_on=False,
        camera_probability=camera_probability,
        inductive_reading=False,
        final_prediction=""
    )
