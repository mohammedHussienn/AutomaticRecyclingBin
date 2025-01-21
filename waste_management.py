from typing import List, Any, TypedDict, Optional, Literal


#State class
class State(TypedDict):
    camera_on: bool
    camera_probability: dict[str, float]
    inductive_reading: bool
    final_prediction: str

class Readings:
    def __init__(self, state: State):
        self.state = state
    
    def open_camera(self):
        #insert camera initialization code here
        #return state with camera on
        pass

    def get_inductive_reading(self):
        #insert inductive reading code here
        #return state with inductive reading
        pass

    def get_camera_prediction(self):
        #insert camera prediction code here
        #return state with camera predictions
        pass


class Algorithm:
    def __init__(self, state: State):
        self.state = state

    def calculate_final_prediction(self):
        #insert final prediction code here
        #return state with final prediction
        pass

    def send_to_arduino(self):
        #insert arduino sending code here
        pass


if __name__ == "__main__":
    state = State()
