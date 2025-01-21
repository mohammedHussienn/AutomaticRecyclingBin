from typing import TypedDict

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
        try:
            # Initialize serial if not already done
            import serial
            ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
            
            # Request reading from Arduino
            ser.write(b'R')
            ser.flush()
            
            # Read response
            response = ser.readline().decode('utf-8').strip()
            
            # Update state
            self.state['inductive_reading'] = bool(int(response))
            
            return self.state
        except Exception as e:
            print(f"Error reading inductive sensor: {e}")
            return self.state

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
        try:
            # Initialize serial if not already done
            import serial
            ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
            
            # Send prediction to Arduino
            if self.state['final_prediction']:
                ser.write(f"{self.state['final_prediction']}\n".encode())
                ser.flush()
        except Exception as e:
            print(f"Error sending to Arduino: {e}")


if __name__ == "__main__":
    # Initialize state with default values
    state = State(
        camera_on=False,
        camera_probability={},
        inductive_reading=False,
        final_prediction=""
    )
    
    import serial
    import time
    
    try:
        ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
        readings = Readings(state)
        algorithm = Algorithm(state)
        
        while True:
            if ser.in_waiting:
                message = ser.readline().decode('utf-8').strip()
                if message == "START":
                    # Motion detected, start the process
                    state = readings.open_camera()
                    state = readings.get_inductive_reading()
                    state = readings.get_camera_prediction()
                    state = algorithm.calculate_final_prediction()
                    algorithm.send_to_arduino()
            
            time.sleep(0.1)  # Small delay to prevent CPU overuse
            
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        if 'ser' in locals():
            ser.close()
