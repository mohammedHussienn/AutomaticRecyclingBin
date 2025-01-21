/*
Hardware Connections:
--------------------
HC-SR501 PIR Motion Sensor:
- VCC -> 5V
- GND -> GND
- Signal -> Pin 7

Status LED:
- Anode (longer leg) -> Pin 13 through 220Ω resistor
- Cathode (shorter leg) -> GND

LJ12A3-4 Inductive Sensor:
- Brown wire -> 12V
- Blue wire -> GND
- Black wire (Signal) -> Pin 2 (Use voltage divider for 3.3V logic)
  * 10kΩ resistor between signal and ground
  * 20kΩ resistor between signal and sensor output

Voltage Divider Note:
The LJ12A3-4 outputs 12V when detecting metal. Use a voltage 
divider to bring this down to a safe 3.3V for the Arduino input.
*/

const int MOTION_SENSOR_PIN = 7;
const int STATUS_LED_PIN = 13;
const int INDUCTIVE_SENSOR_PIN = 2;

bool motionDetected = false;
String inputString = "";
bool stringComplete = false;

void setup() {
  Serial.begin(9600);
  pinMode(MOTION_SENSOR_PIN, INPUT);
  pinMode(STATUS_LED_PIN, OUTPUT);
  pinMode(INDUCTIVE_SENSOR_PIN, INPUT);
  inputString.reserve(200);
}

void loop() {
  // Check motion sensor
  if (digitalRead(MOTION_SENSOR_PIN) == HIGH && !motionDetected) {
    digitalWrite(STATUS_LED_PIN, HIGH);
    Serial.println("START");
    motionDetected = true;
  } 
  else if (digitalRead(MOTION_SENSOR_PIN) == LOW && motionDetected) {
    digitalWrite(STATUS_LED_PIN, LOW);
    motionDetected = false;
  }

  // Read inductive sensor when requested
  if (Serial.available() > 0) {
    char inChar = (char)Serial.read();
    if (inChar == 'R') {  // 'R' for Read request
      int inductiveValue = digitalRead(INDUCTIVE_SENSOR_PIN);
      Serial.println(inductiveValue);
    }
  }

  // Process incoming prediction from RPi
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n') {
      stringComplete = true;
    } else {
      inputString += inChar;
    }
  }

  if (stringComplete) {
    Serial.print("Received prediction: ");
    Serial.println(inputString);
    inputString = "";
    stringComplete = false;
  }
} 