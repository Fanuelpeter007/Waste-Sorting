import os
import cv2
import numpy as np
import tensorflow as tf
from gpiozero import DigitalInputDevice, DigitalOutputDevice
import time

# GPIO pins
IR_SENSOR_PIN = 17
CONVEYOR_BELT_PIN = 18
PNEUMATIC_ACTUATOR_PIN = 27

# Initialize GPIO
ir_sensor = DigitalInputDevice(IR_SENSOR_PIN)
conveyor_belt = DigitalOutputDevice(CONVEYOR_BELT_PIN)
pneumatic_actuator = DigitalOutputDevice(PNEUMATIC_ACTUATOR_PIN)

# Function to start conveyor belt
def start_conveyor_belt():
    conveyor_belt.on()

# Function to stop conveyor belt
def stop_conveyor_belt():
    conveyor_belt.off()

# Function to activate pneumatic actuator
def activate_pneumatic_actuator():
    pneumatic_actuator.on()
    time.sleep(0.5)  # Adjust time as per actuator response time
    pneumatic_actuator.off()

# Load the TensorFlow Lite model
tflite_model_path = 'C:\\Users\\H00422001\\Desktop\\FINAL_MODEL\\model.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# List of all the classes
class_list = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def normalize_and_resize(frame):
    # Convert the OpenCV frame to bytes
    _, buffer = cv2.imencode('.jpg', frame)
    bytes_frame = buffer.tobytes()

    # Decode the bytes into a TensorFlow tensor
    my_img = tf.image.decode_image(bytes_frame, channels=3)

    # Resize the image
    resized_frame = tf.image.resize(my_img, [256, 256])

    # Normalize data
    normalized_frame = resized_frame / 255.0

    return normalized_frame

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Failed to open webcam.")
else:
    try:
        while True:
            if not ir_sensor.value:
                start_conveyor_belt()

                # Capture frame-by-frame
                ret, frame = cap.read()

                # Check if the frame is captured successfully
                if not ret:
                    print("Error: Failed to capture frame from webcam.")
                    break

                # Preprocess the captured frame
                processed_frame = normalize_and_resize(frame).numpy().astype(np.float32)

                # Prepare input tensor
                input_tensor = np.expand_dims(processed_frame, axis=0)

                # Set the tensor to point to the input data to be inferred
                interpreter.set_tensor(input_details[0]['index'], input_tensor)

                # Run the interpreter
                interpreter.invoke()

                # Get the prediction result
                prediction = interpreter.get_tensor(output_details[0]['index'])

                # Get the predicted class name
                predicted_index = np.argmax(prediction)
                predicted_class = class_list[predicted_index]

                # Display the predicted class on the frame
                cv2.putText(frame, "Predicted class: " + predicted_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Captured Frame', frame)

                # Determine delay based on predicted class
                delay_map = {
                    'cardboard': 2.88,
                    'glass': 4.44,
                    'metal': 5.07,
                    'paper': 7.53,
                    'plastic': 8.81,
                    'trash': None
                }
                delay = delay_map.get(predicted_class, None)

                # Activate pneumatic actuator after the delay
                if delay is not None:
                    time.sleep(delay)
                    activate_pneumatic_actuator()

                # Break the loop when 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                stop_conveyor_belt()
                time.sleep(10)  # Wait 10 seconds before checking sensor again

    except KeyboardInterrupt:
        print("Cleaning up on KeyboardInterrupt")

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

print("Cleaning up on normal exit")
