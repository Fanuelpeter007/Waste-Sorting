"""Conveyor waste sorter for Raspberry Pi 5.

An IR sensor detects an item at the camera. The conveyor runs, the camera
takes a frame, and a TensorFlow Lite classifier labels it as one of six
classes. After a class-specific delay, the item has travelled to its bin and a
pneumatic actuator pushes it off the belt. Items classified as "trash" stay on
the belt.

Usage:
    python src/sorter.py --model models/model.tflite

Press q in the preview window or Ctrl+C to stop.
"""

import argparse
import time

import cv2
import numpy as np
import tensorflow as tf
from gpiozero import DigitalInputDevice, DigitalOutputDevice

# GPIO pins (BCM numbering)
IR_SENSOR_PIN = 17           # IR obstacle sensor, input (physical pin 11)
CONVEYOR_BELT_PIN = 18       # conveyor motor driver, output (physical pin 12)
PNEUMATIC_ACTUATOR_PIN = 27  # pneumatic valve driver, output (physical pin 13)

# Class order must match the order used when the model was trained.
CLASS_LIST = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Model input size in pixels (height, width).
INPUT_SIZE = (256, 256)

# Time in seconds from classification until the item reaches its bin.
# Each value equals (distance from camera to bin [m]) / (belt speed [m/s]).
# TODO: record the belt speed in m/s and the bin distances in m.
DELAY_MAP_S = {
    'cardboard': 2.88,
    'glass': 4.44,
    'metal': 5.07,
    'paper': 7.53,
    'plastic': 8.81,
    'trash': None,           # no bin: the item stays on the belt
}

# How long the actuator stays open, in seconds.
ACTUATOR_PULSE_S = 0.5

# How long to wait before checking the IR sensor again when the belt is idle.
IDLE_WAIT_S = 10


def parse_args():
    """Read command line options."""
    p = argparse.ArgumentParser(description="Conveyor waste sorter for Raspberry Pi 5.")
    p.add_argument("--model", default="models/model.tflite",
                   help="Path to the TensorFlow Lite classifier (default: models/model.tflite)")
    p.add_argument("--camera", type=int, default=0,
                   help="OpenCV camera index (default: 0)")
    return p.parse_args()


def activate_pneumatic_actuator(actuator):
    """Open the pneumatic valve for one pulse to push the item into its bin."""
    actuator.on()
    time.sleep(ACTUATOR_PULSE_S)  # adjust to the actuator response time
    actuator.off()


def normalize_and_resize(frame):
    """Turn a BGR OpenCV frame into the model input.

    The frame is JPEG-encoded and decoded again with TensorFlow, which yields
    an RGB tensor. It is then resized to INPUT_SIZE and scaled to [0, 1].
    """
    _, buffer = cv2.imencode('.jpg', frame)
    my_img = tf.image.decode_image(buffer.tobytes(), channels=3)
    resized_frame = tf.image.resize(my_img, INPUT_SIZE)
    return resized_frame / 255.0


def classify(interpreter, input_details, output_details, frame):
    """Run the classifier on one frame and return the predicted class name."""
    processed_frame = normalize_and_resize(frame).numpy().astype(np.float32)
    input_tensor = np.expand_dims(processed_frame, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    return CLASS_LIST[int(np.argmax(prediction))]


def main():
    args = parse_args()

    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    ir_sensor = DigitalInputDevice(IR_SENSOR_PIN)
    conveyor_belt = DigitalOutputDevice(CONVEYOR_BELT_PIN)
    pneumatic_actuator = DigitalOutputDevice(PNEUMATIC_ACTUATOR_PIN)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: Failed to open webcam.")
        return 1

    try:
        while True:
            # The IR sensor output is low when an item is in front of it.
            if not ir_sensor.value:
                conveyor_belt.on()

                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame from webcam.")
                    break

                predicted_class = classify(interpreter, input_details, output_details, frame)

                cv2.putText(frame, "Predicted class: " + predicted_class, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Captured Frame', frame)

                # Wait until the item reaches its bin, then push it off.
                # The loop is blocked during this wait, so items must be spaced
                # far enough apart on the belt.
                delay = DELAY_MAP_S.get(predicted_class)
                if delay is not None:
                    time.sleep(delay)
                    activate_pneumatic_actuator(pneumatic_actuator)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                conveyor_belt.off()
                time.sleep(IDLE_WAIT_S)

    except KeyboardInterrupt:
        print("Cleaning up on KeyboardInterrupt")

    finally:
        # Leave the hardware in a safe state however the loop ends.
        conveyor_belt.off()
        pneumatic_actuator.off()
        cap.release()
        cv2.destroyAllWindows()
        print("Stopped: conveyor and actuator off")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
