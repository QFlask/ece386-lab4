"""This script loads a .tflite model into LiteRT and continuously takes pictures with a webcam,
printing if the picture is of a cat or a dog."""

import cv2
from ai_edge_litert.interpreter import Interpreter, SignatureRunner
import sys
import numpy as np
import matplotlib.pyplot as plt


def get_litert_runner(model_path: str) -> SignatureRunner:
    """Opens a .tflite model from path and returns a LiteRT SignatureRunner that can be called for inference

    Args:
        model_path (str): Path to a .tflite model

    Returns:
        SignatureRunner: An AI-Edge LiteRT runner that can be invoked for inference."""

    interpreter = Interpreter(model_path=model_path)
    # Allocate the model in memory. Should always be called before doing inference
    interpreter.allocate_tensors()
    # show the model's signature dict
    print(f"Allocated LiteRT with signatures {interpreter.get_signature_list()}")

    # Create callable object that runs inference based on signatures
    # 'serving_default' is default... but in production should parse from signature
    return interpreter.get_signature_runner()


# TODO: Function to resize picture and then convert picture to numpy for model ingest
def image_to_np(image) -> np.ndarray:
    """Resize and convert image to numpy array"""

    image_array = np.array(image, dtype=np.uint8)  # convert to numpy array
    image_array = cv2.resize(image_array, (150, 150))  # resize to model input shape
    image_array = np.expand_dims(image_array, axis=0)  # add batch dimension

    return image_array


# TODO: Function to conduct inference
def conduct_inference(runner: SignatureRunner, image: np.ndarray) -> tuple[str, float]:
    """Conduct inference on an image"""
    output = runner(catdog_input=image)
    prediction = output["output_0"][0][0]
    if prediction > 0:
        return ("DOG", prediction)
    return ("CAT", abs(prediction))


def main():

    # Verify arguments
    if len(sys.argv) != 2:
        print("Usage: python litert.py <model_path.tflite>")
        exit(1)

    # Create LiteRT SignatureRunner from model path given as argument
    model_path = sys.argv[1]
    runner = get_litert_runner(model_path)
    # Print input and output details of runner
    print(f"Input details:\n{runner.get_input_details()}")
    print(f"Output details:\n{runner.get_output_details()}")

    # Initialize the camera
    cap = cv2.VideoCapture(0)  # 0 is the default camera index

    if not cap.isOpened():
        print("Failed to initialize camera")
        exit(1)

    try:

        while True:

            # Capture a frame
            ret, frame = cap.read()

            # Only process of ret is True
            if ret:
                # Convert BGR (OpenCV default) to RGB for TFLite
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Convert to a NumPy array
                img_array = image_to_np(frame_rgb)

                # Conduct inference
                (prediction, value) = conduct_inference(runner, img_array)

                print(f"Prediction: {prediction}: {value}")

            #                cv2.imshow("Captured Image", frame)
            #                while True:
            #                    if cv2.waitKey(0):
            #                        cv2.destroyAllWindows()
            #                        break

            else:
                print("\nFailed to capture image.\n")

    except Exception as e:
        print(f"Exception occured: {e}")

    except KeyboardInterrupt:
        pass

    finally:
        cap.release()
        cv2.destroyAllWindows()


# Executes when script is called by name
if __name__ == "__main__":
    main()
