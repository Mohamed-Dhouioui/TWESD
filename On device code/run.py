import time
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import argparse

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to TFLite model")
ap.add_argument("-i", "--image", required=True, help="path to test image")
ap.add_argument("-s", "--size", required=True, type=int, help="image size")
args = vars(ap.parse_args())

# Load TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path=args["model"])
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load image
image = Image.open(args["image"])
image = image.convert('RGB')
image = image.resize((args["size"], args["size"]))
if "int" in args["model"]:
    image = np.array(image, dtype=np.uint8)
else:
    image = np.array(image, dtype=np.float32)
    image /= 255.0
image = np.expand_dims(image, axis=0)

# Run inference and time it
start_time = time.time()
interpreter.set_tensor(input_details[0]['index'], image)
interpreter.invoke()
end_time = time.time()

# Get the output tensor and print the result
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_class = output_data.squeeze().argmax(axis = 0)
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
print("Inference result:", class_names[predicted_class])
print("Inference time:", end_time - start_time, "seconds")