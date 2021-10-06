import tensorflow as tf
import numpy as np 
import cv2

image_path = "input_image.jpeg"
image = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image)

input_image = tf.expand_dims(image, axis=0)
input_image = tf.image.resize_with_pad(input_image, 192, 192)

model_path = "movenet_lightning_fp16.tflite"
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()

input_image = tf.cast(input_image, dtype=tf.uint8)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
interpreter.invoke()
keypoints = interpreter.get_tensor(output_details[0]['index'])

width = 640
height = 640

KEYPOINT_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7),
    (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16)]

input_image = tf.expand_dims(image, axis=0)
input_image = tf.image.resize_with_pad(input_image, width, height)
input_image = tf.cast(input_image, dtype=tf.uint8)

image_np = np.squeeze(input_image.numpy(), axis=0)
image_np = cv2.resize(image_np, (width, height))
image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

for keypoint in keypoints[0][0]:
    x = int(keypoint[1] * width)
    y = int(keypoint[0] * height)

    cv2.circle(image_np, (x, y), 4, (0, 0, 255), -1)

for edge in KEYPOINT_EDGES:
    
    x1 = int(keypoints[0][0][edge[0]][1] * width)
    y1 = int(keypoints[0][0][edge[0]][0] * height)

    x2 = int(keypoints[0][0][edge[1]][1] * width)
    y2 = int(keypoints[0][0][edge[1]][0] * height)

    cv2.line(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("pose estimation", image_np)
cv2.waitKey()
