import cv2
import numpy as np
import os

height = 200;
width = 200;

print(os.getcwd())

img = np.zeros((height, width, 3), np.uint8)

img[100, 100] = [255, 255, 255]

cv2.imwrite("mamamamamamamamamama.png", img)


#with open("test.7z", "rb") as f:
#    byte = f.read(1)
 #   while byte:
  #      print("Test")

