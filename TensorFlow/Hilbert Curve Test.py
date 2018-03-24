import numpy as np
import cv2
import math

def rot(n, x, y, rx, ry):
    if (ry == 0):
        if (rx == 1):
            x = n-1 - x
            y = n-1 - y

        t = x
        x = y
        y = t
    return x, y


#Generate 3D Model
import os
path = 'C:\\Users\\Ben\\Desktop\\Malware\\VirusShare_00176\\test\\'

for filename in os.listdir(path):
    with open(path+filename, "rb") as f:
        byteString = f.read()

    batch = 0;

    size = len(byteString)**(1/2)
    size = math.ceil(math.log(size, 2))
    size = 2**size

    img = np.zeros((1, size, size, 3), np.float32)

    for a in range(0, len(byteString)):
        b = a
        x = 0
        y = 0
        rx = 0
        ry = 0

        count = 1
        while count<len(byteString):
            rx = 1 & (int(b/2))
            ry = 1 & int(int(b) ^ rx)
            x, y = rot(count, x, y, rx, ry)

            x += count * rx
            y += count * ry
            b = b/4;
            count = count*2

        img[batch, x, y, 1] = byteString[a]

        test = img[batch]
    toWritepath = 'C:\\Users\\Ben\\Desktop\\Malware\\VirusShare_00176\\images\\'
    cv2.imwrite(toWritepath+filename+".png", test)

