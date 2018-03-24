import cv2
import numpy as np
import os

height = 200;
width = 200;

print(os.getcwd())

img = np.zeros((height, width, 3), np.uint8)

img[100, 100] = [255, 255, 255]

size = os.path.getsize('test.7z')


height = int((size/256)+0.5)

img = np.zeros((height, 256, 3), np.uint8)

currentRow = 0
currentColumn = 0;
with open("test.7z", "rb") as f:
    byte = f.read(1)
    
    while byte:
        comparison = int(bytes.hex(byte), 16)
        if comparison >= 0x61 and comparison <= 0x7A or comparison >= 0x41 and comparison <= 0x5A:
            toWrite1 = 0xDA
            toWrite2 = 0xF7
            toWrite3 = 0xA6
        elif comparison >= 0x30 and comparison <= 0x39:
            toWrite1 = 0xFF
            toWrite2 = 0x57
            toWrite3 = 0x33
        elif comparison >= 0x20 and comparison <= 0x7E:
            toWrite1 = 0x85
            toWrite2 = 0xC1
            toWrite3 = 0xE9
        else:
            comparison = comparison >> 1
            toWrite1 = comparison
            toWrite2 = comparison
            toWrite3 = comparison


        #toWrite1 = 0
        #toWrite2 = test
        #toWrite3 = 0
        
        img[currentRow, currentColumn] = [int(str(toWrite1), 16), int(str(toWrite2), 16), int(str(toWrite3), 16)]
        currentColumn = currentColumn + 1
        if currentColumn > 255:
            currentColumn = 0
            currentRow = currentRow + 1
        print("Current Column: " + str(currentColumn))
        print("Current Row: " + str(currentRow))
        byte = f.read(1)
        print(str(byte))

cv2.imwrite("ma123ma.png", img)
        

