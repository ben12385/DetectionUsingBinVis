import numpy as np

with open("test.7z", "rb") as f:
    byteString = f.read()

height = 256;
width = 256;
depth = 256;

img = np.zeros((height, width, depth, 1), np.uint8)

     
iterations = (int)(len(byteString)/3)
for x in range(0, iterations):
    currentRow = byteString[x*3]
    currentColumn = byteString[x*3+1]
    currentDepth = byteString[x*3+2]
    img[currentRow, currentColumn, currentDepth] = img[currentRow, currentColumn, currentDepth] + 1

print("Completed")