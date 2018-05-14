import numpy as np
import cv2
import math
import os

def rot(n, x, y, rx, ry):
    if (ry == 0):
        if (rx == 1):
            x = n-1 - x
            y = n-1 - y

        t = x
        x = y
        y = t
    return x, y


path = './ToRender/'

for filename in sorted(os.listdir(path)):
    count = 0
    byteString = []
    with open(path+filename, "rb") as f:
        while(True):
            toStore = f.read(1)
            if(toStore is None or len(toStore) == 0):
                break
            else:
                byteString.append(ord(toStore))
                count = count + 1
            

    entropyList = []

    byteCount = []
    for a in range(0, 256):
        byteCount.append(byteString.count(a))

    byteList = [byteString[0], byteString[1], byteString[2], byteString[3], byteString[4]]
    for a in range(0, len(byteString)):
        toSum = []
        alreadyDone = []
        for b in range(0, len(byteList)):
            toSum.append(float(byteCount[byteList[b]])/len(byteString))
            alreadyDone.append(byteList[b])
    
        #Shannon entropy
        total = 0
        for c in range(0, len(toSum)):
            total = total + toSum[c]*math.log(1.0/toSum[c])
        
        total = -total
        scale = (-510/1.599)*total

        entropyList.append(scale)

        if(a < 5):
            byteList.append(byteString[a+5])
        elif((a + 5) >= len(byteString)):
            byteList.pop(0)
        else:
            byteList.append(byteString[a+5])
            byteList.pop(0)


    batch = 0;

    size = len(entropyList)**(1.0/2)
    size = math.ceil(math.log(size, 2))
    amount = int(2**size)
    img = np.zeros((1, amount, amount, 3), np.float32)

    for a in range(0, len(entropyList)):
        b = a
        x = 0
        y = 0
        rx = 0
        ry = 0

        count = 1
        while count<len(entropyList):
            rx = 1 & (int(b/2))
            ry = 1 & int(int(b) ^ rx)
            x, y = rot(count, x, y, rx, ry)

            x += count * rx
            y += count * ry
            b = b/4;
            count = count*2

        img[batch, x, y, 1] = entropyList[a]

        test = img[batch]
    toWritepath = './EntropyImages/'
    cv2.imwrite(toWritepath+filename+"Entropy"+".png", test)

    batch = 0;

    size = len(entropyList)**(1.0/2)
    size = math.ceil(math.log(size, 2))
    amount = int(2**size)
    img = np.zeros((1, amount, amount, 3), np.float32)

    img = np.zeros((1, amount, amount, 3), np.float32)

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
    toWritepath = 'Images/'
    cv2.imwrite(toWritepath+filename+".png", test)