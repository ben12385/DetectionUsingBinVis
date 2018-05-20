import numpy as np
import cv2
import math
import os

path = './Test/'
batch = 0

for filename in sorted(os.listdir(path)):
    count = 0
    byteString = []
	
	#Load file into array byteString
    with open(path+filename, "rb") as f:
        while(True):
            toStore = f.read(1)
            if(toStore is None or len(toStore) == 0):
                break
            else:
                byteString.append(ord(toStore))
                count = count + 1
				
        byteCount = []
    for a in range(0, 256):
        byteCount.append(byteString.count(a))
	
	
	#Calculate individual entropy of each byte as well as total entropy
    totalEntropy = 0
    for a in range(0, 256):
        bytePercent = byteCount[a]/len(byteString)
        if(bytePercent > 0):
            byteEntropy = bytePercent*math.log(bytePercent, 2)
            byteCount[a] = byteEntropy
            totalEntropy = totalEntropy + byteEntropy

    #Base image of all 0s with 256xlength
    lengthOfImage = math.ceil(len(byteString)/256)
    img = np.zeros((1, lengthOfImage, 256, 3), np.float32)

    #Prep moving window
    windowSize = 256
    entropy = [];
    for a in range(0, windowSize):
        entropyOfByte = byteCount[byteString[a]]
        entropy.append(entropyOfByte)

    #Create image with both byte and entropy
    for a in range(0, len(byteString)):
        x = a%256
        y = a//256
		
	#Byte Plot
        img[batch, y, x, 0] = byteString[a]
		
        #Entropy Plot
        entropyRatio = totalEntropy/sum(entropy)
        #print(entropyRatio)
        img[batch, y, x, 1] = int(round(entropyRatio*255))
		
	#Update window if updatable
        if(a>=windowSize/2 and len(byteString)-(windowSize/2) > a):
            del entropy[0]
            entropy.append(byteCount[byteString[a+windowSize//2]])
		

    
    #Create the image
    test = img[batch]
    toWritepath = './Test/'
    cv2.imwrite(toWritepath+filename+"Entropy"+".png", test)
