import numpy as np
import cv2
import math
import os

path = 'C:\Users\Ben\Desktop\Malware\VirusShare_00176\To Render'

listOfRendered = sorted(os.listdir('C:\\Users\\Anna\\Desktop\\Dont touch\\Images\\'))

for filename in sorted(os.listdir(path)):
    if(filename+'.png' in listOfRendered):
        os.remove(path+filename)
    else:
        break
