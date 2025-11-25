import cv2
from mtcnn import MTCNN
import os 

detector = MTCNN()

cam = cv2.VideoCapture(0)
count = 0

print("collecting your face images mawa.....")

while count < 40:
    ret, frame = cam.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    res