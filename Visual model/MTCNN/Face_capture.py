import sys
import os

sys.path.append(os.pardir)
from importlib import import_module
import cv2
from src.detect import FaceDetector


detector = FaceDetector()


def face_capture(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bounding_boxes = detector.detect(image)
    if (bounding_boxes.numel()):
        i=1
        cropped = frame[int(bounding_boxes[0][1]):int(bounding_boxes[0][3]),
                  int(bounding_boxes[0][0]):int(bounding_boxes[0][2]), :]
    else:
        i=0
        cropped=0
    return i,cropped


