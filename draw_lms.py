import cv2

def drawCircles(frame, lms):
  for x,y in lms:
    cv2.circle(frame, (x,y), 1, (0,0,255), 1)