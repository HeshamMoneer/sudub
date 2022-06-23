import cv2

from dlib_implementation import detectLmsDlib, filterMouthLmsDlib
from draw_lms import drawCircles, drawDelaunayMesh, drawMeshMP
from mediapipe_implementation import detectLmsMP, filterMouthLmsMP


def main():
  cap = cv2.VideoCapture(0)
  while True:
    ret, frame = cap.read()
    if not ret: break
    lms = detectLmsMP(frame)
    lms = filterMouthLmsMP(lms)
    drawMeshMP(frame, lms)
    cv2.imshow('', frame)
    if cv2.waitKey(1) & 0xFF == 27: break


if __name__ == '__main__':
  main()