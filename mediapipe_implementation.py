import cv2
import mediapipe as mp
import numpy as np

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()

mouth_lms_ids = [0,11,12,13,14,15,16,37,38,39,40,41,42,61,62,72,73,74,76,77,78,80,81,82,84,85,86,87,88,89,90,91,95,96,146,178,179,180,181,183,184,185,191,267,268,269,270,271,272,291,292,302,303,304,306,307,308,310,311,312,314,315,316,317,318,319,320,321,324,325,375,402,403,404,405,407,408,409,415]

def detectLms(frame):
  ih, iw, _ = frame.shape
  imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  lms = faceMesh.process(imgRGB)
  if lms.multi_face_landmarks: 
    return list(map(lambda lm: (int(lm.x * iw), int(lm.y * ih)) ,
                    lms.multi_face_landmarks[0].landmark))
  else: return []

def filterMouthLms(lms):
  if len(lms):
    return list(np.array(lms)[mouth_lms_ids])
  else: return []

def drawLms(frame, lms):
  for x,y in lms:
    cv2.circle(frame, (x,y), 1, 255, 1)


def main():
  cap = cv2.VideoCapture(0)
  while True:
    ret, frame = cap.read()
    if not ret: return
    lms = filterMouthLms(detectLms(frame))
    drawLms(frame, lms)
    cv2.imshow('MediaPipe',frame)
    if cv2.waitKey(1) & 0xFF == 27: break
  


if __name__ == '__main__':
  main()