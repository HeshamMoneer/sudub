import cv2
import mediapipe as mp
import numpy as np

from CONSTANTS import FACE_BOUNDS, MOUTH_LMS_IDS

detectors = {}

def detectLmsMP(frame, dictID = 0):
  if not dictID in detectors.keys():
    detectors[dictID] = mp.solutions.face_mesh.FaceMesh()
  faceMesh = detectors[dictID]
  ih, iw, _ = frame.shape
  imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  lms = faceMesh.process(imgRGB)
  if lms.multi_face_landmarks: 
    return list(map(lambda lm: (int(lm.x * iw), int(lm.y * ih)) ,
                    lms.multi_face_landmarks[0].landmark))
  else: return []

def filterMouthLmsMP(lms):
  if len(lms):
    return list(np.array(lms)[MOUTH_LMS_IDS])
  else: return []

def filterFaceBoundsLmsMP(lms):
  if len(lms):
    return list(np.array(lms)[FACE_BOUNDS])
  else: return []

def faceSquareMP(frame, dictID = 0):
  lms = detectLmsMP(frame, dictID)
  if lms == []: return frame
  x, y = zip(*lms)
  return frame[min(y):max(y), min(x): max(x)]
