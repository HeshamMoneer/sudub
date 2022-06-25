import cv2
import mediapipe as mp
import numpy as np

from CONSTANTS import FACE_BOUNDS, MOUTH_BOUNDS, MOUTH_LMS_IDS

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

def filterMouthBoundsMP(lms):
  if len(lms):
    return list(np.array(lms)[MOUTH_BOUNDS])
  else: return []

def filterFaceBoundsLmsMP(lms):
  if len(lms):
    return list(np.array(lms)[FACE_BOUNDS])
  else: return []

def faceRegionMP(frame, dictID = 0):
  lms = detectLmsMP(frame, dictID)
  if lms == []: return frame, lms
  x, y = zip(*lms)
  x1,x2,y1,y2 = min(x),max(x),min(y),max(y)
  lms = list(map(lambda tup: (tup[0]-x1,tup[1]-y1), lms))
  return frame[y1:y2,x1:x2], lms
