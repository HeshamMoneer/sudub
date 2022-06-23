import cv2
import dlib

faces_classifier = cv2.CascadeClassifier('models/cc.xml')
predictor = dlib.shape_predictor('models/dlib_predictor.dat')

def detectLmsDlib(img):
  faces = faces_classifier.detectMultiScale(img)
  if len(faces):
    faceRegion = faces[0]
  else: return []

  x,y,w,h = faceRegion
  img = img[y:y+h,x:x+w]

  rect = dlib.rectangle(0, 0, img.shape[1], img.shape[0])
  result = []
  shape = predictor(img, rect)

  for i in range(0,68):
    lmx = shape.part(i).x
    lmy = shape.part(i).y
    result.append((lmx+x, lmy+y))

  return result

def filterMouthLmsDlib(lms):
  return lms[48:68]