import cv2
import dlib

from delaunay_triangulation import make_delaunay
from morph import generate_morph_frame

faces_classifier = cv2.CascadeClassifier('models/cc.xml')
predictor = dlib.shape_predictor('models/dlib_predictor.dat')

def faceRegion(img):
  x, y, w, h = faces_classifier.detectMultiScale(img)[0]
  return img[y : y + h, x : x + w]

def landmarks(faceRegion):
  rect = dlib.rectangle(0, 0, faceRegion.shape[1], faceRegion.shape[0])
  result = []
  shape = predictor(faceRegion, rect)
  minX, minY = 999, 999
  maxX, maxY = -1, -1

  for i in range(49,68):
    x = shape.part(i).x
    y = shape.part(i).y
    result.append((x, y))

    minX = min(minX, x)
    minY = min(minY, y)

    maxX = max(maxX, x)
    maxY = max(maxY, y)


  result.append((minX, minY))
  result.append((maxX, minY))
  result.append((minX, maxY))
  result.append((maxX, maxY))
  return result


closed_mouth = faceRegion(cv2.imread('samples/closed_mouth.jpg'))
closed_mouth = cv2.resize(closed_mouth, (500,500))
closed_mouth_lms = landmarks(closed_mouth)

def main():
  img = cv2.imread('samples/img0.jpg')
  img = faceRegion(img)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = cv2.resize(img, (500,500))
  lms = landmarks(img)
  tri_list = make_delaunay(img.shape, lms)
  morphed = generate_morph_frame(img, lms, closed_mouth_lms, tri_list, 1)
  img[lms[19][1]:lms[22][1] , lms[19][0]:lms[22][0]] = 0
  cv2.imshow('', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()