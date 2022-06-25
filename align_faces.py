import numpy as np
import cv2

from delaunay_triangulation import make_delaunay
from mediapipe_implementation import detectLmsMP, faceRegionMP, filterFaceBoundsLmsMP, filterFaceLmsMP, filterMouthBoundsMP, filterMouthLmsMP
from draw_lms import drawCircles, drawDelaunayMesh
from morph import generate_morph_frame

def align(img1, img2):
  face1, lms1 = faceRegionMP(img1, 1)

  face2, lms2 = faceRegionMP(img2, 2)


  face1, lms1 = squareFit(face1, lms1)
  face1, lms1 = resize(face1, lms1)

  face2, lms2 = squareFit(face2, lms2)
  face2, lms2 = resize(face2, lms2)

  mlms1 = filterMouthLmsMP(lms1)
  mlms2 = filterMouthLmsMP(lms2)

  blms1 = filterMouthBoundsMP(lms1)
  blms2 = filterMouthBoundsMP(lms2)

  ulms1 = mlms1 + blms1
  ulms2 = mlms2 + blms2

  frame = generate_morph_frame(face1, ulms1, ulms2)
  frame = generate_morph_frame(frame, blms2, blms1)
  frame[frame==0] = face1[frame==0]

  return frame

def fom(img1, img2): # Empty trainagles occur in the trianglation, needs speculation
  face1, lms1 = faceRegionMP(img1, 1)

  face2, lms2 = faceRegionMP(img2, 2)


  face1, lms1 = squareFit(face1, lms1)
  face1, lms1 = resize(face1, lms1)

  face2, lms2 = squareFit(face2, lms2)
  face2, lms2 = resize(face2, lms2)

  flms1 = filterFaceLmsMP(lms1)
  flms2 = filterFaceLmsMP(lms2)

  blms1 = filterFaceBoundsLmsMP(lms1)
  blms2 = filterFaceBoundsLmsMP(lms2)

  ulms1 = flms1 + blms1
  ulms2 = flms2 + blms2

  frame = generate_morph_frame(face1, lms1, lms2)
  frame = generate_morph_frame(frame, blms2, blms1)
  frame[frame==0] = face1[frame==0]

  return frame

def resize(img, lms):
  ih, iw, _ = img.shape
  img = cv2.resize(img, (500, 500))
  lms = list(map(lambda tup: (int(tup[0]*500/iw),int(tup[1]*500/ih)), lms))
  return img, lms

def squareFit(img, lms):
  height, width = img.shape[0], img.shape[1]
  if height == width: return img, lms
  dim = max([height, width])
  res = np.zeros((dim, dim, 3), dtype=np.uint8)
  sh = int((dim - height)/2)
  sw = int((dim - width)/2)
  res[sh:height+sh, sw:width+sw] = img
  lms = list(map(lambda tup: (tup[0]+sw,tup[1]+sh), lms))
  return res, lms