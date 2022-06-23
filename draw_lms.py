import cv2

from delaunay_triangulation import make_delaunay

from CONSTANTS import FACEMESH, LIPSMESH_ABS

def drawCircles(frame, lms):
  if lms == []: return
  for x,y in lms:
    cv2.circle(frame, (x,y), 1, (0,0,255), 1)

def drawDelaunayMesh(frame, lms):
  if lms == []: return
  tri = make_delaunay(frame.shape, lms)
  for p1, p2, p3 in tri:
    cv2.line(frame, lms[p1], lms[p2], (0,0,255), 1)
    cv2.line(frame, lms[p3], lms[p2], (0,0,255), 1)
    cv2.line(frame, lms[p1], lms[p3], (0,0,255), 1)

def drawMeshMP(frame, lms):
  if lms == []: return
  mesh = []
  if len(lms) == 468:
    mesh = FACEMESH
  else: mesh = LIPSMESH_ABS
  for p1, p2 in mesh:
    cv2.line(frame, lms[p1], lms[p2], (0,0,255), 1)