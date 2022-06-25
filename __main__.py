from align_faces import align, fom
import cv2

from dlib_implementation import detectLmsDlib, faceSquareDlib, filterMouthLmsDlib
from draw_lms import drawCircles, drawDelaunayMesh, drawMeshMP
from mediapipe_implementation import detectLmsMP, faceRegionMP, filterFaceBoundsLmsMP, filterFaceLmsMP, filterMouthLmsMP
from get_lm import getLM
from delaunay_triangulation import make_delaunay

def main():
  cap = cv2.VideoCapture(0)
  img = cv2.imread('samples/img1.jpg')
  # cv2.namedWindow('image')
  while True:
    ret, frame = cap.read()
    if not ret: break
    frame = align(img, frame)
    # lms = detectLmsMP(frame)
    # cv2.setMouseCallback('image', getLM, lms)
    # lms = filterFaceLmsMP(lms)
    # drawDelaunayMesh(frame, lms)
    # drawCircles(frame, lms)
    cv2.imshow('image', frame)
    if cv2.waitKey(1) & 0xFF == 27: break


if __name__ == '__main__':
  main()