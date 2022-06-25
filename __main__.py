import cv2

from dlib_implementation import detectLmsDlib, faceSquareDlib, filterMouthLmsDlib
from draw_lms import drawCircles, drawDelaunayMesh, drawMeshMP
from mediapipe_implementation import detectLmsMP, faceRegionMP, filterFaceBoundsLmsMP, filterFaceLmsMP, filterMouthBoundsMP, filterMouthLmsMP
from get_lm import getLM
from delaunay_triangulation import make_delaunay
from align_faces import align, fom

def main():
  hesham = cv2.VideoCapture('samples/hesham.mp4')
  michael = cv2.VideoCapture('samples/michael.mp4')
  size = (int(michael.get(3)),int(michael.get(4)))
  fps = michael.get(cv2.CAP_PROP_FPS)
  result = cv2.VideoWriter('dubbed.avi',cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
  NOframes = 0
  while True:
    retM, frameM = michael.read()
    retH, frameH = hesham.read()
    NOframes += 1
    if not retH or not retM: break
    frame = align(frameM, frameH)
    result.write(frame)
    if cv2.waitKey(1) & 0xFF == 27: break
  print(f"processed {NOframes} frames")
  result.release()


if __name__ == '__main__':
  main()