import cv2
import numpy as np

from morph import generate_morph_frame
from mediapipe_implementation import detectLmsMP, filterLipsLmsMP, filterMouthBoundsMP, filterMouthLmsMP


# Get closed mouth frame
def cmf(video_path):
  return cf(video_path, 'samples/closed_mouth.jpg')

# get closest frame to passed image
def cf(video_path, image_path, img = []):
  if img == []:
    img = cv2.imread(image_path)
  img = squareFit(img)
  img = cv2.resize(img, (500,500))
  img_lms = detectLmsMP(img, 20)
  img_llms = filterLipsLmsMP(img_lms)
  img_mlms = filterMouthLmsMP(img_lms)
  resulting_frame = []
  resulting_error = -1
  cap = cv2.VideoCapture(video_path)
  while True:
    ret, frame = cap.read()
    if not ret: break
    frame = squareFit(frame)
    frame = cv2.resize(frame, (500,500))
    new_error = calculate_error(img, img_llms, img_mlms, frame)
    if resulting_frame == []:
      resulting_frame = frame
      resulting_error = new_error
    elif new_error < resulting_error:
      resulting_frame = frame
  return resulting_frame


def calculate_error(img, img_llms, img_mlms, frame):
  frame_lms = detectLmsMP(frame, 21)
  frame_mlms = filterMouthLmsMP(frame_lms)
  morph_frame = generate_morph_frame(frame, frame_mlms, img_mlms)
  morph_frame[morph_frame==0] = img[morph_frame==0]
  frame_llms = filterLipsLmsMP(detectLmsMP(morph_frame, 22))
  total_error = 0
  for i in range(len(frame_llms)):
    total_error += abs_dist(frame_llms[i], img_llms[i])
  return total_error

def abs_dist(p1, p2):
  return np.linalg.norm(p1-p2)

def squareFit(img):
  height, width = img.shape[0], img.shape[1]
  if height == width: return img
  dim = max([height, width])
  res = np.zeros((dim, dim, 3), dtype=np.uint8)
  sh = int((dim - height)/2)
  sw = int((dim - width)/2)
  res[sh:height+sh, sw:width+sw] = img
  return res