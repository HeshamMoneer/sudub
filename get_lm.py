import numpy as np

def closest(A, P):
  A = np.array(A)
  P = np.array(P)
  distances = np.linalg.norm(A-P, axis=1)
  min_index = np.argmin(distances)
  print(f"the closest point is {min_index}")

def getLM(event,x,y,flags,param):  
    if(event == 1):  
      closest(param, (x,y)) 