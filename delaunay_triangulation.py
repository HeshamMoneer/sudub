import cv2

# Check if a point is inside a rectangle
def rect_contains(rect, point):

    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

# Write the delaunay triangles into a file
def draw_delaunay(f_w, f_h, subdiv, dictionary1):

    list4 = []

    triangleList = subdiv.getTriangleList()
    r = (0, 0, f_w, f_h)

    for t in triangleList :
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
            try:
                list4.append((dictionary1[pt1],dictionary1[pt2],dictionary1[pt3]))
            except: pass

    dictionary1 = {}
    return list4

def make_delaunay(shape, narray):

    # Make a rectangle.
    rect = (0, 0, shape[1], shape[0])

    # Create an instance of Subdiv2D.
    subdiv = cv2.Subdiv2D(rect)

    # Make a points list and a searchable dictionary. 
    # narray = narray.tolist()
    points = [(int(x[0]),int(x[1])) for x in narray]
    dictionary = {x[0]:x[1] for x in list(zip(points, range(len(points))))}
    
    # Insert points into subdiv
    for p in points :
        try: subdiv.insert(p)
        except: pass

    # Make a delaunay triangulation list.
    list4 = draw_delaunay(shape[1], shape[0], subdiv, dictionary)
   
    # Return the list.
    return list4
