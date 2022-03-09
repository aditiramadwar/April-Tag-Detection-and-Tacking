import numpy as np
import cv2

prev_paper=[[0,0],[0,0],[0,0],[0,0]]
prev_tag=[[0,0],[0,0],[0,0],[0,0]]

prev_count = 0
# get the two corner points that are closest to the origin 
def getMaxPts(corners_img):
    y_min = float("inf")
    x_min = float("inf")
    c1 = (0, 0)
    c2 = (0, 0)
    for corners in (corners_img):
        x, y = corners.ravel()
        # top
        if (y_min > y):
            y_min = y
            c1 = (x, y_min)
        # left
        if (x_min > x):
            x_min = x
            c2 = (x_min, y)
    return [c1, c2]

# Apply Shi-Tomasi on the image to get all the corners
# obtain the top left and bottom left corners
def getCorners(img, number, dist):
    tag_points = []
    blurred = cv2.GaussianBlur(img,(3, 3),cv2.BORDER_DEFAULT)
    ret, thres = cv2.threshold(np.uint8(blurred), 50, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(np.uint8(thres),(5,5),cv2.BORDER_DEFAULT)
    corners_img = cv2.goodFeaturesToTrack(np.uint8(blur), number , 0.01, dist)
    if corners_img is not None:
        corners_img = np.int0(corners_img)
        tag_points = getMaxPts(corners_img)
    return tag_points, corners_img
    
# calculate the farthest corner detected in the image for that particular main corner
def get_max_dist(main_point, all_points):
    flag = True
    if(isinstance(all_points, list)):
        flag = False
    max_dist = 0
    max_pt = main_point
    for p in all_points:
        point1 = np.array((main_point[0],main_point[1]))
        if (flag):
            point2 = np.array((p[0][0],p[0][1]))
        else:
            point2 = np.array((p[0],p[1]))
        dist = np.linalg.norm(point1 - point2)
        if max_dist < dist:
            max_dist = dist
            max_pt = p
    if (flag):
        return max_pt[0]
    else:
        return max_pt

# Check if select point is an outlier or not
def is_on_right_side(x, y, xy0, xy1):
    x0, y0 = xy0
    x1, y1 = xy1
    a = float(y1 - y0)
    b = float(x0 - x1)
    c = - a*x0 - b*y0
    return a*x + b*y + c > 5000

def test_point(x, y, vertices):
    num_vert = len(vertices)
    is_right = [is_on_right_side(x, y, vertices[i], vertices[(i + 1) % num_vert]) for i in range(num_vert)]
    all_left = not any(is_right)
    all_right = all(is_right)
    return all_left or all_right

# Get the two corners that are closest to the origin
# then calculate their respective fartherst points to get total 4 corners of the paper
def getPapercorners(edges, prev, number, dist):
    points, corners_in_img = getCorners(edges, number, dist)
    right = get_max_dist(points[1], corners_in_img)
    bottom = get_max_dist(points[0], corners_in_img)
    left = get_max_dist((right[0],right[1]), corners_in_img)
    top = get_max_dist((bottom[0],bottom[1]), corners_in_img)

    dist_rt = np.linalg.norm(right - top)
    dist_bl = np.linalg.norm(bottom - left)
    dist_rb = np.linalg.norm(right - bottom)
    dist_lt = np.linalg.norm(left - top) 

    if dist_rt >= 150 or dist_bl >= 150 or dist_rb >= 90 or dist_lt >= 90:
        prev[3] = right
        prev[2] = bottom
        prev[1] = left
        prev[0] = top
    else:

        right = prev[3]
        bottom = prev[2]
        left = prev[1]
        top = prev[0]
    return [top, left, bottom, right], prev, corners_in_img

# Check which corners lie inside of the paper and get the 4 corners of the tag same as the paper corners
def getTagCorners(paper, prev, corners_in_img):
    flag = False
    if len(corners_in_img)>0:
        corners_in_img = np.int0(corners_in_img)
        tag = []
        top_y = float('inf')
        top_x = 0
        left_x = float('inf')
        left_y = 0
        for corners in corners_in_img:
            x,y = corners.ravel()
            if test_point(x, y, paper):
                tag.append((x,y))
                if top_y > y:
                    top_y = y
                    top_x = x
                if left_x > x:
                    left_x = x
                    left_y = y
        bottom = get_max_dist((top_x, top_y), tag)
        right = get_max_dist((left_x, left_y), tag)
        top = get_max_dist((bottom[0], bottom[1]), tag)
        left = get_max_dist((right[0], right[1]), tag)

        dist_rt = np.linalg.norm(np.array([right[0],right[1]]) - np.array([top[0],top[1]]))
        dist_bl = np.linalg.norm(np.array([bottom[0],bottom[1]]) - np.array([left[0],left[1]])) 
        dist_rb = np.linalg.norm(np.array([right[0],right[1]]) - np.array([bottom[0],bottom[1]]))
        dist_lt = np.linalg.norm(np.array([top[0],top[1]]) - np.array([left[0],left[1]]))
        if dist_rt < 70 or dist_bl < 70 or dist_rb < 70 or dist_lt < 70 or dist_rt == float('inf') or dist_bl == float('inf') or dist_rb == float('inf') or dist_lt == float('inf'):
            right = prev[3]
            bottom = prev[2]
            left = prev[1]
            top = prev[0]
            flag = True
        else: 
            prev[3] = right
            prev[2] = bottom
            prev[1] = left
            prev[0] = top

        return [top, left, bottom, right], prev, flag

    else:
        return prev, prev, flag

