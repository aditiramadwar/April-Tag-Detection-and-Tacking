import numpy as np
import cv2

vid_name = 'data/1tagvideo.mp4'
testudo_img = 'data/testudo.png'
testudo = cv2.imread(testudo_img)
tag_size = 160
testudo = cv2.resize(testudo, (tag_size, tag_size), interpolation = cv2.INTER_AREA)
testudo_crn = np.float32([[0, 0], [0, tag_size-1], [tag_size-1, tag_size-1], [tag_size-1, 0]])

# resize video frame
scale_percent = 60 # percent of original size
width = int(1920 * scale_percent / 100)
height = int(1080 * scale_percent / 100)
dim = (width, height)

K = np.array([[1346.100595, 0, 932.1633975],
              [0, 1355.933136, 654.8986796],
              [0, 0, 1]])

def eig_sort(val,vec):
	idx = val.argsort()[::-1]   
	val = val[idx]
	vec = vec[:,idx]
	return val,vec

def homography(src, des):    #xs, ys ==> world; xp, yp ==> camera
    A = []
    for i in range(len(src)):
        xs, ys = src[i]
        xp, yp = des[i]
        A1 = np.array([-xs, -ys, -1, 0, 0, 0, xs*xp, ys*xp, xp])
        A.append(A1)
        A2 = np.array([0, 0, 0, -xs, -ys, -1, xs*yp, ys*yp, yp])
        A.append(A2)
    A = np.array(A)

    ATA = np.dot(A.transpose(), A)
    eigen_values_ata, eigen_vectors_ata = np.linalg.eig(ATA)
    eigen_values_ata, eigen_vectors_ata = eig_sort(eigen_values_ata, eigen_vectors_ata)
    VT = eigen_vectors_ata.transpose()
    L = VT[-1,:] / VT[-1,-1]
    H = L.reshape(3, 3)
    return H

def warp(image, H, dest_size, og = None):
    # Inverse warping
    if (og is None):
        h, w = np.indices((dest_size[0], dest_size[1]))
        des_coords = np.stack((w.ravel(), h.ravel(), np.ones(h.size)))

        inv_H = np.linalg.inv(H)
        cam_coords = inv_H.dot(des_coords)
        cam_coords /= cam_coords[2,:]
        cam_x, cam_y = cam_coords[:2,:].astype(int)
        cam_x[cam_x >=  image.shape[1]] = image.shape[1]-1
        cam_y[cam_y >=  image.shape[0]] = image.shape[0]-1
        cam_x[cam_x < 0] = 0
        cam_y[cam_y < 0] = 0
        image_transformed = np.zeros((dest_size[0], dest_size[1], 3))
        image_transformed[h.ravel(), w.ravel(), :] = image[cam_y, cam_x, :]
    else:
        # Forward Warping
        image_transformed = og.copy()
        h, w = np.indices((image.shape[0], image.shape[1])) 
        src_coords = np.stack((w.ravel(), h.ravel(), np.ones(h.size)))

        dest_coords = H.dot(src_coords)
        dest_coords /= (dest_coords[2, :])
        dest_x, dest_y = dest_coords[:2,:].astype(int)

        dest_x[dest_x >= dest_size[1]] = dest_size[1] - 1
        dest_y[dest_y >= dest_size[0]] = dest_size[0] - 1
        dest_x[dest_x < 0] = 0
        dest_y[dest_y < 0] = 0
        image_transformed[dest_y, dest_x, :] = image[h.ravel(), w.ravel(), :]

    return image_transformed


# take homography of the tag and then derive the R matrix and t vector
# B = K_inv*H
# B = lamda*[b1,b2,b3]
# r1 = lamda*b1
# r2 = lamda*b2
# r3 = r1 x r2
# t = lamda*b3
# B = [r1,r2,t]

# take B with a positive determinant to get the object located in front of the camera
# r3 = r1 x r2
# lamda is the average length of r1 adn r2
# lamda=()
# 
# x_c = P * x_w
# x_w = [xw, yw, 0, 1]
# P = K * [R|t]
# Read until video is completed

def get_P(H):
    B_tilda = np.linalg.inv(K).dot(H)
    # print(B_t)
    # check determinant to choose side of point
    B_t_det = np.linalg.det(B_tilda)
    # print(B_t_det)
    if B_t_det < 0:
        B = B_tilda*(-1)
    else:
        B = B_tilda
    # print(B)
    # B=[b1,b2,b3]
    b1 = B[:, 0]
    b2 = B[:, 1]
    b3 = B[:, 2]

    # find lamda
    # h1 = H[0:]
    # h2 = H[1:]
    lamda = 2/((np.linalg.norm(b1) + np.linalg.norm(b2)))
    # print(lamda)
    r1 = lamda * b1
    r2 = lamda * b2
    r3 = np.cross(r1, r2)
    t = lamda * b3
    # print("t",t)
    rot_trans = np.array([r1,r2,r3,t]).T
    # P = K* [R|t]
    P = K.dot(rot_trans)
    P = P / P[-1,-1]
    return P