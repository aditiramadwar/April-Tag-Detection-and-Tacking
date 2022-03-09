from Utils.tag_utils import process_tag, pre_process, np, cv2
from Utils.utils import vid_name, tag_size, homography, warp, get_P
from Utils.contours_utils import getPapercorners, getTagCorners, prev_paper, prev_tag
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1920, 1080))
cap = cv2.VideoCapture(vid_name)

while(cap.isOpened()):
    ret, one_frame = cap.read()
    if ret == True:
        frame = one_frame.copy()

############################ DETECT TAG ########################################
        edges = pre_process(frame)
        paper, prev_paper, corners_in_img = getPapercorners(edges, prev_paper, 50, 10)
        tag, prev_tag, dist = getTagCorners(paper, prev_tag, corners_in_img)
        if dist:
            frame = prev_image
        else: 
            prev_image = frame

############################### TAG WARP ######################################
        pts_src = np.array([[tag[0][0], tag[0][1]], [tag[1][0], tag[1][1]], [tag[2][0], tag[2][1]], [tag[3][0], tag[3][1]]])
        pts_dst = np.array([[0, 0], [0, tag_size-1], [tag_size-1, tag_size-1], [tag_size-1, 0]])
        # planar, rotated tag
        M = homography(pts_src, pts_dst)
        warped_img = warp(np.uint8(frame), M, (tag_size, tag_size))

####################### TAG DECODE #######################################
        tag_ID, tag_crn = process_tag(np.uint8(warped_img), Rotate = pts_src)
        # print(tag_ID)

######################### PROJECTION MATRIX #############################
        H = homography(pts_dst, tag_crn)
        P = get_P(H)
        h = np.array([-(tag_size-1), -(tag_size-1), -(tag_size-1), -(tag_size-1)]).reshape(-1,1)
        q = np.array([1, 1, 1, 1]).reshape(-1,1)

        cube_cnr = np.concatenate((pts_dst, h), axis = 1)
        cube_cnr = np.concatenate((cube_cnr, q), axis = 1)
        arr = []
        for i in range(4):
            p = P.dot(cube_cnr[i])
            a1 = ((p/p[-1]).astype(int))[0]
            a2 = ((p/p[-1]).astype(int))[1]
            arr.append(np.array([a1, a2]))
        arr = np.array(arr)

###################### DRAW CUBE #################################
        cv2.drawContours(frame, [tag_crn], 0, (0, 255 , 0), 3)
        cv2.drawContours(frame, [arr], 0, (0, 0 , 255), 3)

        for i in range(0, tag_crn.shape[0]):
            cv2.line(frame, (tag_crn[i,0], tag_crn[i,1]), (arr[i,0], arr[i,1]), (0,255,0), 3)
        # out.write(frame)
        cv2.imshow('cube', np.uint8(frame))
        # cv2.imwrite('results/cube_projection.jpg', np.uint8(frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else: 
        break

