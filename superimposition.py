from Utils.tag_utils import process_tag, pre_process, np, cv2
from Utils.utils import vid_name, dim, tag_size, testudo, testudo_crn, homography, warp
from Utils.contours_utils import getPapercorners, getTagCorners, prev_paper, prev_tag
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, dim)
cap = cv2.VideoCapture(vid_name)
# Read until video is completed
while(cap.isOpened()):
    ret, one_frame = cap.read()
    if ret == True:
        frame = one_frame.copy()
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

########################## DETECT TAG ###################################
        edges = pre_process(frame)
        paper, prev_paper, corners_in_img = getPapercorners(edges, prev_paper, 40, 10)
        tag, prev_tag, dist = getTagCorners(paper, prev_tag, corners_in_img)

        if dist:
            frame = prev_image
        else: 
            prev_image = frame
        #'''
########################## TAG WARP #####################################
        pts_src = np.float32([[tag[0][0], tag[0][1]], [tag[1][0], tag[1][1]], [tag[2][0], tag[2][1]], [tag[3][0], tag[3][1]]])
        pts_dst = np.float32([[0, 0], [0, tag_size-1], [tag_size-1, tag_size-1], [tag_size-1, 0]])
        M = homography(pts_src, pts_dst)
        warped_img = warp(np.uint8(frame), M, (tag_size,tag_size))

########################## DECODE TAG ###################################
        tag_ID, tag_crn = process_tag(np.uint8(warped_img), Rotate = pts_src)
        # print(tag_ID)
    
########################## TESTUDO WARP #################################
        test_M = homography(testudo_crn, tag_crn)
        # test = cv2.warpPerspective(np.uint8(testudo), test_M, (image.shape[1],image.shape[0]))
        frame = warp(np.uint8(testudo), test_M, (frame.shape[0],frame.shape[1]), og = frame)
        # out.write(test)
        #'''
        cv2.imshow('Testudo', np.uint8(frame))
        # cv2.imwrite('results/img_warp.jpg', np.uint8(frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  # Break the loop
    else: 
        break

