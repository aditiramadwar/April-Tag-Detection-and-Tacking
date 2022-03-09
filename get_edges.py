from Utils.tag_utils import pre_process, process_tag, np, cv2
from Utils.utils import vid_name

cap = cv2.VideoCapture(vid_name)

scale_percent = 60 
width = int(1920 * scale_percent / 100)
height = int(1080 * scale_percent / 100)
dim = (width, height)

ret, one_frame = cap.read()
# 1a get edges
if ret == True:
    frame = one_frame.copy()
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    edges = pre_process(frame)
    # cv2.imwrite('results/1a.Edges.jpg', edges)
    cv2.imshow('Edges', np.uint8(edges))
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

# 1b
tag_img = cv2.imread('data/ar_tag.png')
tag_img = cv2.resize(tag_img, (160, 160))
ID, count = process_tag(tag_img)
print("Tag ID for sample: ", ID)

tag_img = cv2.imread('data/tag.jpg')
tag_img = cv2.resize(tag_img, (160, 160))
ID, count = process_tag(tag_img)
print("Tag ID for video: ", ID)