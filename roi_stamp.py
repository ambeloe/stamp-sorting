import numpy as np
import time
import copy
import cv2

scale_fac = 1
min_segment = 0.10
wo = True


def ctime():
    return int(time.time()*1000)


iin_img = cv2.imread("input.jpg")
iin_img = cv2.resize(iin_img, (int(len(iin_img[0]) / scale_fac), int(len(iin_img) / scale_fac)))

# convert to hsv for poggers thresholding
iin_img_hsv = cv2.cvtColor(iin_img, cv2.COLOR_BGR2HSV)

# threshold image
stime = ctime()
thres = 255 - cv2.inRange(iin_img_hsv, (80, 0, 0), (140, 255, 255))
print("threshold took " + str(ctime() - stime) + "ms")

# find contours
stime = ctime()
cont, hierarchy = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("contour finding took " + str(ctime() - stime) + "ms")
# delete tiny contours
shorts = []
min_seg_len = len(iin_img) * min_segment
for i in range(0, len(cont)):
    if len(cont[i]) < min_seg_len:
        shorts.append(i)
cont = np.delete(cont, shorts)
print(str(len(cont)) + " stamps found")

# draw edges
if not wo:
    stime = ctime()
    edges = cv2.drawContours(copy.deepcopy(iin_img), cont, -1, (0, 0, 255))
    print("draw(copy) took " + str(ctime() - stime) + "ms")

# get cropped stamp images
stamps = []
for i in range(0, len(cont)):
    stime = ctime()
    rect = cv2.minAreaRect(cont[i])
    # draw contour number on image because i cant be bothered to visually match every crop
    if not wo:
        cv2.putText(edges, str(i), (int(rect[0][0]), int(rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))
    # required to get them all oriented consistently because opencv rectangle angles are insane
    retarded = rect[1][1] > rect[1][0]
    if retarded:
        rotmat = cv2.getRotationMatrix2D(rect[0], rect[2], 1)
    else:
        rotmat = cv2.getRotationMatrix2D(rect[0], rect[2] + 90, 1)
    stamp = cv2.warpAffine(copy.deepcopy(iin_img), rotmat, (len(iin_img[0]), len(iin_img)))
    if retarded:
        stamp = cv2.getRectSubPix(stamp, (int(rect[1][0]), int(rect[1][1])), (int(rect[0][0]), int(rect[0][1])))
    else:
        stamp = cv2.getRectSubPix(stamp, (int(rect[1][1]), int(rect[1][0])), (int(rect[0][0]), int(rect[0][1])))
    stamps.append(stamp)
    # for debug
    # print(str(i) + ", " + str(rect[2]))
    print("stamp " + str(i + 1) + " processed in " + str(ctime() - stime) + "ms")

while True:
    for i in range(0, len(stamps)):
        if wo:
            cv2.imwrite("out/" + str(i) + ".png", stamps[i])
        else:
            # cv2.namedWindow(str(i), cv2.WINDOW_NORMAL)
            # cv2.resizeWindow(str(i), 200, 200)
            cv2.imshow(str(i), stamps[i])
    if not wo:
        cv2.imshow("pog", edges)
    # cv2.imshow("thres", thres)
    if cv2.waitKey(10) == 27 or wo:
        break
