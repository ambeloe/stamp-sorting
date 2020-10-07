import numpy as np
import time
import copy
import cv2

min_segment = 0.10
wo = True


def ctime():
    return int(time.time() * 1000)


def getstamps(iin_img):
    error = False

    # convert to hsv for poggers thresholding
    iin_img_hsv = cv2.cvtColor(iin_img, cv2.COLOR_BGR2HSV)

    # threshold image
    thres = 255 - cv2.inRange(iin_img_hsv, (80, 0, 0), (140, 255, 255))

    # find contours
    cont, hierarchy = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # delete tiny contours
    shorts = []
    min_seg_len = len(iin_img) * min_segment
    for i in range(0, len(cont)):
        if len(cont[i]) < min_seg_len:
            shorts.append(i)
    cont = np.delete(cont, shorts)
    # print(str(len(cont)) + " stamps found")

    # draw edges
    edges = cv2.drawContours(copy.deepcopy(iin_img), cont, -1, (0, 0, 255), thickness=5)

    # get cropped stamp images
    stamps = []
    for i in range(0, len(cont)):
        # stime = ctime()
        rect = cv2.minAreaRect(cont[i])
        # draw contour number on image because i cant be bothered to visually match every crop
        cv2.putText(edges, str(i), (int(rect[0][0]), int(rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, int(len(iin_img)*0.001), (0, 0, 255), thickness=5)
        # required to get them all oriented consistently because opencv rectangle angles are insane
        why = rect[1][1] > rect[1][0]
        if why:
            rotmat = cv2.getRotationMatrix2D(rect[0], rect[2], 1)
        else:
            rotmat = cv2.getRotationMatrix2D(rect[0], rect[2] + 90, 1)
        stamp = cv2.warpAffine(copy.deepcopy(iin_img), rotmat, (len(iin_img[0]), len(iin_img)))
        if why:
            stamp = cv2.getRectSubPix(stamp, (int(rect[1][0]), int(rect[1][1])), (int(rect[0][0]), int(rect[0][1])))
        else:
            stamp = cv2.getRectSubPix(stamp, (int(rect[1][1]), int(rect[1][0])), (int(rect[0][0]), int(rect[0][1])))
        stamps.append(stamp)
        # print("stamp " + str(i + 1) + " processed in " + str(ctime() - stime) + "ms")

        sedges = cv2.resize(edges, (int(len(edges[0])/5), int(len(edges)/5)))
        # sstamps = []
        # for i in stamps:
        #     sstamps.append(cv2.resize(i, (int(len(i[0])/5), int(len(i)/5))))
    while True:
        # for i in range(0, len(stamps)):
            # cv2.imshow(str(i), sstamps[i])
        cv2.imshow("edges", sedges)
        key = cv2.waitKey()
        if key == 27:
            error = True
            break
        elif key == 32:
            break
        else:
            pass
    return stamps, error
