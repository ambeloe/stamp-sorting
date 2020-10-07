import numpy as np
import time
import cv2
import os

import stampextract

in_ext = ".jpg"
sdir = "Z:/pogs/stamp/in/"
odir = "Z:/pogs/stamp/out/"

imgs = []
for i in os.scandir(sdir):
    if i.name.endswith(in_ext) and i.is_file:
        imgs.append(i.name)

imnum = 0
problem_images = []
for i in range(0, len(imgs)):
    print("\nprocessing image #" + str(i) + " (" + imgs[i] + ")")
    cimg = cv2.imread(sdir + imgs[i])
    stamps, err = stampextract.getstamps(cimg)
    if err:
        problem_images.append(imgs[i])
    for i in range(0, len(stamps)):
        print("    saving stamp #" + str(imnum))
        cv2.imwrite(odir + str(imnum) + ".png", i)
        imnum += 1

if len(problem_images) > 0:
    print("\nerrors on the following images:")
    for i in problem_images:
        print(i)
