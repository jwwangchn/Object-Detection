# -*- coding: UTF-8 -*-
import os
import re
import cv2
import numpy as np

def INRIA_pos_coordinate(filename):
    if str(".txt") not in filename:
        return 0;

    print "Processing:", filename

    f = open(filename, "r")

    fr = f.readlines()

    for line in fr:
        if str(line).__contains__("size"):
            sizes = []
            sizes = re.findall('\d+', line)
            imgWidth = sizes[0]
            imgHeight = sizes[1]
            imgDepth = sizes[2]
            # print imgWidth, imgHeight, imgDepth

        if str(line).__contains__('Objects'):
            nums = re.findall('\d+', line)
            break
    coordinateList = []
    for index in range(1, int(nums[0]) + 1):
        for line in fr:
            if str(line).__contains__("Bounding box for object " + str(index)):
                coordinate = re.findall('\d+', line)
                xmin = int(coordinate[1])
                ymin = int(coordinate[2])
                xmax = int(coordinate[3])
                ymax = int(coordinate[4])
                coordinateList.append([xmin, ymin, xmax, ymax])
                # print xmin, ymin, xmax, ymax
    f.close()
    return coordinateList, int(nums[0])


def INRIA_pos_img(pos_image_path, annotation_path):
    img_list = os.listdir(annotation_path)

    for imgfilename in img_list[0:2]:
        annotation_name = annotation_path + '/' + imgfilename
        image_name = (pos_image_path + '/' + imgfilename).replace(".txt", "") + ".png"
        img = cv2.imread(image_name)
        coordinates, nums = INRIA_pos_coordinate(annotation_name)
        print coordinates
        for index in range(nums):
            imgCut = img[coordinates[index][1]:coordinates[index][3], coordinates[index][0]:coordinates[index][2]]
            imgResize = cv2.resize(imgCut,(160,96), cv2.INTER_CUBIC)
            print imgResize.





# def INRIA_neg_img(neg_image_path):


if __name__ == '__main__':
    pos_image_path = "H:/data/INRIA Person Dataset/INRIAPerson/Train/pos"
    neg_image_path = "H:/data/INRIA Person Dataset/INRIAPerson/Train/neg"
    annotation_path = "H:/data/INRIA Person Dataset/INRIAPerson/Train/annotations"
    # print INRIA_pos_coordinate(annotation_path)
    INRIA_pos_img(pos_image_path, annotation_path)
