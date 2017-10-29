# -*- coding: UTF-8 -*-
import os
import re
import cv2
import numpy as np
from scipy import io

def Pedestrians_MIT_pos_img(pos_image_path):
    img_list = os.listdir(pos_image_path)
    firstIn = True
    for imgfilename in img_list:
        image_name = pos_image_path + '/' + imgfilename
        img = cv2.imread(image_name)
        img = img[np.newaxis, :]
        if firstIn == True:
            imgSave = img
            firstIn = False
        else:
            imgSave = np.vstack((imgSave, img))

    return imgSave

def Pedestrians_MIT_neg_img(neg_image_path):
    img_list = os.listdir(neg_image_path)
    firstIn = True
    for imgfilename in img_list:

        image_name = neg_image_path + '/' + imgfilename
        img = cv2.imread(image_name)

        imgResize = cv2.resize(img, (64, 128), cv2.INTER_CUBIC)
        imgResize = imgResize[np.newaxis, :]
        if firstIn == True:
            imgSave = imgResize
            firstIn = False
        else:
            imgSave = np.vstack((imgSave, imgResize))

    return imgSave


def Pedestrians_MIT_pos_labels(nums):
    return np.ones((nums), dtype=np.int)

def Pedestrians_MIT_neg_labels(nums):
    return np.zeros((nums), dtype=np.int)

if __name__ == '__main__':
    pos_image_path = "H:/data/Pedestrains-MIT/pedestrians128x64"
    neg_image_path = "H:/data/INRIA Person Dataset/INRIAPerson/Train/neg"

    pos = Pedestrians_MIT_pos_img(pos_image_path)
    neg = Pedestrians_MIT_neg_img(neg_image_path)

    pos_length = int(pos.shape[0])
    neg_length = int(neg.shape[0])

    pos_labels = Pedestrians_MIT_pos_labels(pos_length)
    neg_labels = Pedestrians_MIT_neg_labels(neg_length)

    data = np.vstack((pos, neg))
    labels = np.append(pos_labels,neg_labels)

    mat_file = {'data':data, 'labels': labels}
    io.savemat('Pedestrians_MIT.mat', mat_file)