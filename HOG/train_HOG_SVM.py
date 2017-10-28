#!/usr/bin/env python
# -*- coding:utf-8 -*-

import cv2
import digitals_read_data as readdigitals
import numpy as np
from common import clock, mosaic


class StatModel(object):
    def load(self, fn):
        self.model.load(fn)  # Known bug: https://github.com/opencv/opencv/issues/4969

    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C=12.5, gamma=0.50625):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()


def get_hog():
    winSize = (20, 20)
    blockSize = (10, 10)
    blockStride = (5, 5)
    cellSize = (10, 10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)

    return hog
    affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR


def evaluate_model(model, digits, samples, labels):
    resp = model.predict(samples)
    err = (labels != resp).mean()
    print('Accuracy: %.2f %%' % ((1 - err) * 100))

    confusion = np.zeros((10, 10), np.int32)
    for i, j in zip(labels, resp):
        confusion[int(i), int(j)] += 1
    print('confusion matrix:')
    print(confusion)

    vis = []
    for img, flag in zip(digits, resp == labels):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if not flag:
            img[..., :2] = 0

        vis.append(img)
    return mosaic(25, vis)


if __name__ == '__main__':

    print('Loading digits from digits.png ... ')

    # Load data. 导入手写字体数据集
    digits, labels = readdigitals.load_digits('../image/digits.png')

    print "digits and labels type: ", digits.shape, labels.shape

    print('Shuffle data ... ')

    # Shuffle data 打乱数据的顺序, 打乱前, 数据是有序排列的
    rand = np.random.RandomState(10)
    shuffle = rand.permutation(len(digits))
    digits, labels = digits[shuffle], labels[shuffle]

    print('Deskew images ... ')

    # 抗扭斜变换, 手写体识别特有步骤
    digits_deskewed = list(map(readdigitals.deskew, digits))

    print('Defining HoG parameters ...')

    # HoG feature descriptor 生成 HOG 描述子
    hog = get_hog();


    print('Calculating HoG descriptor for every image ... ')

    # 计算每张图像的 HOG 特征
    hog_descriptors = []
    for img in digits_deskewed:
        hog_descriptors.append(hog.compute(img))
    hog_descriptors = np.squeeze(hog_descriptors)

    print('Spliting data into training (90%) and test set (10%)... ')

    # 按照 90% 和 10% 的比例生成训练集和测试集
    train_n = int(0.9 * len(hog_descriptors))
    digits_train, digits_test = np.split(digits_deskewed, [train_n])
    hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [train_n])
    labels_train, labels_test = np.split(labels, [train_n])

    print('Training SVM model ...')

    # 训练 SVM 分类器
    model = SVM()
    model.train(hog_descriptors_train, labels_train)

    print('Saving SVM model ...')

    # 生成 .dat 特征描述文件
    model.save('digits_svm.dat')

    print('Evaluating model ... ')

    # 评估模型
    vis = evaluate_model(model, digits_test, hog_descriptors_test, labels_test)
    cv2.imwrite("digits-classification.jpg", vis)
    cv2.imshow("Vis", vis)
    cv2.waitKey(0)