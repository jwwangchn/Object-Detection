import cv2
import numpy as np

def get_hog():
    winSize = (160, 96)
    blockSize = (80, 48)
    blockStride = (40, 24)
    cellSize = (80, 48)
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


pos = cv2.imread('H:/data/INRIA Person Dataset/INRIAPerson/Train/pos/crop_000010.png')
neg = cv2.imread('H:/data/INRIA Person Dataset/INRIAPerson/Train/neg/00000011a.png')
imgResize = cv2.resize(pos, (160, 96), cv2.INTER_CUBIC)

hog = get_hog()
hog_descriptors = hog.compute(imgResize)
hog_descriptors = hog_descriptors[np.newaxis,:]

svm = cv2.ml.SVM_load("INRIA.dat")

print svm.predict(hog_descriptors)[1].ravel()