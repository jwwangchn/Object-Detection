import cv2
import numpy as np

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


num3 = cv2.imread('../image/3.png')
num3_gray = cv2.cvtColor(num3, cv2.COLOR_BGR2GRAY)

imgResize = cv2.resize(num3_gray, (20, 20), cv2.INTER_CUBIC)

hog = get_hog()
hog_descriptors = hog.compute(imgResize)
hog_descriptors = hog_descriptors[np.newaxis,:]

svm = cv2.ml.SVM_load("digits_svm.dat")

print svm.predict(hog_descriptors)[1].ravel()