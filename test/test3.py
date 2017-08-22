import sys
import glob
import time
import argparse
import numpy as np
import cv2
import commons
import wlcgpFile
import fisherTrain

filepath = ".\ORL"  # file path to dir with test faces

NumPerson = 3  # number of classes
NumPerClass = 10  # number of faces for each class
NumPerClassTrain = 4  # trainging count for each class
NumPerClassTest = NumPerClass - NumPerClassTrain  #

allsamples = np.empty((0, 0))

numx = 4  # image segmentation settings
numy = 2

for i in range(1, NumPerson + 1):
    for j in range(1, NumPerClassTrain + 1):
        filePath = filepath + "\s" + str(i) + "\\" + str(j) + ".pgm"  # filepath to read
        img = cv2.imread(filePath, 0)
        xsize, ysize = img.shape  # get file size xsize - pionowo; ysize - poziomo
        img = np.double(img)

        img = ((img - np.mean(img)) + 128) / np.std(img) * 20  # ok
        iCell = commons.block(numx, numy, img)
        lbpI = []
        lbpI = np.asarray(lbpI)

        for k in range(numx):  # numx
            for m in range(numy):  # numy
                iCellBlock = iCell[k, m]  # k,m
                blockLBPI = wlcgpFile.wlcgp(iCellBlock)
                blockLBPI = np.transpose(blockLBPI)
                if (m == 0) & (k == 0):
                    lbpI = blockLBPI
                else:
                    lbpI = np.concatenate((lbpI, blockLBPI))  # LBP_I=[LBP_I,Block_LBP_I]; moze byc concatenate
                    # shape lbpI jest dobry do tego momentu
                    # if (i<=1) & (j<=1) :
                    #     print "i: " + str(i) + " j: " + str(j) + " k: " + str(k) + " m: " + str(m) + " lbpI: " + str(lbpI.shape)
        if (j == 1) & (i == 1):
            allsamples = lbpI
        else:
            allsamples = np.concatenate((allsamples, lbpI), -1)

            # if (i<=1) & (j<=1) :
            # print "i: " + str(i) + " j: " + str(j) + " lbpI: " + str(lbpI.shape) + " allsamples: " + str(allsamples.shape)

allsamples = allsamples.reshape((NumPerClassTrain * NumPerson, lbpI.size))  # size ok
# print allsamples.shape
sampleMean = np.mean(allsamples)
nTrain = np.size(allsamples, 0)

xmean = np.zeros((nTrain, allsamples.shape[1]))
for i in range(nTrain):
    xmean[i, :] = allsamples[i, :] - sampleMean
# end

###################################################
# PCA
sigma = xmean.dot(np.transpose(xmean))

d, v = np.linalg.eig(sigma)
# d-eigenvalues(8) v-eigenvectors(8x8)

# matrix D of eigenvalues  matrix V whose columns are the corresponding right eigenvectors --MATLAB

# dTemp = np.zeros((d.size,d.size))

d1 = np.squeeze(np.asarray(d))  # rozmiar chyba ok
d2 = d1.sort
index = np.argsort(d1)

rows, cols = v.shape

vsort = np.zeros((rows, cols))
dsort = [0] * index.size

for i in range(cols):
    vsort[:, i] = v[:, index[cols - i - 1]]  # !!
    dsort[i] = d1[index[cols - i - 1]]

dsum = np.sum(dsort)  # dsum = sum(dsort);
dsumExtract = 0
p = 0

while dsumExtract / dsum < 0.95:
    dsumExtract = np.sum(dsort[0:p])  # (dsort[1:p])
    p = p + 1

i = 0
p = nTrain - 1  # nTrain - 1

# print p

# print ((vsort[:, i]).shape) #ok
# print (dsort[1]** (-1 / 2)) #ok

base = np.zeros((allsamples.shape[1], p))
while (i < p) & (dsort[i] > 0):  # (i <= p)!
    base[:, i] = dsort[i] ** (-1 / 2) * np.transpose(xmean).dot(
        vsort[:, i])  # dsort[i] ** (-1 / 2) * np.transpose(xmean) * vsort[:, i]
    i = i + 1
    # print base.shape
    # print p

allcoor = allsamples.dot(base)
# print allcoor.shape #ok numperson jest wazne

temp = fisherTrain.fisher(np.transpose(allcoor), NumPerson, NumPerClassTrain)  # !!!

P = temp[0]
E = temp[1]
accu = 0
