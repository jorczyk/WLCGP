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

NumPerson = 4  # number of classes
NumPerClass = 10  # number of faces for each class
NumPerClassTrain = 8  # trainging count for each class
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

# print E.shape

####################################################

# Testing

div = 0

for i in range(1, NumPerson + 1):  # 1
    for j in range(NumPerClassTrain + 1, NumPerClass + 1):  # 1(1, NumPerClassTest + 1):
        pathname = filepath + "\\s" + str(i) + "\\" + str(j) + ".pgm"

        img2 = cv2.imread(pathname, 0)
        img2 = ((img2 - np.mean(img2)) + 128) / np.std(img2) * 20
        iiCell = commons.block(numx, numy, img2)
        lbpII = []
        lbpII = np.asarray(lbpII)

        for k in range(numx):
            for m in range(numy):
                iiCellBlock = iiCell[k, m]  # k,m
                blockLBPII = wlcgpFile.wlcgp(iiCellBlock)
                blockLBPII = np.transpose(blockLBPII)
                if (m == 0) & (k == 0):
                    lbpII = blockLBPII
                else:
                    lbpII = np.concatenate((lbpII, blockLBPII))  # LBP_I=[LBP_I,Block_LBP_I]; moze byc concatenate
                    # end
        # end
        lbpII = lbpII.reshape(1, lbpII.shape[0])
        tcoor = lbpII.dot(base)  # lbpII * base// base trzeba tu liczyc!!! //rozmiar ok

        # print np.transpose(E).shape #ok
        # print np.transpose(tcoor).shape #ok

        tcoor = np.transpose(E).dot(np.transpose(tcoor))  # E trzeba bedzie tu liczyc? ale z czego?

        # print tcoor.shape #ok
        # k = 1
        # mdist = []
        # while k <= nTrain:  # co z tym nTrain???
        #     k += 1
        mdist = [None] * nTrain
        for k in range(nTrain):
            mdist[k] = np.linalg.norm(tcoor - P[:, k]) #P trzeba wyciagac
        # end

        ####################################

        # 3 NN algorithm
        # dist, index2 = np.sort(mdist)
        index2 = np.argsort(mdist)
        dist = mdist.sort

        class1 = int(np.math.floor(index2[1] / NumPerClassTrain - 0.1) + 1)  # NumPerClassTrain - skad to ma byc potem brane?
        class2 = int(np.math.floor(index2[2] / NumPerClassTrain - 0.1) + 1)
        class3 = int(np.math.floor(index2[3] / NumPerClassTrain - 0.1) + 1)

        if (class1 != class2) & (class2 != class3):
            result = class1
        else:
            if class1 == class2:
                result = class1
            if class2 == class3:
                result = class2
        # end

        div = div + 1

        print "Result: " + str(result) + " Actual: " + str(i)
        if result == i:
            accu = accu + 1
print accu
print div
accuracyRate = float(accu)/float(div)

print "Accuracy: " + str(accuracyRate)
