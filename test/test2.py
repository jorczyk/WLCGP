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

NumPerson = 40  # number of classes
NumPerClass = 10  # number of faces for each class
NumPerClassTrain = 8  # trainging count for each class
NumPerClassTest = NumPerClass - NumPerClassTrain  #

# allsamples = []  #
allsamples = np.empty((0, 0))
numx = 4  # image segmentation settings
numy = 2

# allsamples = np.zeros((C, T))

filePath = filepath + "\s" + str(1) + "\\" + str(1) + ".pgm"  # filepath to read
img = cv2.imread(filePath, 0)
xsize, ysize = img.shape  # get file size xsize - pionowo; ysize - poziomo
img = np.double(img)

# print img[0, 1] # 49 ok
# print np.size(img,1) # 112x92 ok

img = ((img - np.mean(img)) + 128) / np.std(img) * 20  # ok

# print np.size(img,0) # 112x92 ok

iCell = commons.block(numx, numy, img)
# print np.size(iCell,0) #do poprawy kolejnosc ale generalnie ok --chyba ok

lbpI = []  # zmienic na 2d array? albo rzutowac blockLBPI na array
lbpI = np.asarray(lbpI)

j = 0
for j in range(NumPerClassTrain):
    k = 0

    # while k <= numx:
    for k in range(numx):  # numx
        # k += 1
        # m = 0
        # while m <= numy:
        for m in range(numy):  # numy
            # m += 1
            iCellBlock = iCell[k, m]  # k,m
            # iCellBlock = #cell2mat --nie wiem czy potrzebujemy bo ta funkcja jest specyficzna dla matlaba i rozpisuje "cell" na

            blockLBPI = wlcgpFile.wlcgp(iCellBlock)  # !!!!
            # print blockLBPI[:, None].shape
            blockLBPI = np.transpose(blockLBPI)  # transpozycja macierzy blockLBPI
            # lbpI = np.concatenate((lbpI, blockLBPI))  # LBP_I=[LBP_I,Block_LBP_I]; moze byc concatenate
            # print blockLBPI.shape
            # print blockLBPI.shape

            if (m == 0):
                lbpI = blockLBPI
            else:
                lbpI = np.concatenate((lbpI, blockLBPI))  # LBP_I=[LBP_I,Block_LBP_I]; moze byc concatenate

    if (j == 0):
        allsamples = lbpI
    else:
        #print "all: " + str(allsamples.shape)
        #print "lbpi: " + str(lbpI.shape)
        # allsamples = np.stack((allsamples, lbpI), 1)
        allsamples = np.concatenate((allsamples, lbpI), -1)

#print allsamples.shape
allsamples = allsamples.reshape((lbpI.size,NumPerClassTrain))
#print allsamples.shape

sampleMean = np.mean(allsamples)
nTrain = np.size(allsamples, 0)
i = 1

xmean = np.zeros((nTrain, allsamples.shape[1]))

#print xmean.shape
