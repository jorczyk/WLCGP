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

NumPerson = 1  # number of classes
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
            iCellBlock = iCell[k, m]  # k,m
            blockLBPI = wlcgpFile.wlcgp(iCellBlock)
            blockLBPI = np.transpose(blockLBPI)
            # lbpI = np.concatenate((lbpI, blockLBPI))  # LBP_I=[LBP_I,Block_LBP_I]; moze byc concatenate
            if (m == 0) & (k == 0):
                lbpI = blockLBPI
            else:
                lbpI = np.concatenate((lbpI, blockLBPI))  # LBP_I=[LBP_I,Block_LBP_I]; moze byc concatenate
                # END for m in range(numy):
    if (j == 0):
        allsamples = lbpI
    else:
        allsamples = np.concatenate((allsamples, lbpI), -1)

# END for j in range(NumPerClassTrain)
#print allsamples.shape
allsamples = allsamples.reshape((NumPerClassTrain, lbpI.size))  # size ok
#print allsamples.shape

sampleMean = np.mean(allsamples)
nTrain = np.size(allsamples, 0)

xmean = np.zeros((nTrain, allsamples.shape[1]))

# print xmean.shape
i = 0
for i in range(nTrain):
    # while i <= nTrain:
    #     i += 1
    xmean[i, :] = allsamples[i, :] - sampleMean
# end

# print xmean.shape

# PCA
# sigma = xmean * np.transpose(xmean)
sigma = xmean.dot(np.transpose(xmean))

d, v = np.linalg.eig(sigma)  # !!!
# d-eigenvalues(8) v-eigenvectors(8x8)

# matrix D of eigenvalues  matrix V whose columns are the corresponding right eigenvectors --MATLAB

# dTemp = np.zeros((d.size,d.size))

# d1 = np.diag(d) #zmiana v i d byla
# print d.shape
d1 = np.squeeze(np.asarray(d))  # rozmiar chyba ok
# print d1

d2 = d1.sort
index = np.argsort(d1)

#print index
rows, cols = v.shape

vsort = np.zeros((rows,cols))

#print vsort.shape

dsort = [0]*index.size
i = 0
for i in range(cols):
    vsort[:, i] = v[:, index[cols - i -1]] #!!
    dsort[i] = d1[index[cols - i - 1]]

dsum = np.sum(dsort)  # dsum = sum(dsort);
dsumExtract = 0
p = 0

#print dsum

while dsumExtract / dsum < 0.95:
    dsumExtract = np.sum(dsort[0:p]) #(dsort[1:p])
    p = p + 1
    # print "dsumExtract: " + str(dsumExtract/dsum)
    # print p

i = 0
p = nTrain - 1 #nTrain - 1

# print ((vsort[:, i]).shape) #ok
# print (dsort[1]** (-1 / 2)) #ok


base = np.zeros((allsamples.shape[1],p))
while (i <= p) & (dsort[i] > 0):
    base[:, i] = dsort[i] ** (-1 / 2) * np.transpose(xmean).dot(vsort[:, i]) #dsort[i] ** (-1 / 2) * np.transpose(xmean) * vsort[:, i]
    i = i + 1

# print base.shape #ok

allcoor = allsamples.dot(base) #allsamples * base
#print allcoor.shape #ok numperson jest wazne

temp = fisherTrain.fisher(np.transpose(allcoor), NumPerson, NumPerClassTrain) #!!!

# P = temp.P
# E = temp.E
# accu = 0

