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

allsamples = []  #

numx = 4  # image segmentation settings
numy = 2

filePath = filepath + "\s" + str(1) + "\\" + str(1) + ".pgm"  # filepath to read
img = cv2.imread(filePath, 0)
xsize, ysize = img.shape  # get file size xsize - pionowo; ysize - poziomo
img = np.double(img)

img = ((img - np.mean(img))+128)/np.std(img) * 20 # ok


iCell = commons.block(numx, numy, img)

lbpI = []
k = 0

# while k <= numx:
for k in range(1): #numx
    #k += 1
    #m = 0
    #while m <= numy:
    for m in range(1): #numy
        #m += 1
        iCellBlock = iCell[k,m] #k,m
        # iCellBlock = #cell2mat --nie wiem czy potrzebujemy bo ta funkcja jest specyficzna dla matlaba i rozpisuje "cell" na

        blockLBPI = wlcgpFile.wlcgp(iCellBlock) #!!!!
        blockLBPI = np.transpose(blockLBPI)  # transpozycja macierzy blockLBPI
        lbpI = np.concatenate((lbpI, blockLBPI))  # LBP_I=[LBP_I,Block_LBP_I];

        allsamples = np.concatenate((allsamples, lbpI))

sampleMean = np.mean(allsamples)
#nTrain = np.size(allsamples, 1)
i = 1

xmean = np.zeros(nTrain, allsamples.size)

while i <= nTrain:
    i += 1
    xmean[i, :] = allsamples[i, :] - sampleMean
# end
# END OF TRAINING -- w treningu moznaby wywolywac funkcje z matlaba ew.

###################################################

# PCA

sigma = np.multiply(xmean, np.transpose(xmean))
v, d = np.linalg.eig(sigma)
d1 = np.diag(d)
d2, index = np.sort(d1)
rows, cols = v.shape

vsort = []
dsort = []
i = 1
while i < cols:
    i += 1
    vsort[:, i] = v[:, index(cols - i + 1)]
    dsort[i] = d1[index(cols - i + 1)]

dsum = np.sum(dsort)  # dsum = sum(dsort);
dsumExtract = 0
p = 0

while dsumExtract / dsum < 0.95:
    p = p + 1
    dsumExtract = np.sum(dsort[1:p])

i = 1
p = nTrain - 1

base = []
while (i <= p & dsort[i] > 0):
    base[:, i] = dsort[i] ** (-1 / 2) * np.transpose(xmean) * vsort[:, i]
    i = i + 1

allcoor = allsamples * base
P = fisherTrain.fisher(np.transpose(allcoor), NumPerson, NumPerClassTrain).P
E = fisherTrain.fisher(np.transpose(allcoor), NumPerson, NumPerClassTrain).E
accu = 0

############################
# Testing

i = 1
while i <= NumPerson:
    i += 1
    j = NumPerClassTrain + 1
    while j <= NumPerClass:
        j += 1
        pathname = filepath + "\\s" + str(i) + "\\" + str(j) + ".pgm"

        img2 = cv2.imread(pathname, 0)
        img2 = np.multiply(np.divide((np.subtract(img2, (np.mean(img2) + 128))), np.std(img2)), 20)
        iiCell = commons.block(numx, numy, img2)
        lbpII = []
        k = 1
        while k <= numx:
            k += 1
            m = 1
            while m <= numy:
                iiCellBlock = iiCell(k, m)
                # iiCellBlock =  # cell2mat
                blockLBPII = wlcgpFile.wlcgp(iiCellBlock)
                blockLBPII = np.transpose(blockLBPII)  # transpozycja macierzy blockLBPI
                lbpII = np.concatenate((lbpII, blockLBPII))  # LBP_I=[LBP_I,Block_LBP_I];
                # end
        # end

        tcoor = lbpII * base
        tcoor = np.transpose(E) * np.transpose(tcoor)

        k = 1
        mdist = []
        while k <= nTrain:
            k += 1
            mdist[k] = np.linalg.norm(tcoor - P[:, k])
        # end


        # 3 NN algorithm
        dist, index2 = np.sort(mdist)
        class1 = np.math.floor(index2[1] / NumPerClassTrain - 0.1) + 1
        class2 = np.math.floor(index2[2] / NumPerClassTrain - 0.1) + 1
        class3 = np.math.floor(index2[3] / NumPerClassTrain - 0.1) + 1

        if (class1 != class2 & class2 != class3):
            result = class1
        else:
            if (class1 == class2):
                result = class1
            if (class2 == class3):
                result = class2
        # end

        div = div + 1

        if (result == i):
            accu = accu + 1

accu = accu / div

print accu



# np.subtract(x1, x2)
# np.std(a)
# np.mean(a)

# block
# cell2mat
