import numpy as np
import cv2
import commons
import wlcgpFile
import fisherTrain
from tempfile import TemporaryFile

filepath = ".\ORL"  # file path to dir with test faces

NumPerson = 2  # number of classes #3
NumPerClass = 10  # number of faces for each class
NumPerClassTrain = 3  # trainging count for each class #4
NumPerClassTest = NumPerClass - NumPerClassTrain  #

allsamples = np.empty((0, 0))

numx = 4  # image segmentation settings
numy = 2

for i in range(1, NumPerson + 1):
    for j in range(1, NumPerClassTrain + 1):
        filePath = filepath + "\s" + str(i) + "\\" + str(j) + ".pgm"  # filepath to read
        img = cv2.imread(filePath, 0)
        xsize, ysize = img.shape
        img = np.double(img)

        img = ((img - np.mean(img)) + 128) / np.std(img) * 20  # ok
        iCell = commons.block(numx, numy, img)
        lbpI = []
        lbpI = np.asarray(lbpI)

        for k in range(numx):
            for m in range(numy):
                iCellBlock = iCell[k, m]
                blockLBPI = wlcgpFile.wlcgp(iCellBlock)
                blockLBPI = np.transpose(blockLBPI)
                if (m == 0) & (k == 0):
                    lbpI = blockLBPI
                else:
                    lbpI = np.concatenate((lbpI, blockLBPI))  # LBP_I=[LBP_I,Block_LBP_I]; moze byc concatenate

        if (j == 1) & (i == 1):
            allsamples = lbpI
        else:
            allsamples = np.concatenate((allsamples, lbpI), -1)

allsamples = allsamples.reshape((NumPerClassTrain * NumPerson, lbpI.size))  # size ok

sampleMean = np.mean(allsamples, 0).reshape((1, allsamples.shape[1]))
nTrain = np.size(allsamples, 0)

xmean = np.zeros((nTrain, allsamples.shape[1]))
for i in range(nTrain):
    xmean[i, :] = allsamples[i, :] - sampleMean
# end

###################################################
# PCA
sigma = xmean.dot(np.transpose(xmean))

d, v = np.linalg.eig(sigma)

d1 = np.squeeze(np.asarray(d))
d2 = d1.sort
index = np.argsort(d1)

rows, cols = v.shape

vsort = np.zeros((rows, cols))
dsort = [0] * index.size

for i in range(cols):
    vsort[:, i] = v[:, index[cols - i - 1]]  # !!
    dsort[i] = d1[index[cols - i - 1]]

dsum = np.sum(dsort)
dsumExtract = 0
p = 0

while dsumExtract / dsum < 0.95:
    dsumExtract = np.sum(dsort[0:p - 1])
    p = p + 1

i = 0
p = nTrain - 1

base = np.zeros((allsamples.shape[1], p))
while (i < p) & (dsort[i] > 0):  # (i <= p)!
    base[:, i] = (dsort[i] ** (-1 / 2)) * (np.transpose(xmean).dot(
        vsort[:, i]))
    i = i + 1

allcoor = allsamples.dot(base)
temp = fisherTrain.fisher(np.transpose(allcoor), NumPerson, NumPerClassTrain)  # !!!

P = temp[0]
E = temp[1]
accu = 0

#################################

# SAVING variables to files

basePath = './train/base.npy'
np.save(basePath, base)

trainVariables = [NumPerson, NumPerClass, NumPerClassTrain, numx, numy, nTrain]
variablesPath = './train/variables.npy'
np.save(variablesPath, trainVariables)

EPath = './train/E.npy'
PPath = './train/P.npy'
np.save(EPath, E)
np.save(PPath, P)
