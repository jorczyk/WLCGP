import numpy as np
import cv2
import commons
import wlcgpFile

filepath = ".\ORL"  # file path to dir with test faces

basePath = './train/base.npy'
variablesPath = './train/variables.npy'
EPath = './train/E.npy'
PPath = './train/P.npy'

base = np.load(basePath)
trainVariables = np.load(variablesPath)
E = np.load(EPath)
P = np.load(PPath)

NumPerson = trainVariables[0]
NumPerClass = trainVariables[1]
NumPerClassTrain = trainVariables[2]
numx = trainVariables[3]
numy = trainVariables[4]
nTrain = trainVariables[5]

div = 0
accu = 0

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
                    lbpII = np.concatenate((lbpII, blockLBPII))
                    # end
        # end

        lbpII = lbpII.reshape(1, lbpII.shape[0])
        tcoor = lbpII.dot(base)
        tcoor = np.transpose(E).dot(np.transpose(tcoor))

        mdist = [None] * nTrain
        for k in range(nTrain):
            mdist[k] = np.linalg.norm(tcoor - P[:, k])
        # end

        # 3 NN algorithm
        index2 = np.argsort(mdist)
        dist = np.sort(mdist)

        class1 = int(np.math.floor(index2[1] / NumPerClassTrain - 0.1) + 2)
        class2 = int(np.math.floor(index2[2] / NumPerClassTrain - 0.1) + 2)
        class3 = int(np.math.floor(index2[3] / NumPerClassTrain - 0.1) + 2)

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

accuracyRate = float(accu)/float(div)

print "Accuracy: " + str(accuracyRate)
