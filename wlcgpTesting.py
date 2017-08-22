import numpy as np

import commons
import wlcgpFile


def testWlcgp(img2, numx, numy, base, E, P, nTrain, NumPerClassTrain):
    # print img2.shape

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
        mdist[k] = np.linalg.norm(tcoor - P[:, k])  # P trzeba wyciagac
    # end

    ####################################

    # 3 NN algorithm
    # dist, index2 = np.sort(mdist)
    index2 = np.argsort(mdist)
    dist = mdist.sort

    class1 = int(
        np.math.floor(index2[1] / NumPerClassTrain - 0.1) + 1)  # NumPerClassTrain - skad to ma byc potem brane?
    class2 = int(np.math.floor(index2[2] / NumPerClassTrain - 0.1) + 1)
    class3 = int(np.math.floor(index2[3] / NumPerClassTrain - 0.1) + 1)

    # print index2

    if (class1 != class2) & (class2 != class3):
        result = class1
    else:
        if class1 == class2:
            result = class1
        if class2 == class3:
            result = class2

    return result
