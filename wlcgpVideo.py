import numpy as np

import commons
import wlcgpAlgorithm


def testWlcgp(img2, numx, numy, base, E, P, nTrain, NumPerClassTrain):

    img2 = ((img2 - np.mean(img2)) + 128) / np.std(img2) * 20
    iiCell = commons.block(numx, numy, img2)
    lbpII = []
    lbpII = np.asarray(lbpII)

    for k in range(numx):
        for m in range(numy):
            iiCellBlock = iiCell[k, m]
            blockLBPII = wlcgpAlgorithm.wlcgp(iiCellBlock)
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

    result = -1
    if (class1 != class2) & (class2 != class3):
        result = class1
    else:
        if class1 == class2:
            result = class1
        if class2 == class3:
            result = class2
    # end

    return result
