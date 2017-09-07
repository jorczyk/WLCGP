import numpy as np

def wlcgp(image):
    dImage = np.double(image)
    ysize, xsize = dImage.shape  # ysize, xsize = dImage.shape

    bSizey = 3
    bSizex = 3

    BELTA = 5
    ALPHA = 3
    EPSILON = 0.0000001
    PI = 3.141592653589
    numNeighbors = 8

    f00 = np.matrix('1 1 1;1 -8 1;1 1 1')

    # calculate dx and dy
    dx = xsize - bSizex
    dy = ysize - bSizey


    # init the result matrix with 0
    dDifferentialExcitation = np.zeros((dy + 2, dx + 2))
    dGradientOrientation = np.zeros((dy + 2, dx + 2))


    # compute WLCGP code per pixel
    y = 1
    while y <= ysize - 2:
        x = 1
        while x <= xsize - 2:
            # N = dImage[y - 1:y + 1, x - 1:x + 1]
            # center = dImage[y, x]

            v00 = (abs(dImage[y + 1, x] - dImage[y + 1, x - 1])) + abs((dImage[y + 1, x + 1] - dImage[y + 1, x])) + abs(
                (dImage[y, x + 1] \
                 - dImage[y + 1, x + 1])) + abs((dImage[y - 1, x + 1] - dImage[y, x + 1])) + abs(
                (dImage[y - 1, x] - dImage[y - 1, x + 1])) \
                  + abs((dImage[y - 1, x - 1] - dImage[y - 1, x])) + abs(
                (dImage[y, x - 1] - dImage[y - 1, x - 1])) + abs((dImage[y + 1, x - 1] \
                                                                  - dImage[y, x - 1])) + abs(
                (dImage[y + 1, x - 1] - dImage[y, x])) + abs((dImage[y + 1, x] - dImage[y, x])) + abs(
                (dImage[y + 1, x + 1] \
                 - dImage[y, x])) + abs((dImage[y, x + 1] - dImage[y, x])) + abs((dImage[y - 1, x + 1] - dImage[y, x])) \
                  + abs((dImage[y - 1, x] - dImage[y, x])) + abs((dImage[y - 1, x - 1] - dImage[y, x])) + abs(
                (dImage[y, x - 1] - dImage[y, x]))

            v01 = (
                      dImage[y - 1, x - 1] + dImage[y, x - 1] + dImage[y + 1, x - 1] + dImage[y - 1, x] + dImage[
                          y + 1, x] +
                      dImage[y - 1, x + 1] \
                      + dImage[y, x + 1] + dImage[y + 1, x + 1]) / 9

            if v01 != 0:
                dDifferentialExcitation[y, x] = np.math.atan(ALPHA * v00 / v01)
            else:
                dDifferentialExcitation[y, x] = 0.1

            N1 = dImage[y - 1, x]
            N5 = dImage[y + 1, x]
            N3 = dImage[y, x + 1]
            N7 = dImage[y, x - 1]

            if abs(N7 - N3) < EPSILON:
                dGradientOrientation[y, x] = 0
            else:
                v10 = N5 - N1
                v11 = N7 - N3

                dGradientOrientation[y, x] = np.arctan(v10 / v11)
                dGradientOrientation[y, x] = dGradientOrientation[y, x] * 180 / PI

                if (v11 > EPSILON) & (v10 > EPSILON):
                    dGradientOrientation[y, x] = dGradientOrientation[y, x] + 0
                elif (v11 < -EPSILON) & (v10 > EPSILON):
                    dGradientOrientation[y, x] = dGradientOrientation[y, x] + 180
                elif (v11 < EPSILON) & (v10 < -EPSILON):
                    dGradientOrientation[y, x] = dGradientOrientation[y, x] + 180
                elif (v11 > EPSILON) & (v10 < -EPSILON):
                    dGradientOrientation[y, x] = dGradientOrientation[y, x] + 360
                    # end
                    # end
                    # end
            x += 1
        y += 1
    # endk

    # HISTOGRAM
    M = 6
    T = 8
    S = 20
    C = M * S

    # cVal = np.arange(-PI / 2, PI / 2 + PI / C, PI / C)
    cValcen = np.arange(-(PI / 2), PI / 2 + (PI / C) - 0.0000001,
                        PI / C)

    tVal = np.arange(0, 360 + 360 / T, 360 / T)

    h2d = np.zeros((C, T))

    for i in range(T):
        if i > 1:
            temp = dDifferentialExcitation[
                (dGradientOrientation > tVal[i]) & (dGradientOrientation <= tVal[i + 1])]  # boolean?
        else:
            temp = dDifferentialExcitation[
                (dGradientOrientation >= tVal[i]) & (dGradientOrientation <= tVal[i + 1])]  # chyba cos tu nie gra
        # end
        h2d[:, i] = np.histogram(temp, cValcen)[0]
    # end

    h = np.transpose(h2d)
    h = np.reshape(h, (T, S, M), order="F")

    temph = np.empty(0)
    for j in range(M):
        temp = h[:, :, j].flatten(1)
        temph = np.append(temph, temp)
    # end
    h1d = temph

    return h1d
