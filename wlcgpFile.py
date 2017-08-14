import numpy as np

#TODO -- zle obliczanie wartosci w histogramie i te dziwne 0 kolumny

def wlcgp(image):
    dImage = np.double(image)
    ysize, xsize = dImage.shape  # ysize, xsize = dImage.shape

    # print xsize
    bSizey = 3
    bSizex = 3

    BELTA = 5
    ALPHA = 3
    EPSILON = 0.0000001
    PI = 3.141592653589
    numNeighbors = 8

    # filter
    f00 = np.matrix('1 1 1;1 -8 1;1 1 1')

    # calculate dx and dy
    dx = xsize - bSizex
    dy = ysize - bSizey

    # print dx, dy
    # init the result matrix with 0

    dDifferentialExcitation = np.zeros((dy + 2, dx + 2))
    dGradientOrientation = np.zeros((dy + 2, dx + 2))

    # print np.shape(dDifferentialExcitation) #ok
    # compute WLCGP code per pixel
    b = []
    # count = 0
    y = 1  # 2
    while y <= ysize - 2:
        # y += 1
        # count += 1
        # print y
        x = 1  # 2
        while x <= xsize - 2:  # x <= xsize - 1:
            # x += 1
            N = dImage[y - 1:y + 1, x - 1:x + 1]
            # print xsize
            # print x, y
            center = dImage[y, x]

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

            # b = np.append(b,v00)

            if (v01 != 0):
                dDifferentialExcitation[y, x] = np.math.atan(ALPHA * v00 / v01)
            else:
                dDifferentialExcitation[y, x] = 0.1

            N1 = dImage[y - 1, x]
            N5 = dImage[y + 1, x]
            N3 = dImage[y, x + 1]
            N7 = dImage[y, x - 1]

            if (abs(N7 - N3) < EPSILON):
                dGradientOrientation[y, x] = 0
            else:
                v10 = N5 - N1
                v11 = N7 - N3
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
    # print b
    # end

    # histogram

    M = 6
    T = 8
    S = 20
    C = M * S

    cVal = np.arange(-PI / 2, PI / 2 + PI / C, PI / C)
    # cVal = #Cval = -pi/2:pi/C:pi/2; % Excitation

    # cValcen = #(Cval(1:end-1)+Cval(2:end)) #
    cValcen = np.arange(-(PI / 2), PI / 2 + (PI / C) - 0.0000001,
                        PI / C)  # If bins is a sequence, it defines the bin edges, including the rightmost edge,
    # allowing for non-uniform bin widths.

    # cValcen = np.arange(-(PI / 2 + PI / C), PI / 2 + (2 * PI / C),
    #                     PI / C) ////+ (1/2*PI / C)

    #print "cvalen: " + str(np.size(cValcen))
    tVal = np.arange(0, 360 + 360 / T, 360 / T)
    # Tval = 0:360/T:360;

    h2d = np.zeros((C, T))
    #print "h2d size: " + str(np.shape(h2d))

    i = 0
    temp = []
    # while i <= T-1:
    for i in range(T):  # T
        if i > 1:
            temp = dDifferentialExcitation[
                (dGradientOrientation > tVal[i]) & (dGradientOrientation <= tVal[i + 1])]  # boolean?
        else:
            temp = dDifferentialExcitation[
                (dGradientOrientation >= tVal[i]) & (dGradientOrientation <= tVal[i + 1])]  # chyba cos tu nie gra
        # end

        # print dDifferentialExcitation #ok
        # print "temp: " + str(temp)
        #print "histogram: " + str((np.histogram(temp, cValcen)[0]))
        h2d[:, i] = np.histogram(temp, cValcen)[0]
        # zle wartosci - ale jak beda dla kazdego takie same to luz ALE SA 0!! --do poprawienia!!!
    # end

    #print np.size(np.histogram(temp, cValcen)[0])

    h = np.transpose(h2d)
    h = np.reshape(h, (T, S, M))

    j = 0
    temph = np.empty(0)
    for j in range(M):
    # while j <= M:
        # j += 1
        temp = h[:, :, j]
        # temph = np.concatenate(temph, temp[:])  # h = reshape(h,[T,S,M]);
        temph = np.append(temph, temp[:])
    # end
    h1d = temph
    #print h1d.shape #ok
    return h1d
