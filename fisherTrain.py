from collections import namedtuple
import numpy as np


# fisherResult = namedtuple('fisherResult', ['P', 'E'])
# OK

def fisher(X, c, ni):
    r, n = X.shape
    XT = np.transpose(X)
    mxTot = np.mean(XT)  # mx_tot = mean(XT)';

    # MXi = []
    temp = []
    Xi = np.zeros(X.shape)
    # i = 0  # 1
    for i in range(1, c + 1, 1):
        # while i <= c:
        # print "i: " + str(c)
        # print X.shape
        # j = 0
        for j in range(1, ni + 1, 1):  # for j in range(ni - 1):
            # print (ni * (i - 1)) + j -1 #ok
            # print (j * i)
            # print ((ni * (i - 1)) + (j * i))
            # Xi[:, i] = X[:, ni * (i - 1) + range(1, ni)[j] * i]  # !!!!!!!!!! 1:ni range(1, ni)
            Xi[:, i] = X[:, (ni * (i - 1)) + j - 1]  # OK?
            # ni*(i-1)+1:ni*i
        # print Xi #ok?
        XiT = np.transpose(Xi)
        # # print np.transpose(np.mean(XiT,1))
        # print XiT.shape
        temp = np.transpose(np.mean(XiT, 0))
        # print np.mean(XiT,0)

        if i == 1:
            MXi = np.zeros((temp.size, c))
            # print temp.size
        # MXi[:, i] = np.transpose(np.mean(XiT))
        MXi[:, i - 1] = temp
        # print MXi
        # end

    ##################

    FXB = MXi - mxTot * np.ones((1, c))
    SB = ni * FXB * np.transpose(FXB)  # sprawdzic czy dobry wynik; rozmiar OK

    # print FXB.shape
    # print SB.shape
    # i = 1
    MXi = np.asmatrix(MXi)
    # print X.shape
    M = np.zeros(X.shape)
    for i in range(1, c + 1, 1):
        # while i <= c:
        #     i += 1
        # print MXi.shape
        #print (MXi[:, i - 1] * np.ones((1, ni))).shape
        for j in range(1, ni + 1, 1):
            #print (M[:, (ni * (i - 1)) + j - 1]).shape
            M[:, (ni * (i - 1)) + j - 1] = np.asarray(MXi[:, i - 1] * np.ones((1, ni)))[:,
                                           j - 1]  # !!!!!MXi[:, i] * np.ones((1, ni)) ????????????????????????
    # end
    print M.shape
    SW = (X - M).dot(np.transpose(X - M)) #(X - M) * np.transpose(X - M)
    ST = (X - mxTot * np.ones((1, n))).dot(np.transpose((X - mxTot * np.ones((1, n)))))#(X - mxTot * np.ones((1, n))) * np.transpose((X - mxTot * np.ones((1, n))))
    U, S, V = np.linalg.svd(SW)
    rk = np.rank(SW)
    Q = U
    Q[:, 1:rk] = []  # Q(:,1:rk)=[];
    SBnew = Q * np.transpose(Q) * SB * np.transpose(Q * np.transpose(Q))
    E1, EV1, E2 = np.linalg.svd(SBnew)

    sig2 = np.isreal(E1)
    if (sig2 == 0):
        E1 = np.real(E1)
        EV1 = np.real(EV1)
    # end

    EV = np.sum(EV1)
    rk1 = np.rank(EV1)

    E = []

    i = 1
    while i <= rk1:
        i += 1
        maxValue, coln = np.max(EV)  # [max_value,coln]=max(EV);
        E[:, i] = E1[:, coln]
        EV[:, coln] = 0
    # end
    P = np.transpose(E[:, :]) * X
    result = (P, E)
    # result = fisherResult(P, E)

    return result
