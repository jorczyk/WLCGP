from collections import namedtuple
import numpy as np
# from scipy import linalg
import matlab_wrapper

# TODO -- kiedy ni jest rowne 1 to jest problem w petli ale to malo prawdopodobny przypadek

# fisherResult = namedtuple('fisherResult', ['P', 'E'])
# OK
# from scipy import linalg
# from scipy import linalg
# from scipy.io import matlab


def fisher(X, c, ni, matlab):
    # matlab = matlab_wrapper.MatlabSession()
    r, n = X.shape
    XT = np.transpose(X)
    mxTot = np.mean(XT, 0)  # OK

    # print mxTot.shape #OK

    # MXi = []
    temp = []
    # print X
    Xi = np.zeros(((ni * c) - 1, ni))
    # print Xi #teraz rozmiar ok
    # i = 0  # 1
    # print Xi.shape
    # for i in range(ni):
    #     Xi[:, ni-i-1] = X[:,X.shape[1]-i-1]

    # print Xi
    # print X

    for i in range(1, c + 1, 1):
        # while i <= c:
        # print "i: " + str(c)
        # print X.shape
        # j = 0
        # if i == 1:
        for j in range(1, ni + 1, 1):  # ni + 1
            # Xi[:, i] = X[:, ni * (i - 1) + range(1, ni)[j] * i]  # !!!!!!!!!! 1:ni range(1, ni)
            # Xi[:, i-1] = X[:, (ni * (i - 1)) + j - 1]  # OK?
            Xi[:, j - 1] = X[:, (i - 1) * ni + j - 1]  # OK?

        XiT = np.transpose(Xi)

        temp = np.transpose(np.mean(XiT, 0))
        # print np.mean(XiT,0)

        if i == 1:
            MXi = np.zeros((temp.size, c))
            # print temp.size
        # MXi[:, i] = np.transpose(np.mean(XiT))
        MXi[:, i - 1] = temp

        # end
    # print MXi #ok

    # print X #ok
    # print Xi #ok

    ##################
    # print MXi.shape #ok
    # print mxTot.shape

    mxTot = mxTot.reshape((mxTot.shape[0], 1))

    FXB = MXi - mxTot * np.ones((1, c))
    SB = ni * FXB.dot(np.transpose(FXB))  # SB = ni * FXB * np.transpose(FXB) sprawdzic czy dobry wynik; rozmiar OK

    # print FXB #ok
    # print SB #ok
    i = 1
    MXi = np.asmatrix(MXi)
    # print X.shape
    M = np.zeros(X.shape)
    for i in range(1, c + 1, 1):
        # while i <= c:
        #     i += 1
        # print MXi.shape
        # print (MXi[:, i - 1] * np.ones((1, ni))).shape
        for j in range(1, ni + 1, 1):
            # print (M[:, (ni * (i - 1)) + j - 1]).shape
            M[:, (ni * (i - 1)) + j - 1] = np.asarray(MXi[:, i - 1] * np.ones((1, ni)))[:,
                                           j - 1]  # !!!!!MXi[:, i] * np.ones((1, ni)) ????????????????????????
    # end
    # print M #ok

    # print M.shape
    #
    #


    SW = (X - M).dot(np.transpose(X - M))  # (X - M) * np.transpose(X - M)
    # print SW.shape #OK
    ST = (X - mxTot * np.ones((1, n))).dot(np.transpose(
        (X - mxTot * np.ones((1, n)))))  # (X - mxTot * np.ones((1, n))) * np.transpose((X - mxTot * np.ones((1, n))))
    # print ST.shape #OK
    U, S, V = np.linalg.svd(SW)
    S = np.diag(S)

    # print SW #ok
    # print ST #ok

    # # rk = np.rank(SW)  # cos dzialalo zle
    rk = np.linalg.matrix_rank(SW)
    # print SW.shape
    # print rk #ok

    if rk == U.shape[1]:
        Q = np.matrix((None))
    else:
        Q = np.zeros((U.shape[0],U.shape[1]-rk))
        for i in range(U.shape[1]-rk):
            # print U.shape[1]-i
            Q[:, i] = U[:, rk + i]

    # print U #ok
    # print Q #ok

    # print Q.shape
    # print SB.shape

    SBnew = Q.dot(np.transpose(Q)).dot(SB).dot(np.transpose(Q.dot(np.transpose(Q))))  # OK

    # print SBnew
    # E1, EV1, E2 = np.linalg.svd(SBnew)
    # A = np.array((None))
    # matlab.workspace.putvalue('SBnew', SBnew)

    E1, EV1, E2 = matlab.workspace.svd(SBnew,nout=3) #nie wiem dlaczego wywowalnie z matlaba daje inne wyniki niz matlab
    # print A

    # E1 = matlab.workspace.getvalue('B')
    # print E1
    # EV1 = np.diag(EV1) #ok
    # print E1.dot(EV1).dot(np.transpose(E2))
    # print SBnew
    # print E1

    # matlab.close()
##########################################################################################

    sig2 = np.isreal(E1)
    # print np.all(np.isreal((E1)))
    if not (np.all(np.isreal(E1))):  # sig2 == 0
        E1 = np.real(E1)
        EV1 = np.real(EV1)
    #end
    #
    EV = np.sum(EV1, 0).reshape((1, EV1.shape[1]))
    # print EV.shape
    rk1 = np.linalg.matrix_rank(EV1)  # powinno byc 2 jest 11
    # rk1=2 #FOR TESTING ONLY
    # rk1 = np.rank(EV1)
    # print EV1

    E = np.zeros((E1.shape[0], rk1))
    # i = 1
    # while i <= rk1:
    #     i += 1

    # print E.shape
    # print E1.shape
    for i in range(rk1):
        maxValue = np.amax(EV)  # [max_value,coln]=max(EV);
        coln = np.argmax(EV)
        # print coln
        E[:, i] = E1[:, coln]
        EV[:, coln] = 0
    # end
    P = np.transpose(E[:, :]).dot(X)  # np.transpose(E[:, :]) * X
    result = (P, E)
    # result = fisherResult(P, E)

    return result  # result
