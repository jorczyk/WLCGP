import numpy as np


def fisher(X, c, ni, matlab):
    r, n = X.shape
    XT = np.transpose(X)
    mxTot = np.mean(XT, 0)

    Xi = np.zeros(((ni * c) - 1, ni))

    for i in range(1, c + 1, 1):
        for j in range(1, ni + 1, 1):
            Xi[:, j - 1] = X[:, (i - 1) * ni + j - 1]

        XiT = np.transpose(Xi)

        temp = np.transpose(np.mean(XiT, 0))

        if i == 1:
            MXi = np.zeros((temp.size, c))

        MXi[:, i - 1] = temp

        # end
    mxTot = mxTot.reshape((mxTot.shape[0], 1))

    FXB = MXi - mxTot * np.ones((1, c))
    SB = ni * FXB.dot(np.transpose(FXB))

    MXi = np.asmatrix(MXi)
    M = np.zeros(X.shape)
    for i in range(1, c + 1, 1):
        for j in range(1, ni + 1, 1):
            M[:, (ni * (i - 1)) + j - 1] = np.asarray(MXi[:, i - 1] * np.ones((1, ni)))[:,
                                           j - 1]

    SW = (X - M).dot(np.transpose(X - M))

    U, S, V = np.linalg.svd(SW)  # zostawic czy zmienic na matlaba?
    S = np.diag(S)

    rk = np.linalg.matrix_rank(SW)

    if rk == U.shape[1]:
        Q = np.matrix((None))
    else:
        Q = np.zeros((U.shape[0], U.shape[1] - rk))
        for i in range(U.shape[1] - rk):
            Q[:, i] = U[:, rk + i]

    SBnew = Q.dot(np.transpose(Q)).dot(SB).dot(np.transpose(Q.dot(np.transpose(Q))))  # OK

    E1, EV1, E2 = matlab.workspace.svd(SBnew, nout=3)

    if not (np.all(np.isreal(E1))):
        E1 = np.real(E1)
        EV1 = np.real(EV1)
    # end

    EV = np.sum(EV1, 0).reshape((1, EV1.shape[1]))
    rk1 = np.linalg.matrix_rank(EV1)

    E = np.zeros((E1.shape[0], rk1))
    for i in range(rk1):
        coln = np.argmax(EV)
        # print coln
        E[:, i] = E1[:, coln]
        EV[:, coln] = 0
    # end
    P = np.transpose(E[:, :]).dot(X)
    result = (P, E)

    return result
