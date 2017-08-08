from collections import namedtuple
import numpy as np

fisherResult = namedtuple('fisherResult', ['P', 'E'])
#OK

def fisher(X, c, ni):
    r, n = np.size(X)
    XT = np.transpose(X)
    mxTot = np.mean(XT)  # mx_tot = mean(XT)';

    MXi = []
    i = 1
    while i <= c:
        Xi = X[:, ni * (i - 1) + 1:ni * i]
        XiT = np.transpose(Xi)
        MXi[:, i] = np.mean(XiT)
        i = i + 1
    # end

    FXB = MXi - mxTot * np.ones(1, c)
    SB = ni * FXB * np.transpose(FXB)

    i = 1
    M = []
    while i <= c:
        i += 1
        M[:, ni * (i - 1) + 1:ni * i] = MXi[:, i] * np.ones(1, ni)
    # end

    SW = (X - M) * np.transpose(X - M)
    ST = (X - mxTot * np.ones(1, n)) * np.transpose((X - mxTot * np.ones(1, n)))
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
    result = fisherResult(P, E)

    return result
