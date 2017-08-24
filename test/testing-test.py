import sys
import glob
import time
import argparse
import numpy as np
import cv2
import commons
import wlcgpFile
import fisherTrain
import matlab_wrapper


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
        xsize, ysize = img.shape  # get file size xsize - pionowo; ysize - poziomo
        img = np.double(img)

        img = ((img - np.mean(img)) + 128) / np.std(img) * 20  # ok
        iCell = commons.block(numx, numy, img)
        lbpI = []
        lbpI = np.asarray(lbpI)

        for k in range(numx):  # numx
            for m in range(numy):  # numy
                iCellBlock = iCell[k, m]  # k,m
                blockLBPI = wlcgpFile.wlcgp(iCellBlock)
                blockLBPI = np.transpose(blockLBPI)
                if (m == 0) & (k == 0):
                    lbpI = blockLBPI
                else:
                    lbpI = np.concatenate((lbpI, blockLBPI))  # LBP_I=[LBP_I,Block_LBP_I]; moze byc concatenate
                    # shape lbpI jest dobry do tego momentu
                    # if (i<=1) & (j<=1) :
                    #     print "i: " + str(i) + " j: " + str(j) + " k: " + str(k) + " m: " + str(m) + " lbpI: " + str(lbpI.shape)
        if (j == 1) & (i == 1):
            allsamples = lbpI
        else:
            allsamples = np.concatenate((allsamples, lbpI), -1)

            # if (i<=1) & (j<=1) :
            # print "i: " + str(i) + " j: " + str(j) + " lbpI: " + str(lbpI.shape) + " allsamples: " + str(allsamples.shape)

allsamples = allsamples.reshape((NumPerClassTrain * NumPerson, lbpI.size))  # size ok
# print allsamples.shape
sampleMean = np.mean(allsamples, 0).reshape((1, allsamples.shape[1]))
nTrain = np.size(allsamples, 0)

# print allsamples.shape

# print sampleMean.shape

xmean = np.zeros((nTrain, allsamples.shape[1]))
for i in range(nTrain):
    xmean[i, :] = allsamples[i, :] - sampleMean
# end

# print xmean.shape
# print xmean[:,5726] #rozmiar i wyniki ok -- na pewno OK

# np.savetxt("foo.csv", xmean, delimiter=";")

###################################################
# PCA
matlab1 = matlab_wrapper.MatlabSession()

sigma = xmean.dot(np.transpose(xmean))

# d, v = np.linalg.eig(sigma)
# # d-eigenvalues(8) v-eigenvectors(8x8)
#
# # matrix D of eigenvalues  matrix V whose columns are the corresponding right eigenvectors --MATLAB
#
# # dTemp = np.zeros((d.size,d.size))
#
# d1 = np.squeeze(np.asarray(d))  # rozmiar chyba ok

v, d = matlab1.workspace.eig(sigma, nout=2)
d1 = np.diag(d)

# matlab1.workspace.close()

# print d1.shape #6,5,1,4,3,2 wartosci ok

# d2 = d1.sort
d2 = np.sort(d1)
# print d1 #ok
# print d2 #ok
index = np.argsort(d1)
v2 = np.zeros(v.shape)

for i in range(index.size): # po tym macierze sa identyczne jak te z matlaba
    v2[:, i] = v[:,index[i]]

v2[:,v2.shape[1]-1] = v2[:,v2.shape[1]-1] * -1
index = range(0,len(index))

# print v2

# print v #wyniki OK - zawsze jedna kolumna z "-"

# print d1  # 3,4,2,1
# print v #-4,-3,-1,2 --eigeny moga sie nie zgadzac bo tak kolejnosc musi sie zgadzac z z v[:,i] musi odpowiadac w[i]
# print d #4,3,1,2 --znaki nie maja znaczenia dla eigenvectorow, wiec formalnie wszystko jest ok

rows, cols = v.shape  # ok

vsort = np.zeros((rows, cols))
# dsort = [0] * index.size
dsort = [0] * len(index)


for i in range(cols):
    vsort[:, i] = v[:, index[cols - i - 1]]  # kolejnosc nie ma znaczenia bo sobie odpowiadaja
    dsort[i] = d1[index[
        cols - i - 1]]  # czy tu ni epowinno byc d2????? - w orignale jest d1 ale maja te d1 zawsze posortowane i
    # jest to tak jakby d2 tyle ze przy d2 wywala blad w fisherTrain Q[:, i] = U[:, U.shape[1] + i - c] tylko w sumie dlaczego?
    # v2 i d2

# print dsort
# print dsort #elementy te same, zmieniona kolejnosc ale to nie powinno nic zmienic
# print vsort #elementy z -1 i zmieniona kolejnosc
# print np.size(dsort)
# print dsort wartosci nie do konca te same

##################################################################

dsum = np.sum(dsort)  # dsum = sum(dsort);
# print dsum #ok

# po co jest ten fragment??

dsumExtract = 0
p = 0

while dsumExtract / dsum < 0.95: #PO CO???
    dsumExtract = np.sum(dsort[0:p - 1])  # (dsort[1:p])
    p = p + 1


# print dsumExtract #ok

i = 0
p = nTrain - 1  # nTrain - 1


base = np.zeros((allsamples.shape[1], p))

# print base.shape #ok

while (i < p) & (dsort[i] > 0):  # (i <= p)!
    base[:, i] = ((1 / np.math.sqrt(dsort[i])) * np.transpose(xmean).dot(
        vsort[:, i]))  # dsort[i] ** (-1 / 2) * np.transpose(xmean) * vsort[:, i] SPRAWDZIC KOLEJNOSC DZIALAN
    # print i#ok
    # print dsort[i]#ok
    # print vsort[:,i] #ok
    # print (1 / np.math.sqrt(dsort[i])) #ok
    i = i + 1
# print np.transpose(xmean)[6703,:]#ok

# print base[750,0] #OK
# print base.shape #ok
# print p


##########################################

allcoor = allsamples.dot(base)

temp = fisherTrain.fisher(np.transpose(allcoor), NumPerson, NumPerClassTrain, matlab1)  # !!!

P = temp[0]
E = temp[1]
accu = 0
#
# print E #romiar OK
# print P.shape #rozmiar OK
#

matlab1.workspace.close()
# ###################################################
#
# Testing

# div = 0
#
# for i in range(1, NumPerson + 1):  # 1
#     for j in range(NumPerClassTrain + 1, NumPerClass + 1):  # 1(1, NumPerClassTest + 1):
#         pathname = filepath + "\\s" + str(i) + "\\" + str(j) + ".pgm"
#
#         img2 = cv2.imread(pathname, 0)
#         img2 = ((img2 - np.mean(img2)) + 128) / np.std(img2) * 20
#         iiCell = commons.block(numx, numy, img2)
#         lbpII = []
#         lbpII = np.asarray(lbpII)
#
#         for k in range(numx):
#             for m in range(numy):
#                 iiCellBlock = iiCell[k, m]  # k,m
#                 blockLBPII = wlcgpFile.wlcgp(iiCellBlock)
#                 blockLBPII = np.transpose(blockLBPII)
#                 if (m == 0) & (k == 0):
#                     lbpII = blockLBPII
#                 else:
#                     lbpII = np.concatenate((lbpII, blockLBPII))  # LBP_I=[LBP_I,Block_LBP_I]; moze byc concatenate
#                     # end
#         # end
#         lbpII = lbpII.reshape(1, lbpII.shape[0])
#         tcoor = lbpII.dot(base)  # lbpII * base// base trzeba tu liczyc!!! //rozmiar ok
#
#         # print np.transpose(E).shape #ok
#         # print np.transpose(tcoor).shape #ok
#
#         tcoor = np.transpose(E).dot(np.transpose(tcoor))  # E trzeba bedzie tu liczyc? ale z czego?
#
#         # print tcoor.shape #ok
#         # k = 1
#         # mdist = []
#         # while k <= nTrain:  # co z tym nTrain???
#         #     k += 1
#         mdist = [None] * nTrain
#         for k in range(nTrain):
#             mdist[k] = np.linalg.norm(tcoor - P[:, k]) #P trzeba wyciagac
#         # end
#
#         ####################################
#
#         # 3 NN algorithm
#         # dist, index2 = np.sort(mdist)
#         index2 = np.argsort(mdist)
#         dist = mdist.sort
#
#         class1 = int(np.math.floor(index2[1] / NumPerClassTrain - 0.1) + 1)  # NumPerClassTrain - skad to ma byc potem brane?
#         class2 = int(np.math.floor(index2[2] / NumPerClassTrain - 0.1) + 1)
#         class3 = int(np.math.floor(index2[3] / NumPerClassTrain - 0.1) + 1)
#
#         if (class1 != class2) & (class2 != class3):
#             result = class1
#         else:
#             if class1 == class2:
#                 result = class1
#             if class2 == class3:
#                 result = class2
#         # end
#
#         div = div + 1
#
#         print "Result: " + str(result) + " Actual: " + str(i)
#         if result == i:
#             accu = accu + 1
# # print accu
# # print div
# accuracyRate = float(accu)/float(div)
#
# print "Accuracy: " + str(accuracyRate)
