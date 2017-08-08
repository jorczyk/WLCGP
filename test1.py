import sys
import glob
import time
import argparse
import numpy as np
import cv2
import commons
import wlcgpFile
import fisherTrain

filepath = ".\ORL"  # file path to dir with test faces

NumPerson = 40  # number of classes
NumPerClass = 10  # number of faces for each class
NumPerClassTrain = 8  # trainging count for each class
NumPerClassTest = NumPerClass - NumPerClassTrain  #

allsamples = []  #

numx = 4  # image segmentation settings
numy = 2

filePath = filepath + "\s" + str(1) + "\\" + str(1) + ".pgm"  # filepath to read
img = cv2.imread(filePath, 0)
xsize, ysize = img.shape  # get file size xsize - pionowo; ysize - poziomo
img = np.double(img)

#print img[0, 1] # 49 ok
#print np.size(img,1) # 112x92 ok

img = ((img - np.mean(img))+128)/np.std(img) * 20 # ok

#print np.size(img,0) # 112x92 ok

iCell = commons.block(numx, numy, img)
#print np.size(iCell,0) #do poprawy kolejnosc ale generalnie ok --chyba ok

lbpI = []
k = 0

# while k <= numx:
for k in range(numx):
    #k += 1
    #m = 0
    #while m <= numy:
    for m in range(numy):
        #m += 1
        iCellBlock = iCell[k,m]

        #np.set_printoptions(threshold='nan')
        #print (iCellBlock[:,:]) #28x46 ok
        #print np.size(iCellBlock,1) #46 x 28 --powinno byc odwrotnie

        # iCellBlock = #cell2mat --nie wiem czy potrzebujemy bo ta funkcja jest specyficzna dla matlaba i rozpisuje "cell" na

        blockLBPI = wlcgpFile.wlcgp(iCellBlock) #!!!!
        # blockLBPI = np.transpose(blockLBPI)  # transpozycja macierzy blockLBPI
        # lbpI = np.concatenate((lbpI, blockLBPI))  # LBP_I=[LBP_I,Block_LBP_I];

allsamples = np.concatenate((allsamples, lbpI))
