import numpy as np

basePath = './train/base.npy'
variablesPath = './train/variables.npy'
EPath = './train/E.npy'
PPath = './train/P.npy'

basetemp = np.load(basePath)
trainVariables = np.load(variablesPath)
E = np.load(EPath)
P = np.load(PPath)

print E
