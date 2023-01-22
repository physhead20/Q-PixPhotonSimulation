''' 
author michael.rooks@mavs.uta.edu
File: makePickle

Creates a Landau lookup dictionary using the most probable values 
for a TPC geometry. The data here is for Dune TPC: (L,W,H)=(1400,365,1200)  

Michael Rooks, UTA, June 2022
'''
import numpy as np
import pylandau
from numpy import genfromtxt
import pickle

def makePickle():
    timeLandau = np.arange(0,300,1).astype('double')  
    landau = {'open': 0}
    
    counter = 0
    for axis in ['on', 'off']:
        if axis == 'on':
            for eta in etas_on[:,1]:
                for mu in mus_on[:,1]:
                    
                    params = str(round(eta,3)) + '+' + str(round(mu,3)) + '+' +  axis
                    landau[params] = pylandau.landau(timeLandau, round(mu,3), round(eta,3))
                    counter+=1
                    if counter%1000==0:
                        print(counter)
        else:
            for eta in etas_off[:,1]:
                for mu in mus_off[:,1]:
                    
                    params = str(round(eta,3)) + '+' + str(round(mu,3)) + '+' +  axis
                    landau[params] = pylandau.landau(timeLandau, round(mu,3), round(eta,3))
                    counter+=1
                    if counter%1000==0:
                        print(counter)

    f = open("landau_.pickle", "wb")
    pickle.dump(landau, f)
    f.close()

etas_off = genfromtxt('EtaOff_300.csv', delimiter=',')
mus_off = genfromtxt('MuOff_300.csv', delimiter=',')
etas_on = genfromtxt('EtaOn_300.csv', delimiter=',')
mus_on = genfromtxt('MuOn_300.csv', delimiter=',')

makePickle()





