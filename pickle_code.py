''' 
author michael.rooks@mavs.uta.edu
File: pickle_code

For analysing saved data

Michael Rooks, UTA, June 2022
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
import math

def update(frame):
    p = aaa[frame,:,:]
    ax.imshow(p,interpolation='none',extent=[0,200,200,0])
    ax.set_title("Time: {}-{} ns".format(frame,frame+1), fontsize=20)
    ax.set_xlabel('z [cm]', fontsize=15)
    ax.set_ylabel('y [cm]', fontsize=15)

def getPixleMap(file):
    with open(file, 'rb') as f:
        a = pickle.load(f)
    return a

aaa = getPixleMap('Event2_zoo.pkl')
x = range(len(aaa))
FRAME = 20
p = aaa[FRAME,:,:]
y = [sum(sum(aaa[i])) for i in x]
plt.plot(x,y,'.k'); plt.yscale('log')
plt.grid(); plt.xlabel('Time [ns]');plt.ylabel('# Photons')
mx = x[int(np.where(y == max(y))[0][0])]
sss = sum(sum(sum(aaa[0:mx])))
P_total = sum(sum(sum(aaa)))
P_frame = sum(sum(p))
print('Integration Time:',len(aaa),'ns')
print('Total Photons:',P_total)
print('Peak Time:',mx, 'ns')
print('Photons to peak:', sss,round(sss*100/P_total,1),'%')
print('Photons in frame %s:'%FRAME, P_frame,round(P_frame*100/P_total,1),'%')

''' make a gif '''
# plt.rcParams["figure.figsize"] = [5, 5]
# plt.rcParams["figure.autolayout"] = True
# fig, ax = plt.subplots()

# anim = FuncAnimation(fig, update, frames=35, interval=250)
# anim.save('Event2_zoo.gif', writer='imagemagick', fps=2)






