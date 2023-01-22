import numpy as np
import uproot3 as uproot
import math
from numpy import genfromtxt
import pylandau
import pickle
import random
import csv
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from multiprocessing import Pool, cpu_count
from matplotlib.animation import FuncAnimation, PillowWriter
from numpy import loadtxt
from functools import partial
from decimal import Decimal
import math
import bisect


def ND(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def getScintillation():
    global bin_midpoints, cdf

    tau_fast = 4.9                # Singlet decay         [ns]
    tau_slow = 1260               # Triplet decay         [ns]
    tau_mid = 34                  # Intermediate decay    [ns] 
    A = .188                      # Singlet abundance      
    B = 0.738                     # Triplet abundance
    C = 0.074                     # Intermediate abundance
    k = .11/1000                  # N2 Rate constant      [1/ppm*ns] 
    beta = 530                    # Saturation constant   [ppm] 
    N2 = 2                        # Nitrogen              [ppm]   
    TTime = np.arange(0,2000,0.1) # Fill time             [ns]
    
    tff = (1/tau_fast) + k*beta*(1-np.exp(-N2/beta))
    tf = (tff * np.exp(-TTime*tff) * A) / (1 + tau_fast*k*beta*(1-np.exp(-N2/beta))) 
    
    tmm = (1/tau_mid) + k*beta*(1-np.exp(-N2/beta))
    tm = (tmm * np.exp(-TTime*tmm) * C) / (1 + tau_mid*k*beta*(1-np.exp(-N2/beta)))  
    
    tss = (1/tau_slow) + k*beta*(1-np.exp(-N2/beta))
    ts = (tss * np.exp(-TTime*tss) * B) / (1 + tau_slow*k*beta*(1-np.exp(-N2/beta)))  

    weight = tf+tm+ts
    histo, bins = np.histogram(TTime, 10001, density=False, weights = weight)
    bin_midpoints = bins[:-1] + np.diff(bins)/2
    cdf = np.cumsum(histo)
    cdf = cdf / cdf[-1]

def getRandom(PhotonCount):
    values = np.random.rand(PhotonCount)
    value_bins = np.searchsorted(cdf, values)

    return bin_midpoints[value_bins]

def getGH(Dis,Angle):
    GHFit = [(151,27,-1090, 1.4),(149,24,-1202,1.38),(141,24,-1285,1.28),(134,74,-355, 1.17),(154,75,-353, 1.05)
            ,(161,84,-373, 0.93),(197,87,-359, 0.78),(211,97,-372, 0.64),(216,107,-375, 0.50)]
    
    oo = 0
    for aa in [0,10,20,30,40,50,60,70,80]:
        if (Angle >= aa) and (Angle <= aa + 10):
            ghfit = GHFit[oo]
        oo +=1
    
    return ghfit[3] *np.exp((ghfit[0]-Dis)/ghfit[1])*( (Dis-ghfit[2]) / (ghfit[0]-ghfit[2]) )**( (ghfit[0]-ghfit[2])/ghfit[1])

def num_photons(zloc, yloc, idx, Edep, larql):  
    AL = 3000 # Absorbtion Length for 2 ppm n2 [cm]
    
    PixLoc = np.array([0, yloc+0.5*yLen, zloc+0.5*zLen]) 
    EdepLoc = (Edep_x[idx][Edep] + x_off, Edep_y[idx][Edep] + y_off, Edep_z[idx][Edep] + z_off)
    Dis = math.dist(PixLoc,EdepLoc)
   
    n1 = np.arctan( (yloc+yLen - EdepLoc[1])*(zloc+zLen - EdepLoc[2]) / (EdepLoc[0] * np.sqrt( (yloc+yLen - EdepLoc[1])**2 + (zloc+zLen - EdepLoc[2])**2 + EdepLoc[0]**2) ) ) 
    n2 = np.arctan( (yloc    - EdepLoc[1])*(zloc+zLen - EdepLoc[2]) / (EdepLoc[0] * np.sqrt( (yloc    - EdepLoc[1])**2 + (zloc+zLen - EdepLoc[2])**2 + EdepLoc[0]**2) ) )
    n3 = np.arctan( (yloc+yLen - EdepLoc[1])*(zloc    - EdepLoc[2]) / (EdepLoc[0] * np.sqrt( (yloc+yLen - EdepLoc[1])**2 + (zloc    - EdepLoc[2])**2 + EdepLoc[0]**2) ) )
    n4 = np.arctan( (yloc    - EdepLoc[1])*(zloc    - EdepLoc[2]) / (EdepLoc[0] * np.sqrt( (yloc    - EdepLoc[1])**2 + (zloc    - EdepLoc[2])**2 + EdepLoc[0]**2) ) )
    
    SolidAngle = n1 - n2 - n3 + n4
    Angle = np.arccos(EdepLoc[0]/Dis) * (180/np.pi)
    PhotonCount = np.exp(-Dis/AL) * (SolidAngle * larql) * getGH(Dis,Angle) / (4*np.pi) 
    
    return PhotonCount, Dis, Angle

def getRootK(FILE):
    global number_events, g4_nParticles, g4_trkID, g4_PDG, nu_isp_number
    global Edep_number, Edep_dq, Edep_x
    global Edep_z, Edep_trkID, nu_isp_pdg, Edep_y, Edep_t,Edep_t_end,Edep_length
    
    with uproot.open(HEAD_ROOT + FILE) as f:
        tree = f['event_tree']
        Edep_x = tree.array(['hit_start_x'])
        g4_nParticles = tree.array(['number_particles'])
        g4_trkID = tree.array(['particle_track_id'])
        g4_PDG = tree.array(['particle_pdg_code'])
        Edep_number = tree.array(['number_hits'])
        Edep_dq = tree.array(['hit_energy_deposit'])
        
        Edep_x = tree.array(['hit_start_x'])
        Edep_y = tree.array(['hit_start_y'])
        Edep_z = tree.array(['hit_start_z'])
        
        Edep_t = tree.array(['hit_start_t'])
        Edep_trkID = tree.array(['hit_track_id'])
        Edep_t_end = tree.array(['hit_end_t'])
        Edep_length = tree.array(['hit_length'])

def getCSV():
    global etas_on, mus_on, etas_off, mus_off, k_on,k_off
    
    DisLan = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]

    MuOn = [0.1726, 1.90885, 3.7326, 5.64385, 7.6426, 9.72885, 11.9026, 14.16385, 16.5126, 18.94885, 21.4726, 24.08385, 26.67297]
    MuOff = [-0.001, 1.90372, 3.87141, 5.99964, 8.34351, 10.92381, 13.73528, 16.75477, 19.94945, 23.28505, 26.73401, 30.2837, 33.79567]

    EtaOn = [0.29546, 0.27341, 0.30446, 0.3882, 0.52564, 0.71925, 0.97298, 1.29217, 1.68367, 2.15574, 2.71811, 3.38194, 4.12639]


    EtaOff = [0.15, 0.2075, 0.505, 0.9025, 1.4, 1.9975, 2.695, 3.4925, 4.39, 5.3875, 6.485, 7.6825, 8.92618] 

    DisExp = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]


    KOn = [-0.209, -0.16771, -0.1354, -0.11011, -0.09031, -0.07481, -0.06269, -0.05319, -0.04576, -0.03995, -0.0354, -0.03183, -0.02914]

    KOff = [-0.187, -0.14045, -0.10664, -0.08209, -0.06427, -0.05132, -0.04192, -0.0351, -0.03014, -0.02654, -0.02393, -0.02203, -0.0207]

    etas_on = np.array((DisLan,EtaOn)).T
    mus_on = np.array((DisLan,MuOn)).T
    etas_off = np.array((DisLan,EtaOff)).T
    mus_off = np.array((DisLan,MuOff)).T
    k_on = np.array((DisExp,KOn)).T
    k_off = np.array((DisExp,KOff)).T
    
def getLandau(dis,angle,timeLandau,PhotonCount):
    
    if angle < 45:
        axis = 'on'
        
        eta = etas_on[bisect.bisect_left(etas_on[:,0], dis)-1,1]
        mu = mus_on[bisect.bisect_left(mus_on[:,0], dis)-1,1]
        k = k_on[bisect.bisect_left(k_on[:,0], dis)-1,1]  
    else:
        axis = 'off'
        eta = etas_off[bisect.bisect_left(etas_off[:,0], dis)-1,1]
        mu = mus_off[bisect.bisect_left(mus_off[:,0], dis)-1,1]
        k = k_off[bisect.bisect_left(k_off[:,0], dis)-1,1]
    
    params = str(round(eta,3)) + '+' + str(round(mu,3)) + '+' + axis
    lan = ND(landau[params])
    exp = np.exp(timeLandau*k)
    
    idx = np.argwhere(np.diff(np.sign(exp-lan))).flatten()
    landau2 = lan[:idx[1]]; exp = exp[idx[1]:]
    lan_exp = np.concatenate((landau2,exp))
    
    return random.choices(timeLandau, lan_exp, k = PhotonCount)
      
def LArQL(dE,dx):
    
    A = 0.800; k = 0.0486; Wion = 23.6; pLAr = 1.39; Nex_Ni = 0.21;EF = 0.5 
    p1 = 2.15157266e-05; p2 = -3.98850458e+00; p3 = 1.38343421e+00; p4 = 1.99195217e-06
    alpha = 0.032; beta = 0.008258
    Edep = dE/dx; 
    if Edep < 1.0: 
        Edep = 1.0
    Qbirks = (A*Edep*1e6/Wion) / (1+(k*Edep/(pLAr*EF))) 
    QChi = p1 / (p2 + np.exp(p3 + p4*Edep))
    Qcorr = np.exp(-EF / (alpha*np.log(Edep) + beta))
    
    return ((Edep*1e6/Wion)*(1 + Nex_Ni - QChi*Qcorr) - Qbirks)*dx
    
def getParticles(Event, PID):    
    b = np.array([g4_PDG[Event][np.in1d(g4_trkID[Event],Edep_trkID[Event][oo])][0] for oo in range(len(Edep_trkID[Event]))])

    return np.where(b == PID)[0].tolist()

def dP3(dist):
    remainder = 0
    distTotal = []
    distSum = sum(dist)
    pTotal = round(distSum)
    for weight in dist:
        remainder, weighted_value = math.modf(weight*pTotal/distSum + remainder)
        distTotal.append(weighted_value)
        distTotal[-1] = round(distTotal[-1])
    return distTotal  

def Pix(values):
    timeLandau = np.arange(0,300,1).astype('double')  
    TimeMapZY = np.zeros([intTime, int(yDim/yLen), int(zDim/zLen)])
    
    for Edep in values:
        if((Edep_x[idx][Edep] + x_off) < 0 or (Edep_x[idx][Edep] + x_off) > xDim or
            (Edep_y[idx][Edep] + y_off) < 0 or (Edep_y[idx][Edep] + y_off) > yDim or
            (Edep_z[idx][Edep] + z_off) < 0 or (Edep_z[idx][Edep] + z_off) > zDim or
                Edep_t[idx][Edep] > 6000):
            continue
    
        larql = LArQL(Edep_dq[idx][Edep],Edep_length[idx][Edep])
        
        DepTime = Edep_t[idx][Edep]
        
        PC, Dis, Angle = [],[],[]
        
        for zLoc in np.arange(0, zDim, zLen):
            for yLoc in np.arange(0, yDim, yLen):
                
                PhotonCount, dis, angle = num_photons(zLoc,yLoc,idx,Edep,larql)
                PC.append(PhotonCount); Dis.append(dis); Angle.append(angle) 
          
        Pdis = dP3(PC)
       
        row = column = counter = 0
        for zLoc in np.arange(0, zDim, zLen):
            for yLoc in np.arange(0, yDim, yLen):
               
                times = getTimes(Pdis[counter],DepTime,
                                 getLandau(Dis[counter],Angle[counter],timeLandau,Pdis[counter]) )                
              
                if times.size != 0:
                    for oo in times:                    
                        TimeMapZY[oo,row,column]+=1
                row+=1
                counter+=1
            column +=1
            row = 0
    
    return TimeMapZY
    
def getTimes(PhotonCount,depTime,landau):
   
   time = getRandom(PhotonCount) + depTime*np.ones(PhotonCount) + landau
   times = np.around(time).astype(int)
   
   return times[(times < intTime)]   

def start():
    global landau
    getScintillation(); getCSV()
    with open(HEAD +'landau_300_25cm.pickle', 'rb') as f:
        landau = pickle.load(f)

HEAD = ' '
HEAD_ROOT = ' '

''' Detector Parameters'''
xDim = 350; yDim = 600; zDim = 230 
x_off = 0; y_off = 0; z_off = 0 

zLen = 10
yLen = 10          
intTime = 2000             

start() 

getRootK('kaon.root')


idx = 2               
cpu = 4                    

# kaons = getParticles(idx,321)
# pions = getParticles(idx,211)
# muons = getParticles(idx,-13)
# electrons = getParticles(idx,11)
# positrons = getParticles(idx,-11)
zoo = np.arange(0,Edep_number[idx],1).tolist()
chunks = np.array_split(zoo, cpu)

def main():

    start = timer()

    with Pool(processes=cpu) as pool:

        count = pool.map(Pix, chunks)
        final = sum(count)           
        ph = sum(sum(sum(final)))
        print(ph)
        f = open('Even2.pkl', 'wb')
        pickle.dump(final, f)
        f.close()
        end = timer()
        print(f'elapsed time: {(end - start)/60}')

if __name__=='__main__':
    main() 


       



