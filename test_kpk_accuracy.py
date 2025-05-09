# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 20:18:26 2020

@author: pranj
"""


import matplotlib.pyplot as plt
from numpy import exp, sqrt, pi, inf, sinh, log, arange
import numpy as np
from scipy import integrate, optimize
from scipy.interpolate import interp1d
from scipy.integrate import odeint, solve_ivp, ode
from scipy.misc import derivative
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)
from matplotlib.lines import Line2D
import os
from scipy.special import kn
import pandas as pd
import time
from scipy.special import lambertw

os.chdir('../../pranjal work/git-python/cannibal-base-code')
from background import bck, bck2
from equilibrium import wfun, Xcan
from perturbation import pert
os.chdir('../../../draft/cannibal-long_paper_figs')
def table(f,t):
    tablef=np.zeros(t.shape[0])
    for i in np.arange(t.shape[0]):
        tablef[i]=f(t[i])
    return tablef
# =============================================================================
# variation keeping afz and arh fixed
# =============================================================================

afz=1*10**3 #scale factor at which cannibal reactions freezeout and cannibal evolves as CDM
arh=100*10**5 #scale at which we want reheating to happen
Trh=5 #in MeV


xi=10
m,xi,alpha_can,afz2,adom2,arh2,rhocan,rhor,X,w,c2s,Ht,rhodm,hor_i,aeq,gamma,H1=bck2(afz,xi,arh,Trh,2)
ddel,ddelsim,ddel_super_sim,ddel_post_rh,ddel_r=pert(rhocan,rhor,rhodm,Ht,c2s,w,gamma)
#m is mass of cannibal in Mev
arh=arh2
afz=afz2

csfz=np.sqrt(c2s(20*afz2))*20
lambda1=np.real(3*csfz/2/np.sqrt(2)*(lambertw(2*(0.594/3/csfz*adom2/afz2)**(2/3)))**(3/2))
apk=lambda1*afz2
kpk_dom=apk*Ht(apk)

kpk_fz=(3/2*(1+w(2*afz2))/c2s(2*afz2))**0.5*(2*afz2*Ht(2*afz2))/1.4
if adom2>2*afz2:
    kpk=kpk_dom
else:
    kpk=kpk_fz
print(kpk)

K=kpk*np.concatenate((np.arange(0.6,0.95,0.02),np.arange(0.95,1.05,0.01),np.arange(1.05,1.4,0.02)))  #10**np.arange(log(0.6*kpk)/log(10),log(kpk*1.4)/log(10),0.01)
# K=kpk*np.arange(0.3,0.7,0.02)
print(K.shape[0])
delta_dmf=np.zeros(K.shape[0])
delta_dm_primef=np.zeros(K.shape[0])
aend=np.zeros(K.shape[0])
j=0
kosc=1*Ht(1)

for k in K:
    if k>1:
        ahz=optimize.fsolve(lambda a: a*Ht(a)-k,1/k)[0]
    else:
        ahz=optimize.fsolve(lambda a: a*Ht(a)-k,1)[0]
    if ahz<arh/100:#For modes entering the horizon before reheating we completely ignore radiation perturbation
        astart=min(1,ahz/10)
        delta_sim0=[1,-2,k**2/2/astart/Ht(astart),-2/(1+1/3),k**2/2/astart/Ht(astart)]
        if k<kosc:#will take into account cannibal perturbations
            delta_sim=solve_ivp(lambda a,y: ddelsim(a,y,k),[astart,5*arh],delta_sim0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5)#,atol=1e-6,rtol=1e-5
            delta_dmf[j]=delta_sim.y[3,-1] #find final DM_density
            delta_dm_primef[j]=(delta_sim.y[3,-1]-delta_sim.y[3,-2])/(delta_sim.t[-1]-delta_sim.t[-2])#find final derivative of DM density
            aend[j]=delta_sim.t[-1] #find time when we end the simulation
        else:# will ignore cannibal perturbations after 100ahz
            delta_sim=solve_ivp(lambda a,y: ddelsim(a,y,k),[astart,100*ahz],delta_sim0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5)
            deltasupersim0=[delta_sim.y[3,-1],delta_sim.y[4,-1]]#initial condition deep in subhorizon for just DM
            deltasupersim=solve_ivp(lambda a,y: ddel_super_sim(a,y,k),[delta_sim.t[-1],5*arh],deltasupersim0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5)#ignore radiation, cannibal and metric perturbations completely
            delta_dmf[j]=deltasupersim.y[0,-1] #find final DM_density
            delta_dm_primef[j]=(deltasupersim.y[0,-1]-deltasupersim.y[0,-2])/(deltasupersim.t[-1]-deltasupersim.t[-2])#find final derivative of DM density
            aend[j]=deltasupersim.t[-1] #find time when we end the simulation
    else: #for modes entering the horizon close to reheating we solve the full equation till reheating occurs and then ignore cannibal
        astart=1
        a_end=max(100*arh,100*ahz)
        delta0=[1,-2,k**2/2,-2/(1+w(astart)),k**2/2,-2,k**2/2]
        delta=solve_ivp(lambda a,y: ddel(a,y,k),[astart,5*arh],delta0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5)#,atol=1e-6,rtol=1e-5
        delta_post_rh0=[delta.y[0,-1],delta.y[3,-1],delta.y[4,-1],delta.y[5,-1],delta.y[6,-1]]
        delta_post_rh=solve_ivp(lambda a,y: ddel_post_rh(a,y,k),[delta.t[-1],a_end],delta_post_rh0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5)
        delta_dmf[j]=delta_post_rh.y[1,-1]
        delta_dm_primef[j]=(delta_post_rh.y[1,-1]-delta_post_rh.y[1,-2])/(delta_post_rh.t[-1]-delta_post_rh.t[-2])
        aend[j]=delta_post_rh.t[-1]
    j=j+1
    print(j)
A=aend*delta_dm_primef
B=1/(aend)*np.exp(delta_dmf/A)
ahoru=(5*arh)**2*Ht(5*arh)/K #finding horizon entry of DM perturbation in a universe without cannibal
relative_delta_DM=A/9.11*(1+(np.log(ahoru/0.594/aend)+delta_dmf/A)/np.log(4*0.594*np.exp(-3)*aeq/ahoru))#finding ratio of DM density wrt t scenaario where there was no cannibal and only radiation domination in the early universe

delta_DMs=9.11*log(4*0.529*exp(-3)*aeq/ahoru)
matter_power=relative_delta_DM*delta_DMs

i_pk=np.where(abs(relative_delta_DM)==max(abs(relative_delta_DM)))[0][0]
i_err=np.where(abs(relative_delta_DM)>0.94*max(abs(relative_delta_DM)))[0]
kpk_num=K[i_pk]#/hor_i

i_pk2=np.where(abs(matter_power)==max(abs(matter_power)))[0][0]
kpk_num2=K[i_pk2]

plt.figure(1)
miny=0.1
maxy=5*np.max(abs(relative_delta_DM))
plt.semilogx(K,abs(relative_delta_DM))
plt.semilogx([kpk,kpk],[miny,maxy],'--k')
plt.text(1.1*kpk,2*miny,r"$k_{pk}$")
plt.semilogx([K[i_pk],K[i_pk]],[miny,maxy],'--r')
plt.semilogx([K[i_pk2],K[i_pk2]],[miny,maxy],'--b')
plt.ylim([miny,maxy])
plt.xlabel(r"$k/k_{hor,i}$")
plt.ylabel(r"$|T(k)|$",rotation=90, labelpad=0)
    
print(abs(kpk_num-kpk)/kpk_num*100)