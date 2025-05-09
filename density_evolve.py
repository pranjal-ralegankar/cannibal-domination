# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 10:57:30 2019

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

os.chdir('../../pranjal work/git-python/cannibal-base-code')
from background import bck, bck2
from equilibrium import wfun, Xcan
os.chdir('../../../draft/cannibal-long_paper_figs')
def table(f,t):
    tablef=np.zeros(t.shape[0])
    for i in np.arange(t.shape[0]):
        tablef[i]=f(t[i])
    return tablef



afz=10**3 #scale factor at which cannibal reactions freezeout and cannibal evolves as CDM
arh=10**6 #scale at which we want reheating to happen
Trh=10 #in Mev
xi=1
# =============================================================================
#finding dimensionless densities, cannibal thermo quantities and particle parameters from solving background equations
start = time.time()
m,xi,alpha_can,afz2,adom2,arh2,rhocan,rhor,X,w,c2s,Ht,rhodm,hor_i,aeq,gamma,H1=bck(afz,xi,arh,Trh,2)
print("it took", time.time() - start, "seconds.")
#m is mass of cannibal in Mev
#xi is 10*m divided by initial SM temperature
#alpha_can is effective "fine structure constant" of cannibal self interactions
#afz2,adom2,arh2 are scale factors relative to a_i
#rhocan, rhor and rhodm are cannibal, SM radiation and DM density relative to rhocan(a_i)
#X is m/T_can
#w and c2s are cannibal eqn of state and sound speed
#Ht is Hubble relative to H(a_i)
#hor_i is the horizon size at a_i: (a_i*H(a_i))^-1
# =============================================================================
#params
Mpl=2.435*10**21 #in MeV
sigmav2_can=25*sqrt(5)*pi**2*alpha_can**3/5184/m**5 # cannibal 3 to 2 anihilation rate in Mev^-5

#plotting densities
aplt=10**np.arange(0,np.log(10*arh)/np.log(10),0.1)
rhocan_table=np.zeros(aplt.shape[0])
rhor_table=np.zeros(aplt.shape[0])
rhodm_table=np.zeros(aplt.shape[0])
hor_table=np.zeros(aplt.shape[0])
for i in np.arange(0,aplt.shape[0]):
    rhocan_table[i]=rhocan(aplt[i])
    rhor_table[i]=rhor(aplt[i])
    hor_table[i]=(aplt[i]*Ht(aplt[i]))**-1
    rhodm_table[i]=rhodm(aplt[i])

plt.rcParams.update({'font.size': 20})
fig, ax1 = plt.subplots()
miny=rhodm(5*arh)/100
maxy=10**3*max(rhor_table[0],rhocan_table[0])
ax1.loglog(aplt,rhocan_table,label=r"cannibal",color='red')
ax1.loglog(aplt,rhor_table,label="SM",color='orange')
ax1.loglog(aplt,rhodm_table,label="DM",color='blue')
ax1.loglog([afz2,afz2],[miny,maxy],'--g')
ax1.text(1.1*afz2,3*miny,r"$a_{fz}$")
ax1.loglog([arh2,arh2],[miny,maxy],'--b')
ax1.text(1.1*arh2,3*miny,r"$a_{rh}$")
ax1.loglog([adom2,adom2],[miny,maxy],'--p')
ax1.text(1.1*adom2,3*miny,r"$a_{dom}$")
plt.loglog([100,100],[miny,maxy],'--r')
plt.text(1.1*100,2*miny,r"$a_{can}$")
plt.fill_between([100,afz2], maxy, miny, facecolor='red', alpha=0.1)
plt.fill_between([adom2,arh2], maxy, miny, facecolor='yellow', alpha=0.1)
ax1.set_xlabel(r"$a/a_i$")
ax1.set_ylabel(r"$\frac{\rho}{\rho_{can,i}}$",rotation=0, labelpad=16)
ax1.set_ylim([miny,maxy])
ax1.set_xlim([aplt[0],5*arh])
ax1.legend() 

ax2 = plt.axes([0,0,1,1])
ip = InsetPosition(ax1, [0.6,0.52,0.35,0.4])
ax2.set_axes_locator(ip)
mark_inset(ax1, ax2, loc1=3, loc2=1, fc="none", ec='0.5')
miny=rhor(3*arh)
maxy=rhocan(arh/5)
ax2.loglog([aplt[0],aplt[-1]],[gamma**2*(1+rhor(1)),gamma**2*(1+rhor(1))],'--k',alpha=0.7,label=r"$3\Gamma^2M_{pl}^2/\rho_{\phi,I}$")
ax2.loglog(aplt,rhocan_table,label=r"cannibal",color='red')
ax2.loglog(aplt,rhor_table,label="SM",color='orange')
ax2.loglog(aplt,table(lambda x: rhor(5*arh)*(5*arh/x)**4,aplt),'-.',color='orange',label="SM extrapolate")
ax2.loglog([arh2,arh2],[miny,maxy],'--b')
ax2.text(1.05*arh2,2*miny,r"$a_{rh}$")
#ax2.set_xlabel(r"$a$")
#ax2.set_ylabel(r"$\frac{\rho}{\rho_{\phi,I}}$",rotation=0, labelpad=10)
ax2.set_ylim([miny,maxy])
ax2.set_xlim([arh/5,3*arh])
ax2.set_xticklabels([])
ax2.set_yticklabels([])
#ax2.legend() 

custom_lines = [Line2D([0], [0], color='red', linestyle='-'),
                Line2D([0], [0], color='orange', linestyle='-'),
                Line2D([0], [0], color='blue', linestyle='-'),
                Line2D([0], [0], color='orange', linestyle='-.'),
                Line2D([0], [0], color='k', linestyle='--'),]
labels = ['cannibal','SM','DM','SM extrapolated', r"$3\Gamma^2M_{pl}^2/\rho_{can,i}$"]
ax1.legend(custom_lines, labels, loc='lower left')

plt.gcf().set_size_inches(10, 6,forward=True)
plt.gcf().subplots_adjust(left=0.15, right=0.98, bottom=0.13, top=0.98)

mycwd = os.getcwd()
os.chdir("..")
plt.savefig("density_evolve.pdf")
os.chdir(mycwd)

# =============================================================================
#plotting w and c2s
aplt=(10**np.arange(0.1,np.log(2*arh2)/np.log(10),0.1))
w_eq_num=lambda a: wfun(Xcan(a))
def c2s_eq_num(a):
#    if c2sfit[0,0]<=a<fz.t[0]:
#        ans=c2stemp(a)
#    elif fz.t[0]<=a<=fz.t[-2]:
#        ans=np.interp(a,fz.t,c2s_fz)
#    elif a<c2sfit[0,0]:
#        ans=1/3
#    else:
#        ans=c2s_fz[-2]*fz.t[-2]**2/a**2 #setting c2s\propto 1/a^2 after table ends
    if a>3:
        ans=w_eq_num(a)-a*derivative(w_eq_num,a)/(1+w_eq_num(a))/3
    elif 1<a<3:
        ans=w_eq_num(a)-a*(w_eq_num(a)-w_eq_num(a-0.01))/0.01/(1+w_eq_num(a))/3
    return ans
w_eqnum=table(w_eq_num,aplt)
c2s_eqnum=table(c2s_eq_num,aplt)
c2s_fznum=table(c2s,aplt)
w_fznum=table(w,aplt)

i_fz=np.where(aplt>30*afz2)[0][0]

plt.figure(2)
miny_cs=10**-7
maxy_cs=1
plt.rcParams.update({'font.size': 20})
plt.loglog(aplt,c2s_fznum,color="black",label=r"$c^2_s$")
plt.loglog(aplt,w_fznum,color="blue",label=r"$w$")
plt.loglog(aplt,c2s_eqnum,'--',color="red",label=r"$c^2_s$ equilibrium")
plt.loglog(aplt,w_eqnum,'--',color="orange",label=r"$w$ equilibrium")
plt.loglog(aplt,w_fznum[i_fz]*(aplt[i_fz]/aplt[:])**2,'-.',color="skyblue",label=r"$w_{c,fz}a_{fz}^2/a^2$")
plt.loglog(aplt,c2s_fznum[i_fz]*(aplt[i_fz]/aplt[:])**2,'-.',color="grey",label=r"$c^2_{s,fz}a_{fz}^2/a^2$")
plt.loglog([afz2,afz2],[miny_cs,maxy_cs],'--g')
plt.text(1.02*afz2,2*miny_cs,r"$a_{fz}$")
plt.loglog([arh2,arh2],[miny_cs,maxy_cs],'--b')
plt.text(0.5*arh2,2*miny_cs,r"$a_{rh}$")
plt.loglog([adom2,adom2],[miny_cs,maxy_cs],'--p')
plt.text(1.1*adom2,3*miny_cs,r"$a_{dom}$")
plt.loglog([100,100],[miny_cs,maxy_cs],'--r')
plt.text(1.1*100,2*miny_cs,r"$a_{can}$")
plt.fill_between([100,afz2], maxy_cs, miny_cs, facecolor='red', alpha=0.1)
plt.fill_between([adom2,arh2], maxy_cs, miny_cs, facecolor='yellow', alpha=0.1)
plt.ylim([miny_cs,maxy_cs])
plt.xlim([aplt[0],aplt[-1]])
plt.legend()
plt.xlabel(r"$a/a_i$")
plt.ylabel(r"$c^2_s$, $w$",rotation=90,labelpad=10)
plt.gcf().set_size_inches(10, 6,forward=True)
plt.gcf().subplots_adjust(left=0.13, right=0.97, bottom=0.12, top=0.98)
plt.show()

# mycwd = os.getcwd()
# os.chdir("..")
# plt.savefig("c2s_w_fz.pdf")
# os.chdir(mycwd)

#aplt=(10**np.arange(0.1,np.log(3*arh)/np.log(10),0.1))
#c2s_fznum=table(c2s,aplt)
#w_fznum=table(w,aplt)
#plt.figure(2)
#plt.rcParams.update({'font.size': 20})
#plt.loglog(aplt,c2s_fznum,color="black",label=r"$c^2_s$")
#plt.loglog(aplt,w_fznum,color="blue",label=r"$w$")
#plt.loglog(aplt,w(30*afz)*(30*afz/aplt[:])**2,'-.',color="skyblue",label=r"$w_{c,fz}a_{fz}^2/a^2$")
#plt.loglog(aplt,c2s(30*afz)*(30*afz/aplt[:])**2,'-.',color="grey",label=r"$c^2_{s,fz}a_{fz}^2/a^2$")
#plt.loglog([afz2,afz2],[0,0.4],'--g')
#plt.text(1.02*afz2,0.02,r"$a_{fz}$")
#plt.loglog([arh2,arh2],[0,0.4],'--b')
#plt.text(1.02*arh2,0.02,r"$a_{rh}$")
#plt.ylim([0,0.35])
#plt.xlim([aplt[0],aplt[-1]])
#plt.legend()
#plt.xlabel(r"$a/a_i$")
#plt.ylabel(r"$c^2_s$ or $w$",rotation=90,labelpad=10)
#plt.gcf().set_size_inches(10, 6,forward=True)
#plt.gcf().subplots_adjust(left=0.13, right=0.97, bottom=0.12, top=0.98)
#plt.show()

#mycwd = os.getcwd()
#os.chdir("..")
#plt.savefig("c2s_fz.pdf")
#os.chdir(mycwd)

#plt.figure(3)