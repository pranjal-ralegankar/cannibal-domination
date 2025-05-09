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

os.chdir('../../pranjal work/git-python/cannibal-base-code')
from background import bck, bck2
from equilibrium import wfun, Xcan
os.chdir('../../../draft/cannibal-long_paper_figs')
def table(f,t):
    tablef=np.zeros(t.shape[0])
    for i in np.arange(t.shape[0]):
        tablef[i]=f(t[i])
    return tablef

acan=100 #effectively when cannibal phase starts; T=0.1m
afz=10**5 #scale factor at which cannibal reactions freezeout and cannibal evolves as CDM
arh=10**9 #scale at which we want reheating to happen
xi=10 #T_can(a_i)/T_SM(a_i)
Trh=10 #in Mev

m,xi,alpha_can,afz2,adom2,arh2,rhocan,rhor,X,w,c2s,Ht,rhodm,hor_i,aeq,gamma,H1=bck2(afz,xi,arh,Trh,2)

aspan=10**np.arange(0,log(3*arh)/log(10),0.1)
#plotting densities
plt.rcParams.update({'font.size': 22})
plt.figure(1)
miny=10**-3
maxy=10
plt.loglog(aspan[1:],1/table(X,aspan[1:]),color="blue",label="numerical")
aspan_rad=10**np.arange(0,log(10**2)/log(10),0.1)
plt.loglog(aspan_rad,table(lambda x: 10/x,aspan_rad),'--',color="orange",label=r"radiation")
aspan_can=10**np.arange(log(100)/log(10),log(10*afz)/log(10),0.1)
plt.loglog(aspan_can,table(lambda x: 1/3/log(x/25.6),aspan_can),'--',color="red",label=r"MB cannibalism")
plt.loglog([afz2,afz2],[miny,maxy],'--g')
plt.text(1.1*afz2,2*miny,r"$a_{fz}$")
plt.loglog([100,100],[miny,maxy],'--r')
plt.text(1.1*100,2*miny,r"$a_{can}$")
plt.fill_between([100,afz2], maxy, miny, facecolor='red', alpha=0.1)
plt.legend()
plt.xlabel(r"$a/a_i$")
plt.ylabel(r"$T/m$",rotation=90, labelpad=5)
plt.ylim([miny,maxy])
plt.xlim([1,afz*10])
plt.gcf().set_size_inches(10, 6,forward=True)
plt.gcf().subplots_adjust(left=0.12, right=0.97, bottom=0.12, top=0.97)
plt.show()

#mycwd = os.getcwd()
#os.chdir("..")
#plt.savefig("temp_intro.pdf")
#os.chdir(mycwd)

plt.rcParams.update({'font.size': 22})
plt.figure(2)
miny=10**1
maxy=10**4
plt.loglog(aspan[1:],pi**2/30*(10)**4*aspan[1:]**3*table(rhocan,aspan[1:]),color="blue",label="numerical")
aspan_rad=10**np.arange(0,log(10**2)/log(10),0.1)
plt.loglog(aspan_rad,table(lambda x: pi**2/30*(10/x)**4*x**3,aspan_rad),'--',color="orange",label="radiation")
aspan_can=10**np.arange(log(100)/log(10),log(100*afz)/log(10),0.1)
plt.loglog(aspan_can,table(lambda x: 147.7/(x**3*log(x/25.6))*x**3*(1-1/3/log(x/25.6)),aspan_can),'--',color="red",label="MB cannibalism")
plt.loglog([afz2,afz2],[miny,maxy],'--g')
plt.text(1.1*afz2,2*miny,r"$a_{fz}$")
plt.loglog([100,100],[miny,maxy],'--r')
plt.text(1.1*100,2*miny,r"$a_{can}$")
plt.fill_between([100,afz2], maxy, miny, facecolor='red', alpha=0.1)
plt.legend()
plt.xlabel(r"$a/a_i$")
plt.ylabel(r"$(\rho a^3)/m^4$",rotation=90, labelpad=5)
plt.ylim([miny,maxy])
plt.xlim([1,afz*100])
plt.gcf().set_size_inches(10, 6,forward=True)
plt.gcf().subplots_adjust(left=0.12, right=0.97, bottom=0.12, top=0.97)
plt.show()

#mycwd = os.getcwd()
#os.chdir("..")
#plt.savefig("rhocan_intro.pdf")
#os.chdir(mycwd)
