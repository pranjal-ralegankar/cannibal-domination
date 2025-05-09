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

afz=5*10**3 #scale factor at which cannibal reactions freezeout and cannibal evolves as CDM
adom=100*afz
arh=1000*adom #scale at which we want reheating to happen
Trh=5 #in MeV

m,xi,alpha_can,afz2,adom2,arh2,rhocan,rhor,X,w,c2s,Ht,rhodm,hor_i,aeq,gamma,H1=bck2(afz,adom,arh,Trh,1)
ddel,ddelsim,ddel_super_sim,ddel_post_rh,ddel_r=pert(rhocan,rhor,rhodm,Ht,c2s,w,gamma)
#m is mass of cannibal in Mev
arh=arh2
afz=afz2

csfz=np.sqrt(c2s(20*afz2))*20
lambda1=np.real(3*csfz/2/np.sqrt(2)*(lambertw(2*(0.594/3/csfz*adom2/afz2)**(2/3)))**(3/2))
apk=lambda1*afz2
kpk_dom=apk*Ht(apk)
ahoru_pk=(5*arh)**2*Ht(5*arh)/(apk*Ht(apk))
peak=3/2*np.log(4*0.594*np.exp(-3)*adom/apk)*arh/adom*1.29*(1+np.log(ahoru_pk*1.66/0.594/arh)/np.log(4*0.594*np.exp(-3)*aeq/ahoru_pk))

kpk_fz=(3/2*(1+w(2*afz2))/c2s(2*afz2))**0.5*(2*afz2*Ht(2*afz2))/1.4
if adom2>2*afz2:
    kpk=kpk_dom
else:
    kpk=kpk_fz

k=kpk
ahz=apk
astart=min(1,ahz/10)
delta0=[1,-2,k**2/2/astart/Ht(astart),-2/(1+w(astart)),k**2/2/astart/Ht(astart),-2,k**2/2/astart/Ht(astart)]
delta=solve_ivp(lambda a,y: ddel(a,y,k),[astart,100*ahz],delta0,method='BDF',dense_output=True,atol=1e-8,rtol=1e-7)# Complete perturbation equations solving till 100ahz

deltasim0=[delta.y[0,-1],delta.y[1,-1],delta.y[2,-1],delta.y[3,-1],delta.y[4,-1]]#initial condition  deep in subhorizon
deltasim=solve_ivp(lambda a,y: ddelsim(a,y,k),[delta.t[-1],5*arh],deltasim0,method='BDF',dense_output=True,atol=1e-6,rtol=1e-5)#ignore radiation perturbations completely
A=deltasim.t[-1]*(deltasim.y[3,-1]-deltasim.y[3,-2])/(deltasim.t[-1]-deltasim.t[-2]) #Find A and B for dm density evolving as \delta=A*log(B*a) after reheating
B=1/(deltasim.t[-1])*np.exp(deltasim.y[3,-1]/A)#finding B
deltadm_post_rh=A*np.log(B*np.arange(deltasim.t[-1],10*arh,arh))# finding \delta_DM from arh to 1000arh using \delta=A*log(B*a)
delta_dm=np.append(np.append(delta.y[3,:],deltasim.y[3,:]),deltadm_post_rh[:])# joinging the post reheating DM density evolution with pre-reheating
a_full=np.append(np.append(delta.t[:],deltasim.t[:]),np.arange(deltasim.t[-1],10*arh,arh))
delta_r=delta.y[5,:]
a_delta_r=delta.t
delta_can=np.append(delta.y[1,:],deltasim.y[1,:])
a_delta_can=np.append(delta.t[:],deltasim.t[:])
aend=deltasim.t[-1]
delta_dmf=deltasim.y[3,-1]
ahoru=(5*arh)**2*Ht(5*arh)/k #finding horizon entry of DM perturbation in a universe without cannibal
relative_delta_DM=A/9.11*(1+(np.log(ahoru/0.594/aend)+delta_dmf/A)/np.log(4*0.594*np.exp(-3)*aeq/ahoru))#finding ratio of DM density wrt t scenaario where there was no cannibal and only radiation domination in the early universe

delta_lin=3/2*9.11*log(4*0.594*exp(-3)*adom/apk)*a_full/adom
delta_log=3/2*9.11*log(4*0.594*exp(-3)*adom/apk)*arh/adom*1.29*log(1.66*a_full/arh)
print("b1=",A/(3/2*9.11*log(4*0.594*exp(-3)*adom/apk)*arh/adom))
print("b2=",B*arh)

# lambda_J=lambda a:(3/2*(1+w(a))/c2s(a))**-0.5*(a*Ht(a))**-1
# lambda_H=lambda a:(a*Ht(a))**-1
# lambda_Hu=lambda a:(10*arh*Ht(10*arh))**-1*a/(10*arh)
# rs=lambda a: integrate.quad(lambda x: c2s(x)**0.5*lambda_H(x)/x, 0.01, a)[0]
# aj_ent=optimize.fsolve(lambda a: 1/lambda_J(a)-k,ahz)[0]
# aj_esc=optimize.fsolve(lambda a: 1/lambda_J(a)-k,adom)[0]

# aplt=10**np.arange(0,np.log(12*arh)/np.log(10),0.1) #end point corresponds to last mode we have evaluated for rel DM growth
# aplt2=10**np.arange(0,np.log(arh2)/np.log(10),0.1)
# lambda_Htable=table(lambda_H,aplt)
# lambda_Jtable=table(lambda_J,aplt2)
# lambda_Hutable=table(lambda_Hu,aplt)
# rs_table=table(rs,aplt2)
# miny_hor=lambda_H(aplt[0])
# maxy_hor=lambda_H(aplt[-1])

# plt.rcParams.update({'font.size': 20})
# abreak=np.where(aplt2>adom2)[0][0]

# ax1=plt.subplot(211)
# plt.loglog(aplt,lambda_Htable,label=r"Horizon")
# plt.loglog(aplt2[0:abreak],lambda_Jtable[0:abreak],'--',linewidth=1, color='darkorange')
# #plt.loglog(aplt2[0:abreak],1/k_osctable[0:abreak],'-',color='darkgoldenrod',label=r"$k_{osc}^{-1}$")
# plt.loglog(aplt2[0:abreak],rs_table[0:abreak],'-',color='darkgoldenrod',label=r"$r_s$")
# plt.loglog(aplt2[abreak:],rs_table[abreak:],'--',linewidth=1,color='darkgoldenrod')
# plt.loglog(aplt2[abreak:-1],lambda_Jtable[abreak:-1],color='darkorange',label=r"$k_J^{-1}$")
# #plt.loglog(aplt,lambda_Hutable,'--b',linewidth=0.5,label=r"$\Lambda$CDM Horizon")
# plt.loglog([ahz,ahz],[miny_hor,maxy_hor],'-.k')
# plt.text(1.1*ahz,2*miny_hor,r"$a_{hor}$")
# plt.loglog([100,100],[miny_hor,maxy_hor],'--r')
# plt.text(100/5,2*miny_hor,r"$a_{can}$")
# plt.loglog([afz2,afz2],[miny_hor,maxy_hor],'--g')
# plt.text(1.1*afz2,2*miny_hor,r"$a_{fz}$")
# plt.loglog([adom2,adom2],[miny_hor,maxy_hor],'--',color='purple')
# plt.text(1.1*adom2,2*miny_hor,r"$a_{dom}$")
# plt.loglog([arh2,arh2],[miny_hor,maxy_hor],'--b')
# plt.text(1.1*arh2,2*miny_hor,r"$a_{rh}$")
# plt.loglog([aj_ent,aj_ent],[miny_hor,maxy_hor],'-.',color='grey',linewidth=1)
# plt.loglog([aj_esc,aj_esc],[miny_hor,maxy_hor],'-.',color='grey',linewidth=1)
# plt.loglog([aplt[0],aplt[-1]],[k**-1,k**-1],'--k')  
# plt.text(aplt[0]*2,1.1*k**-1,r"$k^{-1}$")
# plt.fill_between(aplt, lambda_Htable, miny_hor, facecolor='blue', alpha=0.1)
# plt.fill_between(aplt2[0:abreak], rs_table[0:abreak], miny_hor, facecolor='yellow', alpha=0.5)
# plt.fill_between(aplt2[abreak-1:-1], lambda_Jtable[abreak-1:-1], miny_hor, facecolor='yellow', alpha=0.5)
# plt.ylim([miny_hor,maxy_hor])
# plt.xlim([aplt[0],10*arh])
# #plt.xlabel(r"$a$")
# plt.ylabel(r"$\frac{k_{hor,i}}{k}$",rotation=0, labelpad=18)
# plt.legend()

# ax3=plt.subplot(212,sharex=ax1)
# miny=3*10**-1
# maxy=arh/adom*log(adom/afz)*50
# plt.figure(1)
# plt.rcParams.update({'font.size': 20})
# plt.loglog(a_delta_can,abs(delta_can),label="Cannibal",color='red')
# plt.loglog(a_full,abs(delta_dm),label="DM",color='blue')
# plt.loglog(a_full,abs(delta_lin),label="linear",color='green')
# plt.loglog(a_full,abs(delta_log),label="log",color='orange')
# plt.loglog([100,100],[miny,maxy],'--r')
# plt.text(100/5,2*miny,r"$a_{can}$")
# plt.loglog([afz2,afz2],[miny,maxy],'--g')
# plt.text(1.1*afz2,2*miny,r"$a_{fz}$")
# plt.loglog([arh2,arh2],[miny,maxy],'--b')
# plt.text(1.1*arh2,2*miny,r"$a_{rh}$")
# plt.loglog([ahz,ahz],[miny,maxy],'-.k')
# plt.text(1.1*ahz,2*miny,r"$a_{hor}$")
# plt.loglog([aj_ent,aj_ent],[miny,maxy],'-.',color='grey',linewidth=1)
# plt.loglog([aj_esc,aj_esc],[miny,maxy],'-.',color='grey',linewidth=1)
# #plt.loglog([acan,acan],[miny,maxy],'--r')
# #plt.text(acan,2*miny,r"$a_{can}$")
# #ahz=optimize.fsolve(lambda a: a*Ht(a)-k,10)[0] #finding scale factor at which mode k enters horizon
# #plt.loglog([ahz,ahz],[miny,maxy],'--k')
# #plt.text(ahz,2*miny,r"$a_{hz}$")
# plt.loglog([adom2,adom2],[miny,maxy],'--',color='purple')
# plt.text(1.1*adom2,2*miny,r"$a_{dom}$")
# plt.ylim([miny,maxy])
# plt.xlabel(r"$a/a_i$")
# plt.ylabel(r"$\frac{|\delta|}{\phi_P}$",rotation=0, labelpad=10)
# plt.legend(bbox_to_anchor=(0, 0.4),loc='center left')
# plt.show()

# plt.gcf().set_size_inches(10, 10,forward=True)
# plt.gcf().subplots_adjust(left=0.13, right=0.95, bottom=0.08, top=0.98)
peak2=log(adom/10/afz)*arh/adom
print(peak2/relative_delta_DM)
