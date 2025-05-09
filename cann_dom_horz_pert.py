# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 12:19:31 2019

@author: pranj
"""

import matplotlib.pyplot as plt
from numpy import exp, sqrt, pi, inf, sinh, log, arange
import numpy as np
from scipy import integrate, optimize
from scipy.interpolate import interp1d
from scipy.integrate import odeint, solve_ivp, ode
from scipy.misc import derivative
import os
from scipy.special import kn
import pandas as pd

os.chdir('../../pranjal work/git-python/cannibal-base-code')
from background import bck, bck2
from equilibrium import wfun, Xcan
from perturbation import pert, pert_wgamma
os.chdir('../../../draft/cannibal-long_paper_figs')
def table(f,t):
    tablef=np.zeros(t.shape[0])
    for i in np.arange(t.shape[0]):
        tablef[i]=f(t[i])
    return tablef

acan=681.936575 #effectively when cannibal phase starts; T=0.1m
afz=10**4/5 #scale factor at which cannibal reactions freezeout and cannibal evolves as CDM
arh=10**10/2 #scale at which we want reheating to happen
xi=500
Trh=5 #in MeV

m,xi,alpha_can,afz2,adom2,arh2,rhocan,rhor,X,w,c2s,Ht,rhodm,hor_i,aeq,gamma,H1=bck(afz,xi,arh,Trh,2)

ddel,ddelsim,ddel_super_sim,ddel_post_rh,ddel_r=pert_wgamma(rhocan,rhor,rhodm,Ht,c2s,w,gamma)

# =============================================================================
# calculating all density perturbations for specific mode
k=(3/2*(1+w(2*afz))/c2s(2*afz))**0.5*(2*afz*Ht(2*afz))/1.2*30#afz/20*Ht(afz/20)##(3/2*(1+w(2*afz))/c2s(2*afz))**0.5*(2*afz*Ht(2*afz))/1.2#afz/2*Ht(afz/2)
k=1/12
ahz=optimize.fsolve(lambda a: a*Ht(a)-k,10)[0]

astart=min(1,ahz/10)
delta0=[1,-2,k**2/2/astart/Ht(astart),-2/(1+w(astart)),k**2/2/astart/Ht(astart)] #super-horizon initial condition
delta=solve_ivp(lambda a,y: ddelsim(a,y,k),[astart,5*arh],delta0,method='BDF',dense_output=True,atol=1e-6,rtol=1e-5)#ignore radiation perturbations completely
A=delta.t[-1]*(delta.y[3,-1]-delta.y[3,-2])/(delta.t[-1]-delta.t[-2]) #Find A and B for dm density evolving as \delta=A*log(B*a) after reheating
B=1/(delta.t[-1])*np.exp(delta.y[3,-1]/A)#finding B
deltadm_post_rh=A*np.log(np.arange(delta.t[-1],5*arh,arh)/delta.t[-1])+delta.y[3,-1]# finding \delta_DM from arh to 1000arh using \delta=A*log(B*a)
delta_dm=delta.y[3,:]#np.append(delta.y[3,:],deltadm_post_rh[:])# joinging the post reheating DM density evolution with pre-reheating
a_full=delta.t[:]#np.append(delta.t[:],np.arange(delta.t[-1],1000*arh,arh))
print("main perturbation done")

delta0_r=[-2,k**2/2/astart/Ht(astart)] #super-horizon initial condition
delta_r=solve_ivp(lambda a,y: ddel_r(a,y,k,delta.sol),[astart,1*arh],delta0_r,method='Radau',dense_output=False,atol=1e-8,rtol=1e-7)
delta0_r2=[delta_r.y[0,-1],delta_r.y[1,-1]]
print("arh")
delta_r2=solve_ivp(lambda a,y: ddel_r(a,y,k,delta.sol),[1*arh,5*arh],delta0_r2,method='BDF',dense_output=False,atol=1e-10,rtol=1e-10)
delta_ry=np.append(delta_r.y[0,:],delta_r2.y[0,:])
delta_r.t=np.append(delta_r.t,delta_r2.t)

lambda_J=lambda a:(3/2*(1+w(a))/c2s(a))**-0.5*(a*Ht(a))**-1
lambda_H=lambda a:(a*Ht(a))**-1
lambda_Hu=lambda a:(10*arh*Ht(10*arh))**-1*a/(10*arh)
aj_ent=optimize.fsolve(lambda a: 1/lambda_J(a)-k,ahz)[0]
aj_esc=optimize.fsolve(lambda a: 1/lambda_J(a)-k,2*afz)[0]

aplt=10**np.arange(0,np.log(12*arh)/np.log(10),0.1) #end point corresponds to last mode we have evaluated for rel DM growth
aplt2=10**np.arange(0,np.log(arh)/np.log(10),0.1)
lambda_Htable=table(lambda_H,aplt)
lambda_Jtable=table(lambda_J,aplt2)
lambda_Hutable=table(lambda_Hu,aplt)
miny_hor=lambda_H(aplt[0])
maxy_hor=lambda_H(aplt[-1])

# =============================================================================
# Plotting all density perturbations
plt.rcParams.update({'font.size': 20})

ax1=plt.subplot(211)
plt.loglog(aplt,lambda_Htable,label=r"Horizon")
plt.loglog(aplt2,lambda_Jtable,label=r"Jeans")
#plt.loglog(aplt,lambda_Hutable,'--b',linewidth=0.5,label=r"usual Horizon")
plt.loglog([ahz,ahz],[miny_hor,maxy_hor],'-.k')
plt.text(1.1*ahz,2*miny_hor,r"$a_{hor}$")
plt.loglog([100,100],[miny_hor,maxy_hor],'--r')
plt.text(1.1*100,2*miny_hor,r"$a_{can}$")
plt.loglog([afz2,afz2],[miny_hor,maxy_hor],'--g')
plt.text(1.1*afz2,2*miny_hor,r"$a_{fz}$")
plt.loglog([arh2,arh2],[miny_hor,maxy_hor],'--b')
plt.text(1.1*arh2,2*miny_hor,r"$a_{rh}$")
plt.loglog([aj_ent,aj_ent],[miny_hor,maxy_hor],'-.',color='grey',linewidth=1)
plt.loglog([aj_esc,aj_esc],[miny_hor,maxy_hor],'-.',color='grey',linewidth=1)
plt.loglog([aplt[0],aplt[-1]],[k**-1,k**-1],'--k')  
plt.text(aplt[0]*2,1.1*k**-1,r"$k^{-1}$")
plt.fill_between(aplt, lambda_Htable, miny_hor, facecolor='blue', alpha=0.1)
plt.fill_between(aplt2, lambda_Jtable, miny_hor, facecolor='yellow', alpha=0.5)
plt.ylim([miny_hor,maxy_hor])
plt.xlim([aplt[0],10*arh])
#plt.xlabel(r"$a$")
plt.ylabel(r"$\frac{k_{hor,i}}{k}$",rotation=0, labelpad=18)
plt.legend()

ax3=plt.subplot(212,sharex=ax1)
miny=10**-8
maxy=10**4
plt.loglog(delta_r.t,abs(delta_ry),label="Radiation",color='darkorange')
plt.loglog(delta.t[:],abs(delta.y[1,:]),label="Cannibal",color='red')
plt.loglog(a_full,abs(delta_dm),label="DM",color='blue')
plt.loglog([ahz,ahz],[miny,maxy],'-.k')
plt.text(1.1*ahz,2*miny,r"$a_{hor}$")
plt.loglog([100,100],[miny,maxy],'--r')
plt.text(1.1*100,2*miny,r"$a_{can}$")
plt.loglog([afz2,afz2],[miny,maxy],'--g')
plt.text(1.1*afz2,2*miny,r"$a_{fz}$")
plt.loglog([arh2,arh2],[miny,maxy],'--b')
plt.text(1.1*arh2,2*miny,r"$a_{rh}$")
plt.loglog([aj_ent,aj_ent],[miny,maxy],'-.',color='grey',linewidth=1)
plt.loglog([aj_esc,aj_esc],[miny,maxy],'-.',color='grey',linewidth=1)
plt.ylim([miny,maxy])
plt.xlabel(r"$a/a_i$")
plt.ylabel(r"$\frac{|\delta|}{\phi_P}$",rotation=0, labelpad=12)
plt.legend()

plt.gcf().set_size_inches(10, 10,forward=True)
plt.gcf().subplots_adjust(left=0.13, right=0.95, bottom=0.08, top=0.97)

#mycwd = os.getcwd()
#os.chdir("..")
#plt.savefig("cann_dom_horz_pert.pdf")
#os.chdir(mycwd)