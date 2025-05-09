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
from scipy.special import lambertw
import os
from scipy.special import kn

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

acan=100 #effectively when cannibal phase starts; T=0.1m
afz=afz #scale factor at which cannibal reactions freezeout and cannibal evolves as CDM
adom=adom
arh=arh #scale at which we want reheating to happen
Trh=Trh #in Mev

m,xi,alpha_can,afz2,adom2,arh2,rhocan,rhor,X,w,c2s,Ht,rhodm,hor_i,aeq,gamma,H1=bck(afz,adom,arh,Trh,1)

ddel,ddelsim,ddel_super_sim,ddel_post_rh,ddel_r=pert_wgamma(rhocan,rhor,rhodm,Ht,c2s,w,gamma)

# =============================================================================
# plotting all density perturbations for specific mode
csfz=np.sqrt(c2s(2*afz))
lambda1=np.real(3*csfz/2/np.sqrt(2)*(lambertw(2*(0.594/3/csfz*adom/afz)**(2/3)))**(3/2))
apk=lambda1*afz
kpk=apk*Ht(apk)
k=k#*20#2*afz*Ht(2*afz)
ahz=optimize.fsolve(lambda a: a*Ht(a)-k,10)[0]

#astart=min(1,ahz/10)
#delta0=[1,-2,k**2/2,-2/(1+w(astart)),k**2/2,-2,k**2/2]
#delta=solve_ivp(lambda a,y: ddel(a,y,k),[astart,100*ahz],delta0,method='BDF',dense_output=False,atol=1e-8,rtol=1e-7)# Complete perturbation equations solving till 100ahz
#
#deltasim0=[delta.y[0,-1],delta.y[1,-1],delta.y[2,-1],delta.y[3,-1],delta.y[4,-1]]#initial condition  deep in subhorizon
#deltasim=solve_ivp(lambda a,y: ddelsim(a,y,k),[delta.t[-1],5*arh],deltasim0,method='BDF',dense_output=True,atol=1e-6,rtol=1e-5)#ignore radiation perturbations completely
#A=deltasim.t[-1]*(deltasim.y[3,-1]-deltasim.y[3,-2])/(deltasim.t[-1]-deltasim.t[-2]) #Find A and B for dm density evolving as \delta=A*log(B*a) after reheating
#B=1/(deltasim.t[-1])*np.exp(deltasim.y[3,-1]/A)#finding B
#deltadm_post_rh=A*np.log(B*np.arange(deltasim.t[-1],10*arh,arh))# finding \delta_DM from arh to 1000arh using \delta=A*log(B*a)
#delta_dm=np.append(np.append(delta.y[3,:],deltasim.y[3,:]),deltadm_post_rh[:])# joinging the post reheating DM density evolution with pre-reheating
#a_full=np.append(np.append(delta.t[:],deltasim.t[:]),np.arange(deltasim.t[-1],10*arh,arh))
#delta_r=delta.y[5,:]
#a_delta_r=delta.t
#delta_can=np.append(delta.y[1,:],deltasim.y[1,:])
#a_delta_can=np.append(delta.t[:],deltasim.t[:])
#
#a_eff=optimize.fsolve(lambda a: gamma*rhocan(a)-Ht(a)*rhor(a),10)[0] #scale when decays become effective in SM radiation bath.
#delta0_r=[delta.y[5,-1],delta.y[6,-1]] #super-horizon initial condition
#deltar=solve_ivp(lambda a,y: ddel_r(a,y,k,deltasim.sol),[delta.t[-1],a_eff],delta0_r,method='BDF',dense_output=False,atol=1e-6,rtol=1e-3)
#print("aeff")
#delta0_r2=[deltar.y[0,-1],deltar.y[1,-1]]
#deltar2=solve_ivp(lambda a,y: ddel_r(a,y,k,deltasim.sol),[a_eff,5*arh],delta0_r2,method='Radau',dense_output=False,atol=1e-10,rtol=1e-10)
#delta_r=np.concatenate((delta.y[5,:],deltar.y[0,:],deltar2.y[0,:]))
#a_delta_r=np.concatenate((a_delta_r,deltar.t,deltar2.t))

#aj_esc=min(optimize.fsolve(lambda a: (3/2*(1+w(a))/c2s(a))**0.5*(a*Ht(a))-k,adom)[0],arh) #point where WKB fails

#radiation amplitude homogeneous
a_max_i_rh=optimize.fsolve(lambda a: derivative(delta.sol,a)[5]-0,15*ahz)[0]# closest a at which delta_r is at max
Dr_i=delta.sol(a_max_i_rh)[5] #value of delta_c at max to serve as initial amplitude
a_wkb_rh=10**np.arange(np.log(a_max_i_rh)/np.log(10),np.log(5*arh)/np.log(10),0.1) #array of scale factors where WKB is valid

n_r=lambda a: 2*gamma*rhocan(a)/Ht(a)/rhor(a)/a #delta' coefficient
N_r=lambda a: integrate.quad(n_r, a_max_i_rh, a)[0] #integral of n(a)dlna
Dr_h=Dr_i*table(lambda a: np.exp(-N_r(a)/2), a_wkb_rh) #full WKB amplitude

#cannibal amplitude homogeneous
a_max_i_ch=optimize.fsolve(lambda a: derivative(delta.sol,a)[1]-0,15*ahz)[0]# closest a at which delta_can is at max
Dc_i=delta.sol(a_max_i_ch)[1] #value of delta_c at max to serve as initial amplitude
a_wkb_ch=10**np.arange(np.log(a_max_i_ch)/np.log(10),np.log(aj_esc)/np.log(10),0.1) #array of scale factors within Jeans
n_c=lambda a: (1-3*w(a))/a #delta' coefficient
N_c=lambda a: integrate.quad(n_c, a_max_i_ch, a)[0] #integral of n(a)dlna
Dc_h=Dc_i*table(lambda a: np.exp(-N_c(a)/2)*(c2s(a_max_i_ch)/c2s(a))**0.25, a_wkb_ch) #full WKB amplitude

#radiation amplitude within jeans
xi= lambda a: gamma*rhocan(a)/Ht(a)/rhor(a)
F_j=lambda a: (2+w(a))/(1+w(a))*xi(a)/a*c2s(a)**0.5*k/(a**2*Ht(a))*Dc_i*np.exp(-N_c(a)/2)*(c2s(a_max_i_ch)/c2s(a))**0.25
a_wkb_rinhj= 10**np.arange(np.log(adom2)/np.log(10),np.log(aj_esc)/np.log(10),0.1)
Dr_inh_j=table(lambda a: F_j(a)/(1/3-c2s(a))*(a**2*Ht(a)/k)**2, a_wkb_rinhj)

#radiation amplitude outside jeans
F2= lambda a: ((2*a*Ht(a)+a**2*derivative(Ht,a))/(a**2*Ht(a))*xi(a)/a+(xi(a)/a)**2+xi(a)/a**2+2/a**2*rhocan(a)/(rhocan(a)+rhor(a)))*deltasim.sol(a)[1]
a_wkb_rinh=10**np.arange(np.log(aj_esc)/np.log(10),np.log(10*arh)/np.log(10),0.1) #array of scale factors outside Jeans
Dr_inh=table(lambda a: F2(a)/(1/3)*(a**2*Ht(a)/k)**2, a_wkb_rinh)

#cannibal amplitude inhomogeneous
F_c= lambda a: 3/2/a**2*(1+w(a))*Dr_i*np.exp(-N_r(a)/2)
a_wkb_cinh=10**np.arange(np.log(a_max_i_rh)/np.log(10),np.log(adom2)/np.log(10),0.1)
Dc_inh=table(lambda a: F_c(a)/(-c2s(a)+1/3)*(a**2*Ht(a))**2/k**2, a_wkb_cinh)

plt.rcParams.update({'font.size': 20})

miny=10**-10
maxy=10**2
plt.loglog(a_delta_r,abs(delta_r),label="numerical",color='orange')
plt.loglog(a_wkb_rh,abs(Dr_h),'--k',label="transient")
plt.loglog(a_wkb_rinhj,abs(Dr_inh_j),'-.',color='blue',linewidth=2,label=r"steady-state $k>k_J$")
plt.loglog(a_wkb_rinh,abs(Dr_inh),'-.',color='red',linewidth=2,label=r"steady-state $k<k_J$")
#plt.loglog(delta.t[:],abs(delta.y[1,:]),label="Cannibal",color='red')
#plt.loglog(a_full,abs(delta_dm),label="DM",color='blue')
plt.loglog([ahz,ahz],[miny,maxy],'-.k')
plt.text(1.1*ahz,2*miny,r"$a_{hor}$")
plt.loglog([afz2,afz2],[miny,maxy],'--g')
plt.text(1.1*afz2,2*miny,r"$a_{fz}$")
plt.loglog([arh2,arh2],[miny,maxy],'--b')
plt.text(1.1*arh2,2*miny,r"$a_{rh}$")
plt.loglog([adom2,adom2],[miny,maxy],'--',color='purple')
plt.text(1.1*adom2,2*miny,r"$a_{dom}$")
plt.ylim([miny,maxy])
plt.xlim([1,3.5*arh2])
plt.xlabel(r"$a/a_i$")
plt.ylabel(r"$\frac{|\delta_r|}{\phi_P}$",rotation=0, labelpad=10)
plt.legend(loc=6)

plt.gcf().set_size_inches(10, 6,forward=True)
plt.gcf().subplots_adjust(left=0.13, right=0.98, bottom=0.12, top=0.97)

mycwd = os.getcwd()
os.chdir("..")
plt.savefig("rad_delta_wkb2.pdf")
os.chdir(mycwd)

plt.rcParams.update({'font.size': 20})

miny=10**-4
maxy=10**3
plt.figure(2)
plt.rcParams.update({'font.size': 20})
plt.loglog(a_delta_can,abs(delta_can),label="numerical",color='red')
plt.loglog(a_wkb_ch,abs(Dc_h),'--k',label="transient")
plt.loglog(a_wkb_cinh,abs(Dc_inh),'-.',color='blue',label="steady-state")
plt.loglog([afz2,afz2],[miny,maxy],'--g')
plt.text(1.1*afz2,2*miny,r"$a_{fz}$")
plt.loglog([arh2,arh2],[miny,maxy],'--b')
plt.text(1.1*arh2,2*miny,r"$a_{rh}$")
plt.loglog([ahz,ahz],[miny,maxy],'-.k')
plt.text(1.1*ahz,2*miny,r"$a_{hor}$")
plt.loglog([adom2,adom2],[miny,maxy],'--',color='purple')
plt.text(1.1*adom2,2*miny,r"$a_{dom}$")
plt.ylim([miny,maxy])
plt.xlim([1,3.5*arh2])
plt.xlabel(r"$a/a_i$")
plt.ylabel(r"$\frac{|\delta_c|}{\phi_P}$",rotation=0, labelpad=10)
plt.legend(loc=5)
plt.show()

plt.gcf().set_size_inches(10, 6,forward=True)
plt.gcf().subplots_adjust(left=0.13, right=0.98, bottom=0.12, top=0.97)

mycwd = os.getcwd()
os.chdir("..")
plt.savefig("can_delta_wkb2.pdf")
os.chdir(mycwd)