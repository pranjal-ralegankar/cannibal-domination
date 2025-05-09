# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 18:50:49 2019

@author: pranj
"""

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
arh=arh 
xi=xi
Trh=Trh

m,xi,alpha_can,afz2,adom2,arh2,rhocan,rhor,X,w,c2s,Ht,rhodm,hor_i,aeq,gamma,H1=bck(afz,xi,arh,Trh,2)

#Solving perturbation equations
ddel,ddelsim,ddel_super_sim,ddel_post_rh,ddel_r=pert_wgamma(rhocan,rhor,rhodm,Ht,c2s,w,gamma)

# =============================================================================
# plotting all density perturbations for specific mode
k=k#afz/20*Ht(afz/20)##(3/2*(1+w(2*afz))/c2s(2*afz))**0.5*(2*afz*Ht(2*afz))/1.2#afz/2*Ht(afz/2)
ahz=optimize.fsolve(lambda a: a*Ht(a)-k,10)[0]

#astart=min(1,ahz/10)
#delta0=[1,-2,k**2/2,-2/(1+1/3),k**2/2] #super-horizon initial condition
#delta=solve_ivp(lambda a,y: ddelsim(a,y,k),[astart,10*arh],delta0,method='BDF',dense_output=True,atol=1e-6,rtol=1e-5)#ignore radiation perturbations completely
#aj_esc=optimize.fsolve(lambda a: (3/2*(1+w(a))/c2s(a))**0.5*(a*Ht(a))-k,2*afz)[0] #point where WKB fails
#
#delta0_r=[-2,k**2/2] #super-horizon initial condition
#delta_r=solve_ivp(lambda a,y: ddel_r(a,y,k,delta.sol),[astart,1*arh],delta0_r,method='Radau',dense_output=False,atol=1e-8,rtol=1e-7)
#delta0_r2=[delta_r.y[0,-1],delta_r.y[1,-1]]
#print("arh")
#delta_r2=solve_ivp(lambda a,y: ddel_r(a,y,k,delta.sol),[1*arh,10*arh],delta0_r2,method='BDF',dense_output=False,atol=1e-10,rtol=1e-10)
#delta_ry=np.append(delta_r.y[0,:],delta_r2.y[0,:])
#delta_r.t=np.append(delta_r.t,delta_r2.t)

#cannibal homogeneous amplitude
a_max_i_c=optimize.fsolve(lambda a: derivative(delta.sol,a)[1]-0,17*ahz)[0]# closest a at which delta_can is at max
Dc_i=delta.sol(a_max_i_c)[1] #value of delta_c at max to serve as initial amplitude
a_wkb_ch=10**np.arange(np.log(a_max_i_c)/np.log(10),np.log(aj_esc)/np.log(10),0.1) #array of scale factors where WKB is valid
n_c=lambda a: (1-3*w(a))/a #delta' coefficient
N_c=lambda a: integrate.quad(n_c, a_max_i_c, a)[0] #integral of n(a)dlna
Dc_h=Dc_i*table(lambda a: np.exp(-N_c(a)/2)*(c2s(a_max_i_c)/c2s(a))**0.25, a_wkb_ch) #full WKB amplitude

#radiation amplitude within jeans
xi= lambda a: gamma*rhocan(a)/Ht(a)/rhor(a)
F_j=lambda a: (2+w(a))/(1+w(a))*xi(a)/a*c2s(a)**0.5*k/(a**2*Ht(a))*Dc_i*np.exp(-N_c(a)/2)*(c2s(a_max_i_c)/c2s(a))**0.25
Dr_inh_j=table(lambda a: F_j(a)/(1/3-c2s(a))*(a**2*Ht(a)/k)**2, a_wkb_ch)

#radiation amplitude outside jeans
F2= lambda a: ((2*a*Ht(a)+a**2*derivative(Ht,a))/(a**2*Ht(a))*xi(a)/a+(xi(a)/a)**2+xi(a)/a**2+2/a**2*rhocan(a)/(rhocan(a)+rhor(a)))*delta.sol(a)[1]
a_wkb_rinh=10**np.arange(np.log(aj_esc)/np.log(10),np.log(10*arh)/np.log(10),0.1) #array of scale factors where WKB is valid
Dr_inh=table(lambda a: F2(a)/(1/3)*(a**2*Ht(a)/k)**2, a_wkb_rinh)

#radiation amplitude homogeneous
index_amax=np.where(delta_r.t>a_max_i_c)[0][0]# index at which a=a_max_i_c in delta_r
Dr_i=delta_ry[index_amax-65] #value of delta_r at max to serve as initial amplitude
a_wkb_rh=10**np.arange(np.log(delta_r.t[index_amax-65])/np.log(10),np.log(5*arh)/np.log(10),0.1) #array of scale factors where WKB is valid
n_r=lambda a: 2*gamma*rhocan(a)/Ht(a)/rhor(a)/a #delta' coefficient
N_r=lambda a: integrate.quad(n_r, a_max_i_c, a)[0] #integral of n(a)dlna
Dr_h=Dr_i*table(lambda a: np.exp(-N_r(a)/2), a_wkb_rh) #full WKB amplitude

plt.rcParams.update({'font.size': 20})
miny=10**-4
maxy=10**5
plt.loglog(delta.t[:],abs(delta.y[1,:]),label="numerical",color='red')
#plt.loglog(a_wkb,abs(Dc),'-k',label="WKB2")
plt.loglog(a_wkb_ch,abs(Dc_h),'--k',label="transient")
plt.loglog([ahz,ahz],[miny,maxy],'-.k')
plt.text(1.1*ahz,2*miny,r"$a_{hor}$")
plt.loglog([afz2,afz2],[miny,maxy],'--g')
plt.text(1.1*afz2,2*miny,r"$a_{fz}$")
plt.loglog([arh2,arh2],[miny,maxy],'--b')
plt.text(1.1*arh2,2*miny,r"$a_{rh}$")
plt.ylim([miny,maxy])
plt.xlim([1,10*arh2])
plt.xlabel(r"$a/a_i$")
plt.ylabel(r"$\frac{|\delta_c|}{\phi_P}$",rotation=0, labelpad=10)
plt.legend(loc='upper center')

plt.gcf().set_size_inches(10, 6,forward=True)
plt.gcf().subplots_adjust(left=0.13, right=0.95, bottom=0.15, top=0.97)

mycwd = os.getcwd()
os.chdir("..")
plt.savefig("can_delta_WKB.pdf")
os.chdir(mycwd)

plt.figure(2)
plt.rcParams.update({'font.size': 20})
miny=10**-9
maxy=10**2
plt.loglog(delta_r.t,abs(delta_ry),label="numerical",color='orange')
plt.loglog(a_wkb_ch,abs(Dr_inh_j),'-.',color='blue',linewidth=2,label=r"steady-state $k>k_J$")
plt.loglog(a_wkb_rinh,abs(Dr_inh),'-.',color='red',linewidth=2,label=r"steady-state $k<k_J$")
plt.loglog(a_wkb_rh,abs(Dr_h),'--k',linewidth=2,label="transient")
plt.loglog([ahz,ahz],[miny,maxy],'-.k')
plt.text(1.1*ahz,2*miny,r"$a_{hor}$")
plt.loglog([afz2,afz2],[miny,maxy],'--g')
plt.text(1.1*afz2,2*miny,r"$a_{fz}$")
plt.loglog([arh2,arh2],[miny,maxy],'--b')
plt.text(1.1*arh2,2*miny,r"$a_{rh}$")
plt.ylim([miny,maxy])
plt.xlim([1,10*arh2])
plt.xlabel(r"$a/a_i$")
plt.ylabel(r"$\frac{|\delta_r|}{\phi_P}$",rotation=0, labelpad=20)
plt.legend()

plt.gcf().set_size_inches(10, 6,forward=True)
plt.gcf().subplots_adjust(left=0.13, right=0.98, bottom=0.15, top=0.97)

mycwd = os.getcwd()
os.chdir("..")
plt.savefig("rad_delta_wkb.pdf")
os.chdir(mycwd)