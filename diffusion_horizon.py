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
import pandas as pd

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

acan=681.936575 #effectively when cannibal phase starts; T=0.1m
afz=5*10**2 #scale factor at which cannibal reactions freezeout and cannibal evolves as CDM
adom=10**8
arh=1*10**10 #scale at which we want reheating to happen
Trh=10 #in MeV

#running background evolution
m,xi,alpha_can,afz2,adom2,arh2,rhocan,rhor,X,w,c2s,Ht,rhodm,hor_i,aeq,gamma,H1=bck2(afz,adom,arh,Trh,1)
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
#Solving perturbation equations

#analytical max peak for adom>afz
csfz=np.sqrt(c2s(20*afz))*20
lambda1=np.real(3*csfz/2/np.sqrt(2)*(lambertw(2*(0.594/3/csfz*adom/afz)**(2/3)))**(3/2))
apk=lambda1*afz
kpk=apk*Ht(apk)


# =============================================================================
# plotting
aplt=10**np.arange(0,np.log(10*arh)/np.log(10),0.1) #end point corresponds to last mode we have evaluated for rel DM growth
aplt2=10**np.arange(0,np.log(arh)/np.log(10),0.1)
lambda_J=lambda a:(3/2*(1+w(a))/c2s(a))**-0.5*(a*Ht(a))**-1
lambda_H=lambda a:(a*Ht(a))**-1
lambda_Hu=lambda a:(10*arh*Ht(10*arh))**-1*a/(10*arh)
k_osc=lambda a: pi/2/c2s(a)**0.5/lambertw(0.594*pi/2/c2s(a)**0.5)/lambda_H(a)
rs=lambda a: integrate.quad(lambda x: c2s(x)**0.5*lambda_H(x)/x, 0.01, a)[0]
can_scat_rate=lambda a:0.63*2**0.5/4/(2*pi)**2*(5/3*(4*pi*alpha_can))**2*X(a)**-0.5*rhocan(a)*pi**2/30*10**4*m/H1
lambda_can_diff=lambda a: X(a)**-0.5*(can_scat_rate(a)*Ht(a))**-0.5*1/a
akd=optimize.fsolve(lambda a: Ht(a)-can_scat_rate(a),10)[0]
fs=lambda a: lambda_can_diff(akd)+integrate.quad(lambda x: X(x)**-0.5*lambda_H(x)/x, akd, a)[0]

aplt=10**np.arange(0,np.log(12*arh2)/np.log(10),0.1) #end point corresponds to last mode we have evaluated for rel DM growth
aplt2=10**np.arange(0,np.log(arh2)/np.log(10),0.1)
lambda_Htable=table(lambda_H,aplt)
lambda_Jtable=table(lambda_J,aplt2)
lambda_Hutable=table(lambda_Hu,aplt)
k_osctable=table(k_osc,aplt2)
rs_table=table(rs,aplt2)
fs_table=table(fs,aplt2)
miny_hor=lambda_H(aplt[0])
maxy_hor=lambda_H(aplt[-1])
krh=arh2*Ht(arh2) #mode which eneters horizon at reheating
kdom=adom2*Ht(adom2) #mode entering horizon at adom
lambda_can_difftable=table(lambda_can_diff,aplt2)

plt.rcParams.update({'font.size': 20})
abreak=np.where(aplt2>adom2)[0][0]
abreak2=np.where(aplt2>akd)[0][0]

ax1=plt.subplot(121)
miny=miny_hor
maxy=maxy_hor
plt.loglog(aplt,lambda_Htable,label=r"Horizon")
# plt.loglog(aplt2[0:abreak],lambda_Jtable[0:abreak],'--',linewidth=1, color='darkorange')
plt.loglog(aplt2[0:abreak],rs_table[0:abreak],'-',color='darkgoldenrod',label=r"$r_s$")
# plt.loglog(aplt2[abreak2:],rs_table[abreak2:],'--',color='darkgoldenrod')
plt.loglog(aplt2[abreak:-1],lambda_Jtable[abreak:-1],color='darkorange',label=r"Jeans")
# plt.loglog(aplt,lambda_Hutable,'--b',linewidth=0.5,label=r"$\Lambda$CDM Horizon")
plt.loglog(aplt2[0:abreak2],lambda_can_difftable[0:abreak2],color='darkgreen',label=r"diffusion + free-streaming")
plt.loglog(aplt2[abreak2-1:abreak+1], fs_table[abreak2-1:abreak+1], color='darkgreen')
plt.loglog([afz2,afz2],[miny,maxy],'--g')
plt.text(1.1*afz2,2*miny,r"$a_{fz}$")
plt.loglog([arh2,arh2],[miny,maxy],'--b')
plt.text(1.1*arh2,2*miny,r"$a_{rh}$")
plt.loglog([adom2,adom2],[miny,maxy],'--',color='purple')
plt.text(1.1*adom2,2*miny,r"$a_{dom}$")
plt.loglog([akd,akd],[miny,maxy],'--',color='grey')
plt.text(1.1*akd,2*miny,r"$a_{kd}$")
plt.loglog([aplt[0],aplt[-1]],[kpk**-1,kpk**-1],'--',color='black')  
plt.text(aplt[-1]/6,kpk**-1*2,r"$k_{pk}^{-1}$")
plt.fill_between(aplt, lambda_Htable, miny, facecolor='blue', alpha=0.1)
plt.fill_between(aplt2[0:abreak2+1], rs_table[0:abreak2+1], miny, facecolor='yellow', alpha=0.5)
# plt.fill_between(aplt2[abreak:-1], lambda_Jtable[abreak:-1], miny, facecolor='yellow', alpha=0.5)
plt.fill_between(aplt2[0:abreak2], lambda_can_difftable[0:abreak2], miny, facecolor='green', alpha=0.5)
plt.fill_between(aplt2[abreak2-1:abreak+1], fs_table[abreak2-1:abreak+1], miny, facecolor='green', alpha=0.5)
plt.fill_between(aplt2[abreak:-1],lambda_Jtable[abreak:-1], miny, facecolor='green', alpha=0.5)
plt.ylim([miny,maxy])
plt.xlim([aplt[0],aplt[-1]])
plt.xlabel(r"$a/a_i$")
plt.ylabel(r"$\frac{k_{hor,i}}{k}$",rotation=0, labelpad=10)
plt.legend()



acan=100 #effectively when cannibal phase starts; T=0.1m
afz=2*10**3 #scale factor at which cannibal reactions freezeout and cannibal evolves as CDM
arh=10**8
xi=100
Trh=10#in Mev

# =============================================================================
#finding dimensionless densities, cannibal thermo quantities and particle parameters from solving background equations
m,xi,alpha_can,afz2,adom2,arh2,rhocan,rhor,X,w,c2s,Ht,rhodm,hor_i,aeq,gamma,H1=bck2(afz,xi,arh,Trh,2)

lambda_J=lambda a:(3/2*(1+w(a))/c2s(a))**-0.5*(a*Ht(a))**-1
kpk=(3/2*(1+w(2*afz))/c2s(2*afz))**0.5*(2*afz*Ht(2*afz))/1.2

lambda_H=lambda a:(a*Ht(a))**-1
lambda_Hu=lambda a:(10*arh2*Ht(10*arh2))**-1*a/(10*arh2)
can_scat_rate=lambda a:0.63*2**0.5/4/(2*pi)**2*(5/3*(4*pi*alpha_can)**0.75)**2*X(a)**-0.5*rhocan(a)*pi**2/30*10**4*m/H1
lambda_can_diff=lambda a: X(a)**-0.5*(can_scat_rate(a)*Ht(a))**-0.5*1/a
akd=optimize.fsolve(lambda a: Ht(a)-can_scat_rate(a),10)[0]
a_j_kd=optimize.fsolve(lambda a: lambda_J(a)-lambda_can_diff(a),5*afz)[0]
rs=lambda a: integrate.quad(lambda x: c2s(x)**0.5*lambda_H(x)/x, 0.01, a)[0]

aplt=10**np.arange(0,np.log(120*arh2)/np.log(10),0.1) #end point corresponds to last mode we have evaluated for rel DM growth
aplt2=10**np.arange(0,np.log(arh2)/np.log(10),0.1)
lambda_Htable=table(lambda_H,aplt)
lambda_Jtable=table(lambda_J,aplt2)
lambda_Hutable=table(lambda_Hu,aplt)
lambda_can_difftable=table(lambda_can_diff,aplt2)
miny_hor=lambda_H(aplt[0])
maxy_hor=lambda_H(aplt[-1])
krh=arh2*Ht(arh2) #mode which eneters horizon at reheating
# rs_table=table(rs,aplt2)

abreak2=np.where(aplt2>a_j_kd)[0][0]

ax3=plt.subplot(122)
# miny=miny_hor
# maxy=maxy_hor
maxx=12*arh2
plt.loglog(aplt,lambda_Htable,label=r"Horizon")
plt.loglog(aplt2,lambda_Jtable,label=r"Jeans")
# plt.loglog(aplt2,rs_table,'--',color='darkgoldenrod')
# plt.loglog(aplt,lambda_Hutable,'--b',linewidth=0.5,label=r"$\Lambda$CDM Horizon")
plt.loglog(aplt2[0:abreak2+1],lambda_can_difftable[0:abreak2+1],color='darkgreen',label=r"diffusion")
plt.loglog([afz2,afz2],[miny,maxy],'--g')
plt.text(1.1*afz2,2*miny,r"$a_{fz}$")
plt.loglog([arh2,arh2],[miny,maxy],'--b')
plt.text(1.1*arh2,2*miny,r"$a_{rh}$")
plt.loglog([akd,akd],[miny,maxy],'--',color='grey')
plt.text(0.25*akd,10*miny,r"$a_{kd}$")
plt.loglog([aplt[0],maxx],[kpk**-1,kpk**-1],'--',color='black')  
plt.text(maxx/7,kpk**-1*1.5,r"$k_{pk}^{-1}$")
plt.fill_between(aplt, lambda_Htable, miny, facecolor='blue', alpha=0.1)
plt.fill_between(aplt2[0:abreak2+1], lambda_Jtable[0:abreak2+1], miny, facecolor='yellow', alpha=0.5)
plt.fill_between(aplt2[0:abreak2+1], lambda_can_difftable[0:abreak2+1], miny, facecolor='green', alpha=0.5)
plt.fill_between(aplt2[abreak2:],lambda_Jtable[abreak2:], miny, facecolor='green', alpha=0.5)
plt.ylim([miny,maxy])
plt.xlim([aplt[0],maxx])
plt.xlabel(r"$a/a_i$")
# plt.ylabel(r"$\frac{k_{hor,i}}{k}$",rotation=0, labelpad=15)
# plt.legend()
plt.gcf().set_size_inches(15, 6,forward=True)
plt.gcf().subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.97)

#mycwd = os.getcwd()
#os.chdir("..")
#plt.savefig("can_dom_horz_peak2.pdf")
#os.chdir(mycwd)