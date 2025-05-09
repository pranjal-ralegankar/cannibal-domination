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

acan=100 #effectively when cannibal phase starts; T=m/5
afz=10**4/5 #scale factor at which cannibal reactions freezeout and cannibal evolves as CDM
adom=2*10**5/3
arh=1*10**9/3 #scale at which we want reheating to happen
Trh=10 #in Mev

m,xi,alpha_can,afz2,adom2,arh2,rhocan,rhor,X,w,c2s,Ht,rhodm,hor_i,aeq,gamma,H1=bck2(afz,adom,arh,Trh,1)

ddel,ddelsim,ddel_super_sim,ddel_post_rh,ddel_r=pert_wgamma(rhocan,rhor,rhodm,Ht,c2s,w,gamma)

# =============================================================================
#Solving perturbation equations

#finding kpk
csfz=np.sqrt(c2s(2*afz))
lambda1=np.real(3*csfz/2/np.sqrt(2)*(lambertw(2*(0.594/3/csfz*adom/afz)**(2/3)))**(3/2))
apk=lambda1*afz
kpk=apk*Ht(apk)
k=0.0075*5/3
ahz=optimize.fsolve(lambda a: a*Ht(a)-k,10)[0]

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
print("main perturb eqs done")

a_eff=optimize.fsolve(lambda a: gamma*rhocan(a)-Ht(a)*rhor(a),10)[0] #scale when decays become effective in SM radiation bath.
delta0_r=[delta.y[5,-1],delta.y[6,-1]] #super-horizon initial condition
deltar=solve_ivp(lambda a,y: ddel_r(a,y,k,deltasim.sol),[delta.t[-1],a_eff],delta0_r,method='BDF',dense_output=False,atol=1e-8,rtol=1e-8)
print("aeff")
delta0_r2=[deltar.y[0,-1],deltar.y[1,-1]]
deltar2=solve_ivp(lambda a,y: ddel_r(a,y,k,deltasim.sol),[a_eff,5*arh],delta0_r2,method='Radau',dense_output=False,atol=1e-10,rtol=1e-10)
delta_r=np.concatenate((delta.y[5,:],deltar.y[0,:],deltar2.y[0,:]))
a_delta_r=np.concatenate((a_delta_r,deltar.t,deltar2.t))

lambda_J=lambda a:(3/2*(1+w(a))/c2s(a))**-0.5*(a*Ht(a))**-1
lambda_H=lambda a:(a*Ht(a))**-1
lambda_Hu=lambda a:(10*arh*Ht(10*arh))**-1*a/(10*arh)
k_osc=lambda a: pi/2/c2s(a)**0.5/lambertw(0.594*pi/2/c2s(a)**0.5)/lambda_H(a)
rs=lambda a: integrate.quad(lambda x: c2s(x)**0.5*lambda_H(x)/x, 0.01, a)[0]
# aj_ent=optimize.fsolve(lambda a: 1/lambda_J(a)-k,ahz)[0]
aj_ent=optimize.fsolve(lambda a: 1/rs(a)-k,ahz)[0]
aj_esc=optimize.fsolve(lambda a: 1/lambda_J(a)-k,adom)[0]

aplt=10**np.arange(0,np.log(12*arh)/np.log(10),0.1) #end point corresponds to last mode we have evaluated for rel DM growth
aplt2=10**np.arange(0,np.log(arh2)/np.log(10),0.01)
lambda_Htable=table(lambda_H,aplt)
lambda_Jtable=table(lambda_J,aplt2)
lambda_Hutable=table(lambda_Hu,aplt)
k_osctable=table(k_osc,aplt2)
rs_table=table(rs,aplt2)
miny_hor=lambda_H(aplt[0])
maxy_hor=lambda_H(aplt[-1])

plt.rcParams.update({'font.size': 20})
abreak=np.where(aplt2>adom2)[0][0]

ax1=plt.subplot(211)
plt.loglog(aplt,lambda_Htable,label=r"Horizon")
plt.loglog(aplt2[0:abreak],lambda_Jtable[0:abreak],'--',linewidth=1, color='darkorange')
#plt.loglog(aplt2[0:abreak],1/k_osctable[0:abreak],'-',color='darkgoldenrod',label=r"$k_{osc}^{-1}$")
plt.loglog(aplt2[0:abreak],rs_table[0:abreak],'-',color='darkgoldenrod',label=r"$r_s$")
plt.loglog(aplt2[abreak:],rs_table[abreak:],'--',linewidth=1,color='darkgoldenrod')
plt.loglog(aplt2[abreak:-1],lambda_Jtable[abreak:-1],color='darkorange',label=r"$k_J^{-1}$")
#plt.loglog(aplt,lambda_Hutable,'--b',linewidth=0.5,label=r"$\Lambda$CDM Horizon")
plt.loglog([ahz,ahz],[miny_hor,maxy_hor],'-.k')
plt.text(1.1*ahz,2*miny_hor,r"$a_{hor}$")
plt.loglog([100,100],[miny_hor,maxy_hor],'--r')
plt.text(100/5,2*miny_hor,r"$a_{can}$")
plt.loglog([afz2,afz2],[miny_hor,maxy_hor],'--g')
plt.text(1.1*afz2,2*miny_hor,r"$a_{fz}$")
plt.loglog([adom2,adom2],[miny_hor,maxy_hor],'--',color='purple')
plt.text(1.1*adom2,2*miny_hor,r"$a_{dom}$")
plt.loglog([arh2,arh2],[miny_hor,maxy_hor],'--b')
plt.text(1.1*arh2,2*miny_hor,r"$a_{rh}$")
plt.loglog([aj_ent,aj_ent],[miny_hor,maxy_hor],'-.',color='grey',linewidth=1)
plt.loglog([aj_esc,aj_esc],[miny_hor,maxy_hor],'-.',color='grey',linewidth=1)
plt.loglog([aplt[0],aplt[-1]],[k**-1,k**-1],'--k')  
plt.text(aplt[0]*2,1.1*k**-1,r"$k^{-1}$")
plt.fill_between(aplt, lambda_Htable, miny_hor, facecolor='blue', alpha=0.1)
plt.fill_between(aplt2[0:abreak], rs_table[0:abreak], miny_hor, facecolor='yellow', alpha=0.5)
plt.fill_between(aplt2[abreak-1:-1], lambda_Jtable[abreak-1:-1], miny_hor, facecolor='yellow', alpha=0.5)
plt.ylim([miny_hor,maxy_hor])
plt.xlim([aplt[0],3.5*arh])
#plt.xlabel(r"$a$")
plt.ylabel(r"$\frac{k_{hor,i}}{k}$",rotation=0, labelpad=18)
plt.legend()

ax3=plt.subplot(212,sharex=ax1)
miny=3*10**-5
maxy=600
plt.figure(1)
plt.rcParams.update({'font.size': 20})
plt.loglog(a_delta_r,abs(delta_r),label="Radiation",color='orange')
plt.loglog(a_delta_can,abs(delta_can),label="Cannibal",color='red')
plt.loglog(a_full,abs(delta_dm),label="DM",color='blue')
plt.loglog([100,100],[miny,maxy],'--r')
plt.text(100/5,2*miny,r"$a_{can}$")
plt.loglog([afz2,afz2],[miny,maxy],'--g')
plt.text(1.1*afz2,2*miny,r"$a_{fz}$")
plt.loglog([arh2,arh2],[miny,maxy],'--b')
plt.text(1.1*arh2,2*miny,r"$a_{rh}$")
plt.loglog([ahz,ahz],[miny,maxy],'-.k')
plt.text(1.1*ahz,2*miny,r"$a_{hor}$")
plt.loglog([aj_ent,aj_ent],[miny,maxy],'-.',color='grey',linewidth=1)
plt.loglog([aj_esc,aj_esc],[miny,maxy],'-.',color='grey',linewidth=1)
#plt.loglog([acan,acan],[miny,maxy],'--r')
#plt.text(acan,2*miny,r"$a_{can}$")
#ahz=optimize.fsolve(lambda a: a*Ht(a)-k,10)[0] #finding scale factor at which mode k enters horizon
#plt.loglog([ahz,ahz],[miny,maxy],'--k')
#plt.text(ahz,2*miny,r"$a_{hz}$")
plt.loglog([adom2,adom2],[miny,maxy],'--',color='purple')
plt.text(1.1*adom2,2*miny,r"$a_{dom}$")
plt.ylim([miny,maxy])
plt.xlabel(r"$a/a_i$")
plt.ylabel(r"$\frac{|\delta|}{\phi_P}$",rotation=0, labelpad=10)
plt.legend(bbox_to_anchor=(0, 0.4),loc='center left')
plt.show()

plt.gcf().set_size_inches(10, 10,forward=True)
plt.gcf().subplots_adjust(left=0.13, right=0.95, bottom=0.08, top=0.98)

#mycwd = os.getcwd()
#os.chdir("..")
#plt.savefig("SM_dom_horz_pert.pdf")
#os.chdir(mycwd)