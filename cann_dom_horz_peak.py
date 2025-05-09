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
from perturbation import pert
os.chdir('../../../draft/cannibal-long_paper_figs')
def table(f,t):
    tablef=np.zeros(t.shape[0])
    for i in np.arange(t.shape[0]):
        tablef[i]=f(t[i])
    return tablef

acan=100 #effectively when cannibal phase starts; T=0.1m
afz=3*10**3 #scale factor at which cannibal reactions freezeout and cannibal evolves as CDM
arh=10**8
xi=100
Trh=10#in Mev

# =============================================================================
#finding dimensionless densities, cannibal thermo quantities and particle parameters from solving background equations
m,xi,alpha_can,afz2,adom2,arh2,rhocan,rhor,X,w,c2s,Ht,rhodm,hor_i,aeq,gamma,H1=bck2(afz,xi,arh,Trh,2)
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
Tr_i=10*m/xi #finding temperature of the SM at a_i assuming SM and HS were in equilibrium early on
sigmav2_can=25*sqrt(5)*pi**2*alpha_can**3/5184/m**5 # cannibal 3 to 2 anihilation rate in Mev^-5


#Perturbation equations
ddel,ddelsim,ddel_super_sim,ddel_post_rh,ddel_r=pert(rhocan,rhor,rhodm,Ht,c2s,w,gamma)

# =============================================================================
# Calculating relative transfer function
j=0
kstart=5*arh*Ht(5*arh)
krh=arh*Ht(arh)
k0=1*Ht(1)
kfz=afz*Ht(afz)

lambda_J=lambda a:(3/2*(1+w(a))/c2s(a))**-0.5*(a*Ht(a))**-1
kosc=2*lambda_J(arh)**-1 #modes larger than kosc never escape jeans horizon. Thus can ignore cannibal perturbations.
kosc=k0
K=10**np.concatenate((np.arange(log(kstart)/log(10),log(kfz)/log(10),0.1),np.arange(log(kfz)/log(10),log(kosc)/log(10),0.01),np.arange(log(kosc)/log(10),log(k0)/log(10),0.05)))
print(K.shape[0])
delta_dmf=np.zeros(K.shape[0])
delta_dm_primef=np.zeros(K.shape[0])
aend=np.zeros(K.shape[0])
#delta_dmsim=np.zeros(K.shape[0])
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
relative_delta_DM=A/9.11*(1+(np.log(ahoru/0.594/aend)+delta_dmf/A)/np.log(4*0.594*np.exp(-3)*aeq/ahoru))

#analytical max peak for adom<afz
kpk=(3/2*(1+w(2*afz))/c2s(2*afz))**0.5*(2*afz*Ht(2*afz))/1.2
ahoru_pk=(5*arh2)**2*Ht(5*arh2)/kpk
peak=2.86*arh2/afz2/9.11*(1+np.log(ahoru_pk*1.55/0.594/arh)/np.log(4*0.594*np.exp(-3)*aeq/ahoru_pk))
# =============================================================================

# =============================================================================
# plotting all density perturbations for specific mode
kpk=(3/2*(1+w(2*afz))/c2s(2*afz))**0.5*(2*afz*Ht(2*afz))/1.2
k1=kpk/10
k2=kpk*5

k=k1#afz/20*Ht(afz/20)##(3/2*(1+w(2*afz))/c2s(2*afz))**0.5*(2*afz*Ht(2*afz))/1.2#afz/2*Ht(afz/2)
ahz=optimize.fsolve(lambda a: a*Ht(a)-k,10)[0]
if ahz<arh/100: #solve differently if mode enters the horizon much before reheating
    astart=min(1,ahz/10)
    delta0=[1,-2,k**2/2/astart/Ht(astart),-2/(1+1/3),k**2/2/astart/Ht(astart)] #super-horizon initial condition
    delta=solve_ivp(lambda a,y: ddelsim(a,y,k),[astart,5*arh],delta0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5)#ignore radiation perturbations completely
    A=delta.t[-1]*(delta.y[3,-1]-delta.y[3,-2])/(delta.t[-1]-delta.t[-2]) #Find A and B for dm density evolving as \delta=A*log(B*a) after reheating
    B=1/(delta.t[-1])*np.exp(delta.y[3,-1]/A)#finding B
    deltadm_post_rh=A*np.log(np.arange(delta.t[-1],10*arh,arh)/delta.t[-1])+delta.y[3,-1]# finding \delta_DM from arh to 1000arh using \delta=A*log(B*a)
    delta_dm=delta.y[3,:]#np.append(delta.y[3,:],deltadm_post_rh[:])# joinging the post reheating DM density evolution with pre-reheating
    a_full=delta.t[:]#np.append(delta.t[:],np.arange(delta.t[-1],1000*arh,arh))
else:
    astart=1#min(ahz/20,arh/100)
    aend=max(ahz*100,10*arh)
    delta0=[1,-2,k**2/2,-2/(1+w(astart)),k**2/2,-2,k**2/2]
    delta=solve_ivp(lambda a,y: ddel(a,y,k),[astart,5*arh],delta0,method='BDF',dense_output=False,atol=1e-8,rtol=1e-7)# Complete perturbation equations
    delta_post_rh0=[delta.y[0,-1],delta.y[3,-1],delta.y[4,-1],delta.y[5,-1],delta.y[6,-1]]
    delta_post_rh=solve_ivp(lambda a,y: ddel_post_rh(a,y,k),[delta.t[-1],aend],delta_post_rh0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5) #after reheating ignore cannibal perturbation equations
    a_full=np.append(delta.t[:],delta_post_rh.t[:])
    delta_dm=np.append(delta.y[3,:],delta_post_rh.y[1,:])# joinging the post reheating DM density evolution with pre-reheating
    delta_r=np.append(delta.y[5,:],delta_post_rh.y[3,:])# joinging the post reheating radiation density evolution with pre-reheating
    delta_can=delta.y[1,:]
    a_delta_can=delta.t[:]

delta_can1=delta.y[1,:]
a_delta_can1=delta.t[:]
a_full1=a_full
delta_dm1=delta_dm

k=kpk
ahz=optimize.fsolve(lambda a: a*Ht(a)-k,10)[0]
if ahz<arh/100: #solve differently if mode enters the horizon much before reheating
    astart=min(1,ahz/10)
    delta0=[1,-2,k**2/2/astart/Ht(astart),-2/(1+1/3),k**2/2/astart/Ht(astart)]#super-horizon initial condition
    delta=solve_ivp(lambda a,y: ddelsim(a,y,k),[astart,5*arh],delta0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5)#ignore radiation perturbations completely
    A=delta.t[-1]*(delta.y[3,-1]-delta.y[3,-2])/(delta.t[-1]-delta.t[-2]) #Find A and B for dm density evolving as \delta=A*log(B*a) after reheating
    B=1/(delta.t[-1])*np.exp(delta.y[3,-1]/A)#finding B
    deltadm_post_rh=A*np.log(np.arange(delta.t[-1],10*arh,arh)/delta.t[-1])+delta.y[3,-1]# finding \delta_DM from arh to 1000arh using \delta=A*log(B*a)
    delta_dm=delta.y[3,:]#np.append(delta.y[3,:],deltadm_post_rh[:])# joinging the post reheating DM density evolution with pre-reheating
    a_full=delta.t[:]#np.append(delta.t[:],np.arange(delta.t[-1],1000*arh,arh))
else:
    astart=1#min(ahz/20,arh/100)
    aend=max(ahz*100,10*arh)
    delta0=[1,-2,k**2/2,-2/(1+w(astart)),k**2/2,-2,k**2/2]
    delta=solve_ivp(lambda a,y: ddel(a,y,k),[astart,5*arh],delta0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5)# Complete perturbation equations
    delta_post_rh0=[delta.y[0,-1],delta.y[3,-1],delta.y[4,-1],delta.y[5,-1],delta.y[6,-1]]
    delta_post_rh=solve_ivp(lambda a,y: ddel_post_rh(a,y,k),[delta.t[-1],aend],delta_post_rh0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5) #after reheating ignore cannibal perturbation equations
    a_full=np.append(delta.t[:],delta_post_rh.t[:])
    delta_dm=np.append(delta.y[3,:],delta_post_rh.y[1,:])# joinging the post reheating DM density evolution with pre-reheating
    delta_r=np.append(delta.y[5,:],delta_post_rh.y[3,:])# joinging the post reheating radiation density evolution with pre-reheating
    delta_can=delta.y[1,:]
    a_delta_can=delta.t[:]

delta_can2=delta.y[1,:]
a_delta_can2=delta.t[:]
a_full2=a_full
delta_dm2=delta_dm

k=k2
ahz=optimize.fsolve(lambda a: a*Ht(a)-k,10)[0]
if ahz<arh/100: #solve differently if mode enters the horizon much before reheating
    astart=min(1,ahz/10)
    delta0=[1,-2,k**2/2/astart/Ht(astart),-2/(1+1/3),k**2/2/astart/Ht(astart)] #super-horizon initial condition
    delta=solve_ivp(lambda a,y: ddelsim(a,y,k),[astart,5*arh],delta0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5)#ignore radiation perturbations completely
    A=delta.t[-1]*(delta.y[3,-1]-delta.y[3,-2])/(delta.t[-1]-delta.t[-2]) #Find A and B for dm density evolving as \delta=A*log(B*a) after reheating
    B=1/(delta.t[-1])*np.exp(delta.y[3,-1]/A)#finding B
    deltadm_post_rh=A*np.log(np.arange(delta.t[-1],10*arh,arh)/delta.t[-1])+delta.y[3,-1]# finding \delta_DM from arh to 1000arh using \delta=A*log(B*a)
    delta_dm=delta.y[3,:]#np.append(delta.y[3,:],deltadm_post_rh[:])# joinging the post reheating DM density evolution with pre-reheating
    a_full=delta.t[:]#np.append(delta.t[:],np.arange(delta.t[-1],1000*arh,arh))
else:
    astart=1#min(ahz/20,arh/100)
    aend=max(ahz*100,10*arh)
    delta0=[1,-2,k**2/2,-2/(1+w(astart)),k**2/2,-2,k**2/2]
    delta=solve_ivp(lambda a,y: ddel(a,y,k),[astart,5*arh],delta0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5)# Complete perturbation equations
    delta_post_rh0=[delta.y[0,-1],delta.y[3,-1],delta.y[4,-1],delta.y[5,-1],delta.y[6,-1]]
    delta_post_rh=solve_ivp(lambda a,y: ddel_post_rh(a,y,k),[delta.t[-1],aend],delta_post_rh0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5) #after reheating ignore cannibal perturbation equations
    a_full=np.append(delta.t[:],delta_post_rh.t[:])
    delta_dm=np.append(delta.y[3,:],delta_post_rh.y[1,:])# joinging the post reheating DM density evolution with pre-reheating
    delta_r=np.append(delta.y[5,:],delta_post_rh.y[3,:])# joinging the post reheating radiation density evolution with pre-reheating
    delta_can=delta.y[1,:]
    a_delta_can=delta.t[:]

delta_can3=delta.y[1,:]
a_delta_can3=delta.t[:]
a_full3=a_full
delta_dm3=delta_dm

# =============================================================================
# plotting
lambda_H=lambda a:(a*Ht(a))**-1
lambda_Hu=lambda a:(10*arh2*Ht(10*arh2))**-1*a/(10*arh2)

aplt=10**np.arange(0,np.log(12*arh2)/np.log(10),0.1) #end point corresponds to last mode we have evaluated for rel DM growth
aplt2=10**np.arange(0,np.log(arh2)/np.log(10),0.1)
lambda_Htable=table(lambda_H,aplt)
lambda_Jtable=table(lambda_J,aplt2)
lambda_Hutable=table(lambda_Hu,aplt)
miny_hor=lambda_H(aplt[0])
maxy_hor=lambda_H(aplt[-1])
krh=arh2*Ht(arh2) #mode which eneters horizon at reheating

plt.rcParams.update({'font.size': 20})

ax1=plt.subplot(221)
miny=miny_hor
maxy=maxy_hor
plt.loglog(aplt,lambda_Htable,label=r"Horizon")
plt.loglog(aplt2,lambda_Jtable,label=r"Jeans")
plt.loglog(aplt,lambda_Hutable,'--b',linewidth=0.5,label=r"$\Lambda$CDM Horizon")
plt.loglog([afz2,afz2],[miny,maxy],'--g')
plt.text(1.1*afz2,2*miny,r"$a_{fz}$")
plt.loglog([arh2,arh2],[miny,maxy],'--b')
plt.text(1.1*arh2,2*miny,r"$a_{rh}$")
plt.loglog([aplt[0],aplt[-1]],[k1**-1,k1**-1],'--',color='red')  
plt.text(aplt[-1]/7,k1**-1*1.5,r"$k_1^{-1}$")
plt.loglog([aplt[0],aplt[-1]],[kpk**-1,kpk**-1],'--',color='black')  
plt.text(aplt[-1]/7,kpk**-1*1.5,r"$k_{pk}^{-1}$")
plt.loglog([aplt[0],aplt[-1]],[k2**-1,k2**-1],'--',color='blue')  
plt.text(aplt[-1]/7,k2**-1*1.5,r"$k_2^{-1}$")
plt.fill_between(aplt, lambda_Htable, miny, facecolor='blue', alpha=0.1)
plt.fill_between(aplt2, lambda_Jtable, miny, facecolor='yellow', alpha=0.5)
plt.ylim([miny,maxy])
plt.xlim([aplt[0],aplt[-1]])
#plt.xlabel(r"$a$")
plt.ylabel(r"$\frac{k_{hor,i}}{k}$",rotation=0, labelpad=15)
plt.legend()

ax3=plt.subplot(223,sharex=ax1)
miny=10**-3
maxy=10**7
plt.loglog(a_delta_can1,abs(delta_can1),label=r"$\delta_c(k_1)$",color='red')
plt.loglog(a_delta_can2,abs(delta_can2),label=r"$\delta_c(k_{pk})$",color='black')
plt.loglog(a_delta_can3,abs(delta_can3),label=r"$\delta_c(k_2)$",color='blue')
plt.loglog(a_full1,abs(delta_dm1),'-.',label=r"$\delta_{DM}(k_1)$",color='red',alpha=0.5)
plt.loglog(a_full2,abs(delta_dm2),'-.',label=r"$\delta_{DM}(k_{pk})$",color='black',alpha=0.5)
plt.loglog(a_full3,abs(delta_dm3),'-.',label=r"$\delta_{DM}(k_2)$",color='blue',alpha=0.5)
#plt.loglog(a_full,abs(delta_dm),label=r"$a_{hz}=a_{hz}/10$")
#plt.loglog(delta.t[:],abs(delta.y[0,:]),label="phi",color='green')
plt.loglog([afz2,afz2],[miny,maxy],'--g')
plt.text(1.1*afz2,2*miny,r"$a_{fz}$")
plt.loglog([arh2,arh2],[miny,maxy],'--b')
plt.text(1.1*arh2,2*miny,r"$a_{rh}$")
#plt.loglog([acan,acan],[miny,maxy],'--r')
#plt.text(acan,2*miny,r"$a_{can}$")
#plt.loglog([ahz,ahz],[miny,maxy],'--k')
#plt.text(ahz,2*miny,r"$a_{hz}$")
plt.ylim([miny,maxy])
plt.xlabel(r"$a/a_i$")
plt.ylabel(r"$\frac{|\delta|}{\phi_P}$",rotation=0, labelpad=10)
plt.legend(ncol=2)

ax2=plt.subplot(222,sharey=ax1)
minx=0.1#np.min(abs(relative_delta_DM))
maxx=5*np.max(abs(relative_delta_DM))
plt.loglog(abs(relative_delta_DM),1/K)
#krad=10*Ht(10)
#plt.loglog([minx,maxx],[krad**-1,krad**-1],'--y')
#plt.text(maxx/2,krad**-1,r"$k_{rad}$")
plt.loglog([minx,maxx],[krh**-1,krh**-1],'--b')
plt.text(minx*1.2,1.5*krh**-1,r"$k_{rh}$")
plt.loglog([minx,maxx],[kpk**-1,kpk**-1],'--k')  
plt.text(minx*1.2,1.5*kpk**-1,r"$k_{pk}$")
plt.xlim([minx,maxx])
#plt.ylim([1/K[0],1/K[-1]])
plt.loglog([peak,peak],[K[0]**-1,K[-1]**-1],'-.',color='black')
plt.ylabel(r"$\frac{k_{hor,i}}{k}$",rotation=0, labelpad=15)
plt.xlabel(r"$|T(k)|$")

ax4=plt.subplot(224)
miny=0.1
maxy=5*np.max(abs(relative_delta_DM))
plt.loglog(K,abs(relative_delta_DM))
plt.loglog([kpk,kpk],[miny,maxy],'--k')
plt.text(1.1*kpk,2*miny,r"$k_{pk}$")
plt.loglog([krh,krh],[miny,maxy],'--b')
plt.text(1.1*krh,2*miny,r"$k_{rh}$")
plt.loglog([K[0],K[-1]],[peak,peak],'-.',color='black')
plt.ylim([miny,maxy])
plt.xlabel(r"$k/k_{hor,i}$")
plt.ylabel(r"$|T(k)|$",rotation=90, labelpad=0)
#plt.legend()

plt.gcf().set_size_inches(15, 10,forward=True)
plt.gcf().subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.97)

#mycwd = os.getcwd()
#os.chdir("..")
#plt.savefig("can_dom_horz_peak2.pdf")
#os.chdir(mycwd)