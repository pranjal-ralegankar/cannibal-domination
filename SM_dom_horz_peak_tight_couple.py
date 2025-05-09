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
import pandas as pd

os.chdir('../../pranjal work/git-python/cannibal-base-code')
from background import bck, bck2
from equilibrium import wfun, Xcan
from perturbation import pert, pert_tk
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
Trh=5 #in MeV

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
# This is for calculating \tau_0 which coupled DM and cannibal kinetically in perturbation equations

sigma_k=Ht(arh2/100)/rhocan(arh2/100)*X(arh2/100)**0.5#Ht(arh2/100)/rhocan(arh2/100)*X(arh2/100)**0.5#100*Ht(arh2/2)/rhocan(arh2/2)*X(arh2/2)**0.5
#plt.loglog(10**np.arange(1,10,0.1),table(lambda a: sigma_k*rhocan(a)*X(a)**-0.5/Ht(a),10**np.arange(1,10,0.1)))
#plt.loglog(10**np.arange(1,10,0.1),1+0*10**np.arange(1,10,0.1))
akd_n=optimize.fsolve(lambda t: sigma_k*rhocan(t)*X(t)**-0.5-5000*Ht(t),10**3)[0]# this is the scale factor when we can numerically start handling the coupling term
akd=optimize.fsolve(lambda t: sigma_k*rhocan(t)*X(t)**-0.5-Ht(t),10**4)[0] #scale factor when kinetic decoupling occurs
#akd=0.1
#akd_n=0.1
# =============================================================================
#Perturbation equations
ddel,ddelsim,ddel_tk,ddelsim_tk,ddel_post_rh=pert_tk(rhocan,rhor,rhodm,Ht,c2s,w,gamma,sigma_k,X)

# =============================================================================
#Solving perturbation equations

#analytical max peak for adom>afz
csfz=np.sqrt(c2s(20*afz))*20
lambda1=np.real(3*csfz/2/np.sqrt(2)*(lambertw(2*(0.594/3/csfz*adom/afz)**(2/3)))**(3/2))
apk=lambda1*afz
kpk=apk*Ht(apk)
ahoru_pk=(5*arh)**2*Ht(5*arh)/(apk*Ht(apk))
peak=3/2*np.log(4*0.594*np.exp(-3)*adom/apk)*arh/adom*1.37*(1+np.log(ahoru_pk*1.58/0.594/arh)/np.log(4*0.594*np.exp(-3)*aeq/ahoru_pk))


# Calculating transfer function
j=0
kstart=5*arh*Ht(5*arh)
krh=arh*Ht(arh)
k0=1*Ht(1)
kfz=afz*Ht(afz)
aosc=np.sqrt(adom/arh*c2s(afz))*afz
kosc=aosc*Ht(aosc)*1.5
#K=10**np.arange(log(kstart)/log(10),log(k0)/log(10),0.01) #step size 0.1 for ECDE and 0.05 for ECDE+EMDE
K=10**np.concatenate((np.arange(log(kstart)/log(10),log(kfz)/log(10),0.05),np.arange(log(kfz)/log(10),log(kosc)/log(10),0.01),np.arange(log(kosc)/log(10),log(k0)/log(10),0.1)))
#K=10**np.arange(log(kpk/2)/log(10),log(kpk*2)/log(10),0.01)
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
        if akd_n<arh:
            astart=min(1,ahz/10)
            delta0=[1,-2,k**2/2/astart/Ht(astart),-2/(1+w(astart)),-2,k**2/2/astart/Ht(astart)]
            delta=solve_ivp(lambda a,y: ddel_tk(a,y,k),[astart,min(100*ahz,akd_n)],delta0,method='BDF',dense_output=False,atol=1e-8,rtol=1e-7)# Complete perturbation equations solving till 100ahz
            if akd_n>100*ahz:
                deltasim0=[delta.y[0,-1],delta.y[1,-1],delta.y[2,-1],delta.y[3,-1]]#initial condition  deep in subhorizon
                deltasim_tk=solve_ivp(lambda a,y: ddelsim_tk(a,y,k),[delta.t[-1],akd_n],deltasim0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5)#ignore radiation perturbations completely
                deltasim0=[deltasim_tk.y[0,-1],deltasim_tk.y[1,-1],deltasim_tk.y[2,-1],deltasim_tk.y[3,-1],deltasim_tk.y[2,-1]]#initial condition  deep in subhorizon
                a_last=deltasim_tk.t[-1]
            else:
                delta0=[delta.y[0,-1],delta.y[1,-1],delta.y[2,-1],delta.y[3,-1],delta.y[2,-1],delta.y[4,-1],delta.y[5,-1]]
                delta2=solve_ivp(lambda a,y: ddel(a,y,k),[delta.t[-1],100*ahz],delta0,method='BDF',dense_output=False,atol=1e-8,rtol=1e-7)# Complete perturbation equations solving till 100ahz
                deltasim0=[delta2.y[0,-1],delta2.y[1,-1],delta2.y[2,-1],delta2.y[3,-1],delta2.y[4,-1]]#initial condition  deep in subhorizon
                a_last=delta2.t[-1]
            deltasim=solve_ivp(lambda a,y: ddelsim(a,y,k),[a_last,5*arh],deltasim0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5)#ignore radiation perturbations completely
        else:
            astart=min(1,ahz/10)
            delta0=[1,-2,k**2/2,-2/(1+w(astart)),-2,k**2/2]
            delta=solve_ivp(lambda a,y: ddel_tk(a,y,k),[astart,100*ahz],delta0,method='BDF',dense_output=False,atol=1e-8,rtol=1e-7)# Complete perturbation equations solving till 100ahz
            deltasim0=[delta.y[0,-1],delta.y[1,-1],delta.y[2,-1],delta.y[3,-1]]#initial condition  deep in subhorizon
            deltasim_tk=solve_ivp(lambda a,y: ddelsim_tk(a,y,k),[delta.t[-1],akd_n],deltasim0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5)#ignore radiation perturbations completely
            deltasim0=[deltasim_tk.y[0,-1],deltasim_tk.y[1,-1],deltasim_tk.y[2,-1],deltasim_tk.y[3,-1],deltasim_tk.y[2,-1]]#initial condition  deep in subhorizon
            deltasim=solve_ivp(lambda a,y: ddelsim(a,y,k),[deltasim_tk.t[-1],5*arh],deltasim0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5)#ignore radiation perturbations completely
        delta_dmf[j]=deltasim.y[3,-1] #find final DM_density
        delta_dm_primef[j]=(deltasim.y[3,-1]-deltasim.y[3,-2])/(deltasim.t[-1]-deltasim.t[-2])#find final derivative of DM density
        aend[j]=deltasim.t[-1]
    else: #for modes entering the horizon close to reheating we solve the full equation till reheating occurs and then ignore cannibal
        astart=1#min(ahz/20,arh/100)
        a_end=max(ahz*100,10*arh)
        delta0=[1,-2,k**2/2,-2/(1+w(astart)),-2,k**2/2]
        delta=solve_ivp(lambda a,y: ddel_tk(a,y,k),[astart,akd_n],delta0,method='RK45',dense_output=False,atol=1e-7,rtol=1e-7)
        delta0=[delta.y[0,-1],delta.y[1,-1],delta.y[2,-1],delta.y[3,-1],delta.y[2,-1],delta.y[4,-1],delta.y[5,-1]]
        delta2=solve_ivp(lambda a,y: ddel(a,y,k),[delta.t[-1],5*arh],delta0,method='RK45',dense_output=False,atol=1e-7,rtol=1e-7)# Complete perturbation equations
        delta_post_rh0=[delta2.y[0,-1],delta2.y[3,-1],delta2.y[4,-1],delta2.y[5,-1],delta2.y[6,-1]]
        delta_post_rh=solve_ivp(lambda a,y: ddel_post_rh(a,y,k),[delta2.t[-1],a_end],delta_post_rh0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5) #after reheating ignore cannibal perturbation equations
        delta_dmf[j]=delta_post_rh.y[1,-1]
        delta_dm_primef[j]=(delta_post_rh.y[1,-1]-delta_post_rh.y[1,-2])/(delta_post_rh.t[-1]-delta_post_rh.t[-2])
        aend[j]=delta_post_rh.t[-1]
    j=j+1
    print(j)
    
A=aend*delta_dm_primef
B=1/(aend)*np.exp(delta_dmf/A)
ahoru=(5*arh)**2*Ht(5*arh)/K #finding horizon entry of DM perturbation in a universe without cannibal
relative_delta_DM=A/9.11*(1+(np.log(ahoru/0.594/aend)+delta_dmf/A)/np.log(4*0.594*np.exp(-3)*aeq/ahoru))#finding ratio of DM density wrt t scenaario where there was no cannibal and only radiation domination in the early universe
# =============================================================================

# =============================================================================
# plotting all density perturbations for specific mode
k1=kpk/10
k2=kpk
k3=kpk*5.5


k=k1#afz/20*Ht(afz/20)##(3/2*(1+w(2*afz))/c2s(2*afz))**0.5*(2*afz*Ht(2*afz))/1.2#afz/2*Ht(afz/2)
ahz=optimize.fsolve(lambda a: a*Ht(a)-k,10)[0]
if ahz<arh/100:#For modes entering the horizon before reheating we completely ignore radiation perturbation
    if akd_n<arh:
        astart=min(1,ahz/10)
        delta0=[1,-2,k**2/2/astart/Ht(astart),-2/(1+w(astart)),-2,k**2/2/astart/Ht(astart)]
        delta=solve_ivp(lambda a,y: ddel_tk(a,y,k),[astart,min(100*ahz,akd_n)],delta0,method='BDF',dense_output=False,atol=1e-8,rtol=1e-7)# Complete perturbation equations solving till 100ahz
        if akd_n>100*ahz: # For these modes first we solve tightly coupled simplified equations till akd_n and then solve the simplified ignoring radiation till reheating and then simple logarithmic DM
            deltasim0=[delta.y[0,-1],delta.y[1,-1],delta.y[2,-1],delta.y[3,-1]]#initial condition  deep in subhorizon
            deltasim_tk=solve_ivp(lambda a,y: ddelsim_tk(a,y,k),[delta.t[-1],akd_n],deltasim0,method='BDF',dense_output=False,atol=1e-8,rtol=1e-7)#ignore radiation perturbations completely
            delta_dm=np.append(delta.y[3,:],deltasim_tk.y[3,:])
            a_full=np.append(delta.t,deltasim_tk.t)
            delta_r=delta.y[4,:]
            a_delta_r=delta.t
            delta_can=np.append(delta.y[1,:],deltasim_tk.y[1,:])
            a_delta_can=np.append(delta.t[:],deltasim_tk.t[:])
            deltasim0=[deltasim_tk.y[0,-1],deltasim_tk.y[1,-1],deltasim_tk.y[2,-1],deltasim_tk.y[3,-1],deltasim_tk.y[2,-1]]#initial condition  deep in subhorizon
            a_last=deltasim_tk.t[-1]
        else:
            delta0=[delta.y[0,-1],delta.y[1,-1],delta.y[2,-1],delta.y[3,-1],delta.y[2,-1],delta.y[4,-1],delta.y[5,-1]]
            delta2=solve_ivp(lambda a,y: ddel(a,y,k),[delta.t[-1],100*ahz],delta0,method='BDF',dense_output=False,atol=1e-8,rtol=1e-7)# Complete perturbation equations solving till 100ahz
            delta_dm=np.append(delta.y[3,:],delta2.y[3,:])
            a_full=np.append(delta.t,delta2.t)
            delta_r=np.append(delta.y[4,:],delta2.y[5,:])
            a_delta_r=np.append(delta.t,delta2.t)
            delta_can=np.append(delta.y[1,:],delta2.y[1,:])
            a_delta_can=np.append(delta.t[:],delta2.t[:])
            deltasim0=[delta2.y[0,-1],delta2.y[1,-1],delta2.y[2,-1],delta2.y[3,-1],delta2.y[4,-1]]#initial condition  deep in subhorizon
            a_last=delta2.t[-1]
        deltasim=solve_ivp(lambda a,y: ddelsim(a,y,k),[a_last,5*arh],deltasim0,method='BDF',dense_output=False,atol=1e-10,rtol=1e-10)#ignore radiation perturbations completely
        A=deltasim.t[-1]*(deltasim.y[3,-1]-deltasim.y[3,-2])/(deltasim.t[-1]-deltasim.t[-2]) #Find A and B for dm density evolving as \delta=A*log(B*a) after reheating
        B=1/(deltasim.t[-1])*np.exp(deltasim.y[3,-1]/A)#finding B
        deltadm_post_rh=A*np.log(B*np.arange(deltasim.t[-1],100*arh,arh))# finding \delta_DM from arh to 1000arh using \delta=A*log(B*a)
        delta_dm=np.concatenate((delta_dm,deltasim.y[3,:],deltadm_post_rh[:]),axis=0)# joinging the post reheating DM density evolution with pre-reheating
        a_full=np.concatenate((a_full,deltasim.t,np.arange(deltasim.t[-1],100*arh,arh)),axis=0)
        delta_can=np.append(delta_can,deltasim.y[1,:])
        a_delta_can=np.append(a_delta_can,deltasim.t[:])#find time when we end the simulation
    else:
        astart=min(1,ahz/10)
        delta0=[1,-2,k**2/2,-2/(1+w(astart)),-2,k**2/2]
        delta=solve_ivp(lambda a,y: ddel_tk(a,y,k),[astart,100*ahz],delta0,method='BDF',dense_output=False,atol=1e-8,rtol=1e-7)# Complete perturbation equations solving till 100ahz
        deltasim0=[delta.y[0,-1],delta.y[1,-1],delta.y[2,-1],delta.y[3,-1]]#initial condition  deep in subhorizon
        deltasim_tk=solve_ivp(lambda a,y: ddelsim_tk(a,y,k),[delta.t[-1],akd_n],deltasim0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5)#ignore radiation perturbations completely
        deltasim0=[deltasim_tk.y[0,-1],deltasim_tk.y[1,-1],deltasim_tk.y[2,-1],deltasim_tk.y[3,-1],deltasim_tk.y[2,-1]]#initial condition  deep in subhorizon
        deltasim=solve_ivp(lambda a,y: ddelsim(a,y,k),[deltasim_tk.t[-1],5*arh],deltasim0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5)#ignore radiation perturbations completely
        A=deltasim.t[-1]*(deltasim.y[3,-1]-deltasim.y[3,-2])/(deltasim.t[-1]-deltasim.t[-2]) #Find A and B for dm density evolving as \delta=A*log(B*a) after reheating
        B=1/(deltasim.t[-1])*np.exp(deltasim.y[3,-1]/A)#finding B
        deltadm_post_rh=A*np.log(B*np.arange(deltasim.t[-1],100*arh,arh))# finding \delta_DM from arh to 1000arh using \delta=A*log(B*a)
        delta_dm=np.concatenate((delta.y[3,:],deltasim_tk.y[3,:],deltasim.y[3,:],deltadm_post_rh[:]),axis=0)# joinging the post reheating DM density evolution with pre-reheating
        a_full=np.concatenate((delta.t[:],deltasim_tk.t[:],deltasim.t[:],np.arange(deltasim.t[-1],100*arh,arh)),axis=0)
        delta_r=delta.y[5,:]
        a_delta_r=delta.t
        delta_can=np.concatenate((delta.y[1,:],deltasim_tk.y[1,:],deltasim.y[1,:]),axis=0)
        a_delta_can=np.concatenate((delta.t[:],deltasim_tk.t[:],deltasim.t[:]),axis=0)#find time when we end the simulation            
else: #for modes entering the horizon close to reheating we solve the full equation till reheating occurs and then ignore cannibal
    astart=1#min(ahz/20,arh/100)
    aend=max(ahz*100,100*arh)
    delta0=[1,-2,k**2/2,-2/(1+w(astart)),-2,k**2/2]
    delta=solve_ivp(lambda a,y: ddel_tk(a,y,k),[astart,akd_n],delta0,method='RK45',dense_output=False,atol=1e-7,rtol=1e-7)
    delta0=[delta.y[0,-1],delta.y[1,-1],delta.y[2,-1],delta.y[3,-1],delta.y[2,-1],delta.y[4,-1],delta.y[5,-1]]
    delta2=solve_ivp(lambda a,y: ddel(a,y,k),[delta.t[-1],5*arh],delta0,method='RK45',dense_output=False,atol=1e-7,rtol=1e-7)# Complete perturbation equations
    delta_post_rh0=[delta2.y[0,-1],delta2.y[3,-1],delta2.y[4,-1],delta2.y[5,-1],delta2.y[6,-1]]
    delta_post_rh=solve_ivp(lambda a,y: ddel_post_rh(a,y,k),[delta2.t[-1],aend],delta_post_rh0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5) #after reheating ignore cannibal perturbation equations
    a_full=np.concatenate((delta.t[:],delta2.t[:],delta_post_rh.t[:]),axis=0)
    delta_dm=np.concatenate((delta.y[3,:],delta2.y[3,:],delta_post_rh.y[1,:]),axis=0)# joinging the post reheating DM density evolution with pre-reheating
    delta_r=np.concatenate((delta.y[4,:],delta2.y[5,:],delta_post_rh.y[3,:]),axis=0)# joinging the post reheating radiation density evolution with pre-reheating
    a_delta_r=a_full
    delta_can=np.concatenate((delta.y[1,:],delta2.y[1,:]),axis=0)
    a_delta_can=np.concatenate((delta.t,delta2.t),axis=0)

delta_can1=delta_can
a_delta_can1=a_delta_can
a_full1=a_full
delta_dm1=delta_dm

k=k2
ahz=optimize.fsolve(lambda a: a*Ht(a)-k,10)[0]
if ahz<arh/100:#For modes entering the horizon before reheating we completely ignore radiation perturbation
    if akd_n<arh:
        astart=min(1,ahz/10)
        delta0=[1,-2,k**2/2/astart/Ht(astart),-2/(1+w(astart)),-2,k**2/2/astart/Ht(astart)]
        delta=solve_ivp(lambda a,y: ddel_tk(a,y,k),[astart,min(100*ahz,akd_n)],delta0,method='BDF',dense_output=False,atol=1e-8,rtol=1e-7)# Complete perturbation equations solving till 100ahz
        if akd_n>100*ahz: # For these modes first we solve tightly coupled simplified equations till akd_n and then solve the simplified ignoring radiation till reheating and then simple logarithmic DM
            deltasim0=[delta.y[0,-1],delta.y[1,-1],delta.y[2,-1],delta.y[3,-1]]#initial condition  deep in subhorizon
            deltasim_tk=solve_ivp(lambda a,y: ddelsim_tk(a,y,k),[delta.t[-1],akd_n],deltasim0,method='BDF',dense_output=False,atol=1e-8,rtol=1e-7)#ignore radiation perturbations completely
            delta_dm=np.append(delta.y[3,:],deltasim_tk.y[3,:])
            a_full=np.append(delta.t,deltasim_tk.t)
            delta_r=delta.y[4,:]
            a_delta_r=delta.t
            delta_can=np.append(delta.y[1,:],deltasim_tk.y[1,:])
            a_delta_can=np.append(delta.t[:],deltasim_tk.t[:])
            deltasim0=[deltasim_tk.y[0,-1],deltasim_tk.y[1,-1],deltasim_tk.y[2,-1],deltasim_tk.y[3,-1],deltasim_tk.y[2,-1]]#initial condition  deep in subhorizon
            a_last=deltasim_tk.t[-1]
        else:
            delta0=[delta.y[0,-1],delta.y[1,-1],delta.y[2,-1],delta.y[3,-1],delta.y[2,-1],delta.y[4,-1],delta.y[5,-1]]
            delta2=solve_ivp(lambda a,y: ddel(a,y,k),[delta.t[-1],100*ahz],delta0,method='BDF',dense_output=False,atol=1e-8,rtol=1e-7)# Complete perturbation equations solving till 100ahz
            delta_dm=np.append(delta.y[3,:],delta2.y[3,:])
            a_full=np.append(delta.t,delta2.t)
            delta_r=np.append(delta.y[4,:],delta2.y[5,:])
            a_delta_r=np.append(delta.t,delta2.t)
            delta_can=np.append(delta.y[1,:],delta2.y[1,:])
            a_delta_can=np.append(delta.t[:],delta2.t[:])
            deltasim0=[delta2.y[0,-1],delta2.y[1,-1],delta2.y[2,-1],delta2.y[3,-1],delta2.y[4,-1]]#initial condition  deep in subhorizon
            a_last=delta2.t[-1]
        deltasim=solve_ivp(lambda a,y: ddelsim(a,y,k),[a_last,5*arh],deltasim0,method='BDF',dense_output=False,atol=1e-10,rtol=1e-10)#ignore radiation perturbations completely
        A=deltasim.t[-1]*(deltasim.y[3,-1]-deltasim.y[3,-2])/(deltasim.t[-1]-deltasim.t[-2]) #Find A and B for dm density evolving as \delta=A*log(B*a) after reheating
        B=1/(deltasim.t[-1])*np.exp(deltasim.y[3,-1]/A)#finding B
        deltadm_post_rh=A*np.log(B*np.arange(deltasim.t[-1],100*arh,arh))# finding \delta_DM from arh to 1000arh using \delta=A*log(B*a)
        delta_dm=np.concatenate((delta_dm,deltasim.y[3,:],deltadm_post_rh[:]),axis=0)# joinging the post reheating DM density evolution with pre-reheating
        a_full=np.concatenate((a_full,deltasim.t,np.arange(deltasim.t[-1],100*arh,arh)),axis=0)
        delta_can=np.append(delta_can,deltasim.y[1,:])
        a_delta_can=np.append(a_delta_can,deltasim.t[:])#find time when we end the simulation
    else:
        astart=min(1,ahz/10)
        delta0=[1,-2,k**2/2,-2/(1+w(astart)),-2,k**2/2]
        delta=solve_ivp(lambda a,y: ddel_tk(a,y,k),[astart,100*ahz],delta0,method='BDF',dense_output=False,atol=1e-8,rtol=1e-7)# Complete perturbation equations solving till 100ahz
        deltasim0=[delta.y[0,-1],delta.y[1,-1],delta.y[2,-1],delta.y[3,-1]]#initial condition  deep in subhorizon
        deltasim_tk=solve_ivp(lambda a,y: ddelsim_tk(a,y,k),[delta.t[-1],akd_n],deltasim0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5)#ignore radiation perturbations completely
        deltasim0=[deltasim_tk.y[0,-1],deltasim_tk.y[1,-1],deltasim_tk.y[2,-1],deltasim_tk.y[3,-1],deltasim_tk.y[2,-1]]#initial condition  deep in subhorizon
        deltasim=solve_ivp(lambda a,y: ddelsim(a,y,k),[deltasim_tk.t[-1],5*arh],deltasim0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5)#ignore radiation perturbations completely
        A=deltasim.t[-1]*(deltasim.y[3,-1]-deltasim.y[3,-2])/(deltasim.t[-1]-deltasim.t[-2]) #Find A and B for dm density evolving as \delta=A*log(B*a) after reheating
        B=1/(deltasim.t[-1])*np.exp(deltasim.y[3,-1]/A)#finding B
        deltadm_post_rh=A*np.log(B*np.arange(deltasim.t[-1],100*arh,arh))# finding \delta_DM from arh to 1000arh using \delta=A*log(B*a)
        delta_dm=np.concatenate((delta.y[3,:],deltasim_tk.y[3,:],deltasim.y[3,:],deltadm_post_rh[:]),axis=0)# joinging the post reheating DM density evolution with pre-reheating
        a_full=np.concatenate((delta.t[:],deltasim_tk.t[:],deltasim.t[:],np.arange(deltasim.t[-1],100*arh,arh)),axis=0)
        delta_r=delta.y[5,:]
        a_delta_r=delta.t
        delta_can=np.concatenate((delta.y[1,:],deltasim_tk.y[1,:],deltasim.y[1,:]),axis=0)
        a_delta_can=np.concatenate((delta.t[:],deltasim_tk.t[:],deltasim.t[:]),axis=0)#find time when we end the simulation            
else: #for modes entering the horizon close to reheating we solve the full equation till reheating occurs and then ignore cannibal
    astart=1#min(ahz/20,arh/100)
    aend=max(ahz*100,100*arh)
    delta0=[1,-2,k**2/2,-2/(1+w(astart)),-2,k**2/2]
    delta=solve_ivp(lambda a,y: ddel_tk(a,y,k),[astart,akd_n],delta0,method='RK45',dense_output=False,atol=1e-7,rtol=1e-7)
    delta0=[delta.y[0,-1],delta.y[1,-1],delta.y[2,-1],delta.y[3,-1],delta.y[2,-1],delta.y[4,-1],delta.y[5,-1]]
    delta2=solve_ivp(lambda a,y: ddel(a,y,k),[delta.t[-1],5*arh],delta0,method='RK45',dense_output=False,atol=1e-7,rtol=1e-7)# Complete perturbation equations
    delta_post_rh0=[delta2.y[0,-1],delta2.y[3,-1],delta2.y[4,-1],delta2.y[5,-1],delta2.y[6,-1]]
    delta_post_rh=solve_ivp(lambda a,y: ddel_post_rh(a,y,k),[delta2.t[-1],aend],delta_post_rh0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5) #after reheating ignore cannibal perturbation equations
    a_full=np.concatenate((delta.t[:],delta2.t[:],delta_post_rh.t[:]),axis=0)
    delta_dm=np.concatenate((delta.y[3,:],delta2.y[3,:],delta_post_rh.y[1,:]),axis=0)# joinging the post reheating DM density evolution with pre-reheating
    delta_r=np.concatenate((delta.y[4,:],delta2.y[5,:],delta_post_rh.y[3,:]),axis=0)# joinging the post reheating radiation density evolution with pre-reheating
    a_delta_r=a_full
    delta_can=np.concatenate((delta.y[1,:],delta2.y[1,:]),axis=0)
    a_delta_can=np.concatenate((delta.t,delta2.t),axis=0)

delta_can2=delta_can
a_delta_can2=a_delta_can
a_full2=a_full
delta_dm2=delta_dm

k=k3
ahz=optimize.fsolve(lambda a: a*Ht(a)-k,10)[0]
if ahz<arh/100:#For modes entering the horizon before reheating we completely ignore radiation perturbation
    if akd_n<arh:
        astart=min(1,ahz/10)
        delta0=[1,-2,k**2/2/astart/Ht(astart),-2/(1+w(astart)),-2,k**2/2/astart/Ht(astart)]
        delta=solve_ivp(lambda a,y: ddel_tk(a,y,k),[astart,min(100*ahz,akd_n)],delta0,method='BDF',dense_output=False,atol=1e-8,rtol=1e-7)# Complete perturbation equations solving till 100ahz
        if akd_n>100*ahz: # For these modes first we solve tightly coupled simplified equations till akd_n and then solve the simplified ignoring radiation till reheating and then simple logarithmic DM
            deltasim0=[delta.y[0,-1],delta.y[1,-1],delta.y[2,-1],delta.y[3,-1]]#initial condition  deep in subhorizon
            deltasim_tk=solve_ivp(lambda a,y: ddelsim_tk(a,y,k),[delta.t[-1],akd_n],deltasim0,method='BDF',dense_output=False,atol=1e-8,rtol=1e-7)#ignore radiation perturbations completely
            delta_dm=np.append(delta.y[3,:],deltasim_tk.y[3,:])
            a_full=np.append(delta.t,deltasim_tk.t)
            delta_r=delta.y[4,:]
            a_delta_r=delta.t
            delta_can=np.append(delta.y[1,:],deltasim_tk.y[1,:])
            a_delta_can=np.append(delta.t[:],deltasim_tk.t[:])
            deltasim0=[deltasim_tk.y[0,-1],deltasim_tk.y[1,-1],deltasim_tk.y[2,-1],deltasim_tk.y[3,-1],deltasim_tk.y[2,-1]]#initial condition  deep in subhorizon
            a_last=deltasim_tk.t[-1]
        else:
            delta0=[delta.y[0,-1],delta.y[1,-1],delta.y[2,-1],delta.y[3,-1],delta.y[2,-1],delta.y[4,-1],delta.y[5,-1]]
            delta2=solve_ivp(lambda a,y: ddel(a,y,k),[delta.t[-1],100*ahz],delta0,method='BDF',dense_output=False,atol=1e-8,rtol=1e-7)# Complete perturbation equations solving till 100ahz
            delta_dm=np.append(delta.y[3,:],delta2.y[3,:])
            a_full=np.append(delta.t,delta2.t)
            delta_r=np.append(delta.y[4,:],delta2.y[5,:])
            a_delta_r=np.append(delta.t,delta2.t)
            delta_can=np.append(delta.y[1,:],delta2.y[1,:])
            a_delta_can=np.append(delta.t[:],delta2.t[:])
            deltasim0=[delta2.y[0,-1],delta2.y[1,-1],delta2.y[2,-1],delta2.y[3,-1],delta2.y[4,-1]]#initial condition  deep in subhorizon
            a_last=delta2.t[-1]
        deltasim=solve_ivp(lambda a,y: ddelsim(a,y,k),[a_last,5*arh],deltasim0,method='BDF',dense_output=False,atol=1e-10,rtol=1e-10)#ignore radiation perturbations completely
        A=deltasim.t[-1]*(deltasim.y[3,-1]-deltasim.y[3,-2])/(deltasim.t[-1]-deltasim.t[-2]) #Find A and B for dm density evolving as \delta=A*log(B*a) after reheating
        B=1/(deltasim.t[-1])*np.exp(deltasim.y[3,-1]/A)#finding B
        deltadm_post_rh=A*np.log(B*np.arange(deltasim.t[-1],100*arh,arh))# finding \delta_DM from arh to 1000arh using \delta=A*log(B*a)
        delta_dm=np.concatenate((delta_dm,deltasim.y[3,:],deltadm_post_rh[:]),axis=0)# joinging the post reheating DM density evolution with pre-reheating
        a_full=np.concatenate((a_full,deltasim.t,np.arange(deltasim.t[-1],100*arh,arh)),axis=0)
        delta_can=np.append(delta_can,deltasim.y[1,:])
        a_delta_can=np.append(a_delta_can,deltasim.t[:])#find time when we end the simulation
    else:
        astart=min(1,ahz/10)
        delta0=[1,-2,k**2/2,-2/(1+w(astart)),-2,k**2/2]
        delta=solve_ivp(lambda a,y: ddel_tk(a,y,k),[astart,100*ahz],delta0,method='BDF',dense_output=False,atol=1e-8,rtol=1e-7)# Complete perturbation equations solving till 100ahz
        deltasim0=[delta.y[0,-1],delta.y[1,-1],delta.y[2,-1],delta.y[3,-1]]#initial condition  deep in subhorizon
        deltasim_tk=solve_ivp(lambda a,y: ddelsim_tk(a,y,k),[delta.t[-1],akd_n],deltasim0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5)#ignore radiation perturbations completely
        deltasim0=[deltasim_tk.y[0,-1],deltasim_tk.y[1,-1],deltasim_tk.y[2,-1],deltasim_tk.y[3,-1],deltasim_tk.y[2,-1]]#initial condition  deep in subhorizon
        deltasim=solve_ivp(lambda a,y: ddelsim(a,y,k),[deltasim_tk.t[-1],5*arh],deltasim0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5)#ignore radiation perturbations completely
        A=deltasim.t[-1]*(deltasim.y[3,-1]-deltasim.y[3,-2])/(deltasim.t[-1]-deltasim.t[-2]) #Find A and B for dm density evolving as \delta=A*log(B*a) after reheating
        B=1/(deltasim.t[-1])*np.exp(deltasim.y[3,-1]/A)#finding B
        deltadm_post_rh=A*np.log(B*np.arange(deltasim.t[-1],100*arh,arh))# finding \delta_DM from arh to 1000arh using \delta=A*log(B*a)
        delta_dm=np.concatenate((delta.y[3,:],deltasim_tk.y[3,:],deltasim.y[3,:],deltadm_post_rh[:]),axis=0)# joinging the post reheating DM density evolution with pre-reheating
        a_full=np.concatenate((delta.t[:],deltasim_tk.t[:],deltasim.t[:],np.arange(deltasim.t[-1],100*arh,arh)),axis=0)
        delta_r=delta.y[5,:]
        a_delta_r=delta.t
        delta_can=np.concatenate((delta.y[1,:],deltasim_tk.y[1,:],deltasim.y[1,:]),axis=0)
        a_delta_can=np.concatenate((delta.t[:],deltasim_tk.t[:],deltasim.t[:]),axis=0)#find time when we end the simulation            
else: #for modes entering the horizon close to reheating we solve the full equation till reheating occurs and then ignore cannibal
    astart=1#min(ahz/20,arh/100)
    aend=max(ahz*100,100*arh)
    delta0=[1,-2,k**2/2,-2/(1+w(astart)),-2,k**2/2]
    delta=solve_ivp(lambda a,y: ddel_tk(a,y,k),[astart,akd_n],delta0,method='RK45',dense_output=False,atol=1e-7,rtol=1e-7)
    delta0=[delta.y[0,-1],delta.y[1,-1],delta.y[2,-1],delta.y[3,-1],delta.y[2,-1],delta.y[4,-1],delta.y[5,-1]]
    delta2=solve_ivp(lambda a,y: ddel(a,y,k),[delta.t[-1],5*arh],delta0,method='RK45',dense_output=False,atol=1e-7,rtol=1e-7)# Complete perturbation equations
    delta_post_rh0=[delta2.y[0,-1],delta2.y[3,-1],delta2.y[4,-1],delta2.y[5,-1],delta2.y[6,-1]]
    delta_post_rh=solve_ivp(lambda a,y: ddel_post_rh(a,y,k),[delta2.t[-1],aend],delta_post_rh0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5) #after reheating ignore cannibal perturbation equations
    a_full=np.concatenate((delta.t[:],delta2.t[:],delta_post_rh.t[:]),axis=0)
    delta_dm=np.concatenate((delta.y[3,:],delta2.y[3,:],delta_post_rh.y[1,:]),axis=0)# joinging the post reheating DM density evolution with pre-reheating
    delta_r=np.concatenate((delta.y[4,:],delta2.y[5,:],delta_post_rh.y[3,:]),axis=0)# joinging the post reheating radiation density evolution with pre-reheating
    a_delta_r=a_full
    delta_can=np.concatenate((delta.y[1,:],delta2.y[1,:]),axis=0)
    a_delta_can=np.concatenate((delta.t,delta2.t),axis=0)

delta_can3=delta_can
a_delta_can3=a_delta_can
a_full3=a_full
delta_dm3=delta_dm

# =============================================================================
# plotting
aplt=10**np.arange(0,np.log(10*arh)/np.log(10),0.1) #end point corresponds to last mode we have evaluated for rel DM growth
aplt2=10**np.arange(0,np.log(arh)/np.log(10),0.1)
lambda_J=lambda a:(3/2*(1+w(a))/c2s(a))**-0.5*(a*Ht(a))**-1
lambda_H=lambda a:(a*Ht(a))**-1
lambda_Hu=lambda a:(10*arh*Ht(10*arh))**-1*a/(10*arh)
k_osc=lambda a: pi/2/c2s(a)**0.5/lambertw(0.594*pi/2/c2s(a)**0.5)/lambda_H(a)
rs=lambda a: integrate.quad(lambda x: c2s(x)**0.5*lambda_H(x)/x, 0.01, a)[0]
lambda_Htable=table(lambda_H,aplt)
lambda_Jtable=table(lambda_J,aplt2)
lambda_Hutable=table(lambda_Hu,aplt)
k_osctable=table(k_osc,aplt2)
rs_table=table(rs,aplt2)
miny_hor=lambda_H(aplt[0])
maxy_hor=lambda_H(aplt[-1])
krh=arh2*Ht(arh2) #mode which eneters horizon at reheating
kdom=adom2*Ht(adom2) #mode entering horizon at adom

plt.rcParams.update({'font.size': 20})
abreak=np.where(aplt2>adom2)[0][0]

ax1=plt.subplot(221)
miny=miny_hor
maxy=maxy_hor
plt.loglog(aplt,lambda_Htable,label=r"Horizon")
plt.loglog(aplt2[0:abreak],lambda_Jtable[0:abreak],'--',linewidth=1, color='darkorange')
#plt.loglog(aplt2[0:abreak],1/k_osctable[0:abreak],'-',color='darkgoldenrod',label=r"$k_{osc}^{-1}$")
plt.loglog(aplt2[0:abreak],rs_table[0:abreak],'-',color='darkgoldenrod',label=r"$r_s$")
plt.loglog(aplt2[abreak:-1],lambda_Jtable[abreak:-1],color='darkorange',label=r"$k_J^{-1}$")
plt.loglog(aplt,lambda_Hutable,'--b',linewidth=0.5,label=r"$\Lambda$CDM Horizon")
plt.loglog([afz2,afz2],[miny,maxy],'--g')
plt.text(1.1*afz2,2*miny,r"$a_{fz}$")
plt.loglog([arh2,arh2],[miny,maxy],'--b')
plt.text(1.1*arh2,2*miny,r"$a_{rh}$")
plt.loglog([adom2,adom2],[miny,maxy],'--',color='purple')
plt.text(1.1*adom2,2*miny,r"$a_{dom}$")
plt.loglog([aplt[0],aplt[-1]],[k1**-1,k1**-1],'--',color='red')  
plt.text(aplt[-1]/6,k1**-1*2,r"$k_1$")
plt.loglog([aplt[0],aplt[-1]],[k2**-1,k2**-1],'--',color='black')  
plt.text(aplt[-1]/6,k2**-1*2,r"$k_{pk}$")
plt.loglog([aplt[0],aplt[-1]],[k3**-1,k3**-1],'--',color='blue')  
plt.text(aplt[-1]/6,k3**-1/5,r"$k_2$")
plt.fill_between(aplt, lambda_Htable, miny, facecolor='blue', alpha=0.1)
plt.fill_between(aplt2[0:abreak], rs_table[0:abreak], miny, facecolor='yellow', alpha=0.5)
plt.fill_between(aplt2[abreak:-1], lambda_Jtable[abreak:-1], miny, facecolor='yellow', alpha=0.5)
plt.ylim([miny,maxy])
plt.xlim([aplt[0],aplt[-1]])
#plt.xlabel(r"$a$")
plt.ylabel(r"$\frac{k_{hor,i}}{k}$",rotation=0, labelpad=10)
plt.legend()

ax3=plt.subplot(223,sharex=ax1)
miny=10**-2
maxy=10**7
plt.rcParams.update({'font.size': 20})
plt.loglog(a_delta_can1,abs(delta_can1),label=r"$\delta_c(k_1)$",color='red')
plt.loglog(a_delta_can2,abs(delta_can2),label=r"$\delta_c(k_{pk})$",color='black')
plt.loglog(a_delta_can3,abs(delta_can3),label=r"$\delta_c(k_2)$",color='blue')
plt.loglog(a_full1,abs(delta_dm1),'-.',label=r"$\delta_{DM}(k_1)$",color='red',alpha=0.5)
plt.loglog(a_full2,abs(delta_dm2),'-.',label=r"$\delta_{DM}(k_{pk})$",color='black',alpha=0.5)
plt.loglog(a_full3,abs(delta_dm3),'-.',label=r"$\delta_{DM}(k_2)$",color='blue',alpha=0.5)
plt.loglog([afz2,afz2],[miny,maxy],'--g')
plt.text(1.1*afz2,2*miny,r"$a_{fz}$")
plt.loglog([arh2,arh2],[miny,maxy],'--b')
plt.text(1.1*arh2,2*miny,r"$a_{rh}$")
#plt.loglog([acan,acan],[miny,maxy],'--r')
#plt.text(acan,2*miny,r"$a_{can}$")
plt.loglog([adom2,adom2],[miny,maxy],'--',color='purple')
plt.text(1.1*adom2,2*miny,r"$a_{dom}$")
plt.ylim([miny,maxy])
plt.xlabel(r"$a/a_i$")
plt.ylabel(r"$\frac{|\delta|}{\phi_P}$",rotation=0, labelpad=10)
plt.legend(ncol=2)
plt.show()

ax2=plt.subplot(222,sharey=ax1)
minx=0.1#np.min(abs(relative_delta_DM))
maxx=5*np.max(abs(relative_delta_DM))
plt.loglog(abs(relative_delta_DM),1/K)
#krad=10*Ht(10)
#plt.loglog([minx,maxx],[krad**-1,krad**-1],'--y')
#plt.text(maxx/2,krad**-1,r"$k_{rad}$")
plt.loglog([minx,maxx],[krh**-1,krh**-1],'--b')
plt.text(minx*1.2,1.5*krh**-1,r"$k_{rh}$")
plt.loglog([minx,maxx],[kdom**-1,kdom**-1],'--',color='purple')
plt.text(minx*1.2,1.5*kdom**-1,r"$k_{dom}$")
plt.loglog([minx,maxx],[kpk**-1,kpk**-1],'--k')  
plt.text(minx*1.2,1.5*kpk**-1,r"$k_{pk}$")
plt.xlim([minx,maxx])
#plt.ylim([1/K[0],1/K[-1]])
plt.loglog([peak,peak],[K[0]**-1,K[-1]**-1],'-.',color='black')
plt.ylabel(r"$\frac{k_{hor,i}}{k}$",rotation=0, labelpad=10)
plt.xlabel(r"$|T(k)|$")

ax4=plt.subplot(224)
miny=0.1
maxy=5*np.max(abs(relative_delta_DM))
plt.loglog(K,abs(relative_delta_DM),label=r'$a_{kd,DM}>a_{rh}$')
plt.loglog(K_000,abs(relative_delta_DM_000),label=r'$a_{kd,DM}=a_{rh}/100$')
plt.loglog(K_001,abs(relative_delta_DM_001),label=r'no DM-cannibal kinetic coupling')
plt.loglog([kpk,kpk],[miny,maxy],'--k')
plt.text(1.1*kpk,2*miny,r"$k_{pk}$")
plt.loglog([krh,krh],[miny,maxy],'--b')
plt.text(1.1*krh,2*miny,r"$k_{rh}$")
plt.loglog([kdom,kdom],[miny,maxy],'--',color='purple')
plt.text(1.1*kdom,2*miny,r"$k_{dom}$")
plt.loglog([K[0],K[-1]],[peak,peak],'-.',color='black')
plt.ylim([miny,maxy])
plt.xlabel(r"$k/k_{hor,i}$")
plt.ylabel(r"$|T(k)|$",rotation=90, labelpad=0)
plt.legend()

plt.gcf().set_size_inches(15, 10,forward=True)
plt.gcf().subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.97)
#
#mycwd = os.getcwd()
#os.chdir("..")
#plt.savefig("SM_dom_horz_pk.pdf")
#os.chdir(mycwd)