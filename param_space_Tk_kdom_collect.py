# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 10:57:30 2019

@author: pranj
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
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
from scipy.special import lambertw
from matplotlib import ticker
import time
import pandas as pd

os.chdir('../../pranjal work/git-python/cannibal-base-code')
from background import bck, bck2
from equilibrium import wfun, Xcan
os.chdir('../../../draft/cannibal-long_paper_figs')
def table(f,t):
    tablef=np.zeros(t.shape[0])
    for i in np.arange(t.shape[0]):
        tablef[i]=f(t[i])
    return tablef

def ddel(a,delta,k): #full perturbation equations
    phi,dl,th,dldm,thdm,dlr,thr=delta[0],delta[1],delta[2],delta[3],delta[4],delta[5],delta[6] 
    del_phi=-phi/a+(-k**2*phi-3/2*a**2*(rhocan(a)*dl+rhor(a)*dlr+rhodm(a)*dldm)/(rhocan(1)+rhor(1)))/(3*(a*Ht(a))**2)/a
    del_dl=-(1+w(a))*(th/a**2/Ht(a)-3*del_phi)-3/a*(c2s(a)-w(a))*dl-gamma/a/Ht(a)*phi
    del_th=-(1-3*w(a))*th/a-derivative(w,(a+1))/(1 + w(a))*th+c2s(a)/(1 + w(a))*k**2/a**2/Ht(a)*dl+k**2/a**2/Ht(a)*phi
    del_dldm=-(thdm/a**2/Ht(a) - 3*del_phi)
    del_thdm=-thdm/a+k**2/a**2/Ht(a)*phi
    del_dlr= -4/3*(thr/a**2/Ht(a)-3*del_phi)+gamma*rhocan(a)/(a*Ht(a)*rhor(a))*(dl-dlr+phi)
    del_thr= 1/4*k**2/a**2/Ht(a)*dlr + k**2/a**2/Ht(a)*phi + gamma*rhocan(a)/(a*Ht(a)*rhor(a))*(3/4*th*(1 + w(a))-thr)
    return [del_phi,del_dl,del_th,del_dldm,del_thdm,del_dlr,del_thr]

def ddelsim(a,delta,k): #perturbations ignoring radiation componenet; Fails for DM perturbations entering horizon near reheating
    phi,dl,th,dldm,thdm=delta[0],delta[1],delta[2],delta[3],delta[4] 
    del_phi=-phi/a+(-k**2*phi-3/2*a**2*(rhocan(a)*dl)/(rhocan(1)+rhor(1)))/(3*(a*Ht(a))**2)/a
    del_dl=-(1+w(a))*(th/a**2/Ht(a)-3*del_phi)-3/a*(c2s(a)-w(a))*dl-gamma/a/Ht(a)*phi
    del_th=-(1-3*w(a))*th/a-derivative(w,(a+1))/(1 + w(a))*th+c2s(a)/(1 + w(a))*k**2/a**2/Ht(a)*dl+k**2/a**2/Ht(a)*phi
    del_dldm=-(thdm/a**2/Ht(a) - 3*del_phi)
    del_thdm=-thdm/a+k**2/a**2/Ht(a)*phi
    return [del_phi,del_dl,del_th,del_dldm,del_thdm]

def ddel_post_rh(a,delta,k): #post reheating perturbations without any cannibal component; Still ignoring DM perturbations in \phi
    phi,dldm,thdm,dlr,thr=delta[0],delta[1],delta[2],delta[3],delta[4]
    del_phi=-phi/a+(-k**2*phi-3/2*a**2*(rhor(a)*dlr+rhodm(a)*dldm)/(rhocan(1)+rhor(1)))/(3*(a*Ht(a))**2)/a
    del_dldm=-(thdm/a**2/Ht(a) - 3*del_phi)
    del_thdm=-thdm/a+k**2/a**2/Ht(a)*phi
    del_dlr= -4/3*(thr/a**2/Ht(a)-3*del_phi)
    del_thr= 1/4*k**2/a**2/Ht(a)*dlr + k**2/a**2/Ht(a)*phi
    return [del_phi,del_dldm,del_thdm,del_dlr,del_thr]

dof=pd.read_excel('../../pranjal work/git-python/cannibal-base-code/early universe degrees of freedom2.xlsx')
ge=np.array(dof.Ge.values)
T_mev=np.array(dof.K_bT_MeV.values)
gs=np.array(dof.gs.values)
T0=0.0002348221920804/10**6 #temp today in Mev
a_by_aeq=3403*(3.91/gs)**(1/3)*T0/T_mev
    
def T_SM(a): #Temperature of SM in MeV in usual cosmology as a function of a/aeq
    if a<a_by_aeq[0]:
        ans=T_mev[0]*a_by_aeq[0]/a
    elif a_by_aeq[0]<=a<=a_by_aeq[-1]:
        ans=np.exp(np.interp(np.log(a),np.log(a_by_aeq),np.log(T_mev)))
    else:
        ans=T_mev[-1]*a_by_aeq[-1]/a
    return ans

def gsm(T):
    if T>T_mev[0]:
        ans=ge[0]
    elif (T<T_mev[0]) and (T>T_mev[-1]):
        ans=1/np.interp(1/T,1/T_mev,1/ge)
    else:
        ans=ge[-1]
    return ans

def gsm_s(T):
    if T>T_mev[0]:
        ans=gs[0]
    elif (T<T_mev[0]) and (T>T_mev[-1]):
        ans=1/np.interp(1/T,1/T_mev,1/gs)
    else:
        ans=gs[-1]
    return ans
    
Mpl=2.435*10**21 #in MeV

alpha_can=0.1
xi=1 #0.3# 10
m_DM=10# setting M_DM/m

Trh_table=10**np.arange(log(8)/log(10),log(2*10**9)/log(10),0.1) #in Mev
m_table=10**np.arange(log(10**3)/log(10),log(10**12)/log(10),0.1) # in MeV
arh_table=np.zeros([Trh_table.shape[0],m_table.shape[0]])
afz_table=np.zeros([Trh_table.shape[0],m_table.shape[0]])
adom_table=np.zeros([Trh_table.shape[0],m_table.shape[0]])
alpha_DM_table=np.zeros([Trh_table.shape[0],m_table.shape[0]])
Mpk_table=np.zeros([Trh_table.shape[0],m_table.shape[0]])
T_kpk_table=np.zeros([Trh_table.shape[0],m_table.shape[0]])

DM_fs_table=np.zeros([Trh_table.shape[0],m_table.shape[0]])
alpha_DM_table2=np.zeros([Trh_table.shape[0],m_table.shape[0]])
akd_table=np.zeros([Trh_table.shape[0],m_table.shape[0]])

kdomkrh_table=np.zeros([Trh_table.shape[0],m_table.shape[0]])
Tk_27krh_table=np.zeros([Trh_table.shape[0],m_table.shape[0]])
Tk_56krh_table=np.zeros([Trh_table.shape[0],m_table.shape[0]])

i=0
print(Trh_table.shape[0])
for Trh in Trh_table:
    j=0
    for m in m_table:
        print("i=",i,", j=",j,)
        #limits of m
        m_afz=8*alpha_can**(3/7)*(Trh/10)**(6/7)*10**3 # arh>afz condition when afzz>adom
        m_adom=6/xi**3*(Trh/10)*10**3# arh>adom condtion when afz<adom
        m_alphaDM=10**10*(m_DM/10)**-2*(Trh/10)**-1/sqrt(1+np.interp(10*m/xi,np.flip(T_mev),np.flip(ge))/xi**4)*pi#alpha_DM<1 condition
        m_alphaDM2=10**10*(100/10)**-2*(Trh/10)**-1/sqrt(1+np.interp(10*m/xi,np.flip(T_mev),np.flip(ge))/xi**4)*pi#alpha_DM<1 condition for m_DM=100
        m_alphac1=8*10**12*xi**2*alpha_can**3 #adom>afz>100a_i
        m_alphac2=2*10**13*alpha_can**3 #afz>100a_i and afz>adom
        if m>m_alphaDM*10:
            alpha_DM_table[i,j]=10**3
            alpha_DM_table2[i,j]=10**3
        if m>m_afz/10 and m>m_adom/10 and m<m_alphac1*10 and m<m_alphac2*10:
            #solve background
            m,xi,alpha_can,afz,adom,arh,rhocan,rhor,X,w,c2s,Ht,rhodm,hor_i,aeq,gamma,H1=bck2(m,xi,alpha_can,Trh,0)
            
            arh_table[i,j]=arh
            adom_table[i,j]=adom
            afz_table[i,j]=afz
            
            can_scat_rate=lambda a:0.63*2**0.5/4/(2*pi)**2*(5/3*(4*pi*alpha_can))**2*X(a)**-0.5*rhocan(a)*pi**2/30*10**4*m/H1
            akd=optimize.fsolve(lambda a: Ht(a)-can_scat_rate(a),10)[0]
            akd_table[i,j]=akd
            
            kdomkrh_table[i,j]=adom*Ht(adom)/(arh*Ht(arh))
            
            # =============================================================================
            # extra code for Tk
            k=27*arh*Ht(arh)
            ahz=optimize.fsolve(lambda a: a*Ht(a)-k,10)[0]
            astart=1
            delta0=[1,-2,k**2/2,-2/(1+w(astart)),k**2/2,-2,k**2/2]
            delta=solve_ivp(lambda a,y: ddel(a,y,k),[astart,min(5*arh,100*ahz)],delta0,method='BDF',dense_output=False,atol=1e-8,rtol=1e-7)# Complete perturbation equations solving till 100ahz
            if 100*ahz<5*arh:
                deltasim0=[delta.y[0,-1],delta.y[1,-1],delta.y[2,-1],delta.y[3,-1],delta.y[4,-1]]#initial condition  deep in subhorizon
                deltasim=solve_ivp(lambda a,y: ddelsim(a,y,k),[delta.t[-1],5*arh],deltasim0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5)#ignore radiation perturbations completely
                A=deltasim.t[-1]*(deltasim.y[3,-1]-deltasim.y[3,-2])/(deltasim.t[-1]-deltasim.t[-2]) #Find A and B for dm density evolving as \delta=A*log(B*a) after reheating
                B=1/(deltasim.t[-1])*np.exp(deltasim.y[3,-1]/A)#finding B
                aend=deltasim.t[-1]
                delta_dmf=deltasim.y[3,-1]
            else:
                delta_post_rh0=[delta.y[0,-1],delta.y[3,-1],delta.y[4,-1],delta.y[5,-1],delta.y[6,-1]]
                delta_post_rh=solve_ivp(lambda a,y: ddel_post_rh(a,y,k),[delta.t[-1],100*ahz],delta_post_rh0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5)
                delta_dmf=delta_post_rh.y[1,-1]
                delta_dm_primef=(delta_post_rh.y[1,-1]-delta_post_rh.y[1,-2])/(delta_post_rh.t[-1]-delta_post_rh.t[-2])
                aend=delta_post_rh.t[-1]
                A=aend*delta_dm_primef
                B=1/(aend)*np.exp(delta_dmf/A)
                
            ahoru=(5*arh)**2*Ht(5*arh)/k #finding horizon entry of DM perturbation in a universe without cannibal
            Tk_27krh_table[i,j]=A/9.11*(1+(np.log(ahoru/0.594/aend)+delta_dmf/A)/np.log(4*0.594*np.exp(-3)*aeq/ahoru)) #relative transfer function at kpk
            
            k=56*arh*Ht(arh)
            ahz=optimize.fsolve(lambda a: a*Ht(a)-k,10)[0]
            astart=1
            delta0=[1,-2,k**2/2,-2/(1+w(astart)),k**2/2,-2,k**2/2]
            delta=solve_ivp(lambda a,y: ddel(a,y,k),[astart,min(5*arh,100*ahz)],delta0,method='BDF',dense_output=False,atol=1e-8,rtol=1e-7)# Complete perturbation equations solving till 100ahz
            if 100*ahz<5*arh:
                deltasim0=[delta.y[0,-1],delta.y[1,-1],delta.y[2,-1],delta.y[3,-1],delta.y[4,-1]]#initial condition  deep in subhorizon
                deltasim=solve_ivp(lambda a,y: ddelsim(a,y,k),[delta.t[-1],5*arh],deltasim0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5)#ignore radiation perturbations completely
                A=deltasim.t[-1]*(deltasim.y[3,-1]-deltasim.y[3,-2])/(deltasim.t[-1]-deltasim.t[-2]) #Find A and B for dm density evolving as \delta=A*log(B*a) after reheating
                B=1/(deltasim.t[-1])*np.exp(deltasim.y[3,-1]/A)#finding B
                aend=deltasim.t[-1]
                delta_dmf=deltasim.y[3,-1]
            else:
                delta_post_rh0=[delta.y[0,-1],delta.y[3,-1],delta.y[4,-1],delta.y[5,-1],delta.y[6,-1]]
                delta_post_rh=solve_ivp(lambda a,y: ddel_post_rh(a,y,k),[delta.t[-1],100*ahz],delta_post_rh0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5)
                delta_dmf=delta_post_rh.y[1,-1]
                delta_dm_primef=(delta_post_rh.y[1,-1]-delta_post_rh.y[1,-2])/(delta_post_rh.t[-1]-delta_post_rh.t[-2])
                aend=delta_post_rh.t[-1]
                A=aend*delta_dm_primef
                B=1/(aend)*np.exp(delta_dmf/A)
                
            ahoru=(5*arh)**2*Ht(5*arh)/k #finding horizon entry of DM perturbation in a universe without cannibal
            Tk_56krh_table[i,j]=A/9.11*(1+(np.log(ahoru/0.594/aend)+delta_dmf/A)/np.log(4*0.594*np.exp(-3)*aeq/ahoru)) #relative transfer function at kpk
        
            
        j=j+1
    i=i+1