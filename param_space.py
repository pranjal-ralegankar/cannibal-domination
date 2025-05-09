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
xi=0.4 #0.3# 10
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

kdom_table=np.zeros([Trh_table.shape[0],m_table.shape[0]])
krh_table=np.zeros([Trh_table.shape[0],m_table.shape[0]])
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
            
            #DM fine structure
            if m>m_alphaDM2/10 and m<m_alphaDM*10:
                Y_inf=3.83*10**(-40)*aeq**3/m_DM*(1000/m)**4
                xf_guess=optimize.root(lambda x: (x/2/pi)**1.5*exp(-x)-Y_inf,10).x[0]
                sigmavx_DM=H1/100/m**3/m_DM*(xf_guess/Y_inf) # DM annihilation rate in MeV^-2
                alpha_DM_table[i,j]=m_DM**2/pi*m**2*sigmavx_DM
                alpha_DM_table2[i,j]=100**2/pi*m**2*sigmavx_DM #alpha_DM table for mdm=100m
            # =============================================================================
            #Solving perturbation equations
            if arh>max(adom,afz):
                if adom>afz:
                    csfz=np.sqrt(c2s(20*afz))*20
                    lambda1=np.real(3*csfz/2/np.sqrt(2)*(lambertw(2*(0.594/3/csfz*adom/afz)**(2/3)))**(3/2))
                    apk=lambda1*afz
                    kpk=apk*Ht(apk)
                else:
                    kpk=(3/2*(1+w(2*afz))/c2s(2*afz))**0.5*(2*afz*Ht(2*afz))/1.4
                
                # if adom>afz:
                #     ahoru_pk=(5*arh)**2*Ht(5*arh)/(apk*Ht(apk))
                #     T_kpk=3/2*np.log(4*0.594*np.exp(-3)*adom/apk)*arh/adom*1.533*(1+np.log(ahoru_pk*1.58/0.594/arh)/np.log(4*0.594*np.exp(-3)*aeq/ahoru_pk))
                # else:
                #     ahoru_pk=(5*arh)**2*Ht(5*arh)/kpk
                #     T_kpk=2.86*arh/afz/9.11*(1+np.log(ahoru_pk*1.55/0.594/arh)/np.log(4*0.594*np.exp(-3)*aeq/ahoru_pk))
                
                # if (abs(log(adom/afz))<log(10)) or arh<10*max(adom,afz): #i.e. if arh, adom, asfz are too close then we calculate T(kpk) numerically
                k=kpk
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
                T_kpk=A/9.11*(1+(np.log(ahoru/0.594/aend)+delta_dmf/A)/np.log(4*0.594*np.exp(-3)*aeq/ahoru)) #relative transfer function at kpk
            # =============================================================================
            
            arh_table[i,j]=arh
            adom_table[i,j]=adom
            afz_table[i,j]=afz
            if arh>max(adom,afz):
                hor_i=aeq*3402*1/H1*(1/1.564*10**-29)# horizon size at a_i in parsecs. aeq/a0=1/zeq=1/3402. 1Mev=2.47*10^27 pc^-1.
                Mpk_table[i,j]=1.62*10**-7*(kpk**-1*hor_i)**3# peak micro halo mass
                T_kpk_table[i,j]=T_kpk

                can_fs=X(arh)**-0.5*(integrate.quad(lambda a: arh/a**3/(pi**2/90*gsm(T_SM(a/aeq))*T_SM(a/aeq)**4+pi**2/90*3.36*T_SM(1)**4*(aeq/a)**3)**0.5*H1*Mpl, arh, 3403*aeq)[0]) #integrating until today
                DM_fs_table[i,j]=(1/10)**0.5*can_fs*kpk#assuming mDM=10m_can            
            
            can_scat_rate=lambda a:0.63*2**0.5/4/(2*pi)**2*(5/3*(4*pi*alpha_can))**2*X(a)**-0.5*rhocan(a)*pi**2/30*10**4*m/H1
            akd=optimize.fsolve(lambda a: Ht(a)-can_scat_rate(a),10)[0]
            akd_table[i,j]=akd
            
            kdom_table[i,j]=adom*Ht(adom)
            krh_table[i,j]=arh*Ht(arh)
            
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


# =============================================================================
# Plotting m,Trh parameter space
fig, ax1 = plt.subplots()
plt.rcParams.update({'font.size': 26})
if xi==10:
    cs=plt.contourf(m_table/10**3,Trh_table,abs(T_kpk_table),[1,10,100,10**3,10**4,10**np.rint((log(abs(T_kpk_table))/log(10)).max())],locator=ticker.LogLocator(), cmap='Blues', vmin=10**-4, vmax=10**9, zorder=1)
if xi<1:
    cs=plt.contourf(m_table/10**3,Trh_table,abs(T_kpk_table),[10,100,10**3,10**4,10**np.rint((log(abs(T_kpk_table))/log(10)).max())],locator=ticker.LogLocator(), cmap='Blues', vmin=10**-4, vmax=10**9, zorder=1)
if xi==1:
    cs=plt.contourf(m_table/10**3,Trh_table,abs(T_kpk_table),[1,10,100,10**3,10**4,10**np.rint((log(abs(T_kpk_table))/log(10)).max())],locator=ticker.LogLocator(), cmap='Blues', vmin=10**-4, vmax=10**9, zorder=1)

##only for plotting colorbar
# cs=plt.contourf(m_table/10**3,Trh_table,abs(T_kpk_table),[1,10,100,10**3,10**4],locator=ticker.LogLocator(), cmap='Blues', vmin=10**-4, vmax=10**9, zorder=1)
# cbar=plt.colorbar(cs,orientation="horizontal")
# cbar.set_label(r'$T(k_{pk})$')
# plt.gcf().set_size_inches(10,10,forward=True)
# plt.gcf().subplots_adjust(left=0.12, right=0.845, bottom=0.01, top=0.97)


plt.xscale('log')
plt.yscale('log')
plt.xlim([1,5*10**8])
plt.ylim([1,2*10**9])
plt.gcf().set_size_inches(10,8,forward=True)
plt.gcf().subplots_adjust(left=0.12, right=0.845, bottom=0.13, top=0.97)

#Mpk
if xi==10:
    cs2=plt.contour(m_table/10**3,Trh_table,log(Mpk_table)/log(10),6,colors='white',linestyles='dashed', zorder=11)
if xi<1:
    cs2=plt.contour(m_table/10**3,Trh_table,log(Mpk_table)/log(10),4,colors='white',linestyles='dashed', zorder=11)
if xi==1 and alpha_can==0.1:
    cs2=plt.contour(m_table/10**3,Trh_table,log(Mpk_table)/log(10),6,colors='white',linestyles='dashed', zorder=11)
if xi==1 and alpha_can==0.01:
    cs2=plt.contour(m_table/10**3,Trh_table,log(Mpk_table)/log(10),6,colors='white',linestyles='dashed', zorder=11)
fmt = r'$10^{%d}\ M_{\odot}$'
cl=plt.clabel(cs2, cs2.levels, fmt=fmt, fontsize='smaller', use_clabeltext=True, manual=True)

#alphaDM
cs3=plt.contour(m_table/10**3,Trh_table,alpha_DM_table,[1],colors='red',linestyles='dashed', linewidths=2.5, alpha=1,zorder=4)
if xi==10:
    fmt = r'$\alpha_{DM}=%d$'
    cl2=plt.clabel(cs3, cs3.levels, fmt=fmt, use_clabeltext=True, manual=True)
cs3=plt.contour(m_table/10**3,Trh_table,alpha_DM_table2,[1],colors='red',linestyles='dashdot',linewidths=2.5, dashes=(5, 50), alpha=1, zorder=4)
for c in cs3.collections:
    c.set_dashes([(0, (5, 10, 2, 10))])
    
#sigma_DM
# sigma_table=np.zeros([Trh_table.shape[0],m_table.shape[0]])
# for i in np.arange(0,Trh_table.shape[0]):
#     for j in np.arange(0,m_table.shape[0]):
#         if alpha_DM_table[i,j]<=1:
#             sigma_table[i,j]=alpha_DM_table[i,j]/m_DM**2/m_table[j]**2*pi

# cs9=plt.contour(m_table/10**3,Trh_table,log(sigma_table*10**6)/log(10),6,colors='green',linestyles='dashdot',linewidths=2.5, dashes=(5, 50), alpha=1, zorder=4)
# fmt = r'$10^{%d}$'
# cl=plt.clabel(cs9, cs9.levels, fmt=fmt, fontsize='smaller', use_clabeltext=True, manual=True)

# #kpk/krh 
# kpk_krh_table=np.zeros([Trh_table.shape[0],m_table.shape[0]])
# for i in np.arange(0,Trh_table.shape[0]):
#     for j in np.arange(0,m_table.shape[0]):
#             krh=(gsm_s(T0)/gsm_s(Trh_table[i]))**(1/3)*T0/Trh_table[i]*1/3**0.5/Mpl*sqrt(pi**2/30*gsm(Trh_table[i]))*Trh_table[i]**2*1.564*10**29
#             kpk_krh_table[i,j]=(Mpk_table[i,j]/(1.62*10**-7))**(-1/3)/krh
# cs9=plt.contour(m_table/10**3,Trh_table,log(kpk_krh_table)/log(10),6,colors='green',linestyles='dashdot',linewidths=2.5, dashes=(5, 50), alpha=1, zorder=4)
# fmt = r'$10^{%d}$'
# cl=plt.clabel(cs9, cs9.levels, fmt=fmt, fontsize='smaller', use_clabeltext=True, manual=True)

# #kdom/krh
# cs9=plt.contour(m_table/10**3,Trh_table,log(kdomkrh_table)/log(10),[0,1,2,3],colors='green',linestyles='dashdot',linewidths=2.5, dashes=(5, 50), alpha=1, zorder=4)
# fmt = r'$10^{%d}$'
# cl=plt.clabel(cs9, cs9.levels, fmt=fmt, fontsize='smaller', use_clabeltext=True, manual=True)

##Tk 56krh
# cs9=plt.contour(m_table/10**3,Trh_table,-Tk_56krh_table,[10,20,40,80,120,160,180],colors='green',linestyles='dashdot',linewidths=2.5, dashes=(5, 50), alpha=1, zorder=4)
# fmt = r'$%d$'
# cl=plt.clabel(cs9, cs9.levels, fmt=fmt, fontsize='smaller', use_clabeltext=True, manual=True)

#Tk 27krh
cs9=plt.contour(m_table/10**3,Trh_table,-Tk_27krh_table,[10,20,30,40,50],colors='green',linestyles='dashdot',linewidths=2.5, dashes=(5, 50), alpha=1, zorder=4)
fmt = r'$%d$'
cl=plt.clabel(cs9, cs9.levels, fmt=fmt, fontsize='smaller', use_clabeltext=True, manual=True)

#DM fs
cs4=plt.contour(m_table/10**3,Trh_table,DM_fs_table,[1],colors='k',linestyles='dashed', linewidths=2,alpha=1, zorder=5)
if xi==10:
    fmt = r'$k_{pk}=k_{fs}$'
    cl3=plt.clabel(cs4, cs4.levels, fmt=fmt, use_clabeltext=True, manual=True)
cs4=plt.contour(m_table/10**3,Trh_table,DM_fs_table,[10**0.5],colors='k',linestyles='dashdot',linewidths=2,alpha=1, zorder=5)
for c in cs4.collections:
    c.set_dashes([(0, (5, 10, 2, 10))])

#akd
cs3=plt.contour(m_table/10**3,Trh_table,akd_table/adom_table,[1],colors='orange',alpha=1,linestyles='dashed', zorder=12)
if xi<1:
    fmt = r'$a_{kd}=a_{dom}$'
    cl2=plt.clabel(cs3, cs3.levels, fmt=fmt, fontsize=24, use_clabeltext=True, manual=True)
    # for l in cl2:
    #     l.set_rotation(90)

if xi==10:
    plt.contourf(m_table/10**3,Trh_table,np.nan_to_num(arh_table/afz_table),[0,5], colors="white",zorder=16)#painting white over param space where arh<5asfz
if xi<=1:
    plt.contourf(m_table/10**3,Trh_table,np.nan_to_num(arh_table/adom_table),[0,2], colors="white",zorder=17)#painting white over param space where arh<5adom
plt.contourf(m_table/10**3,Trh_table,np.nan_to_num(afz_table),[0,100], colors="white",zorder=17)#painting white over param space where afz<acan

if xi<1:
    plt.text(5*10**6,100,r"$a_{fz}<a_{can}$",rotation=-90,zorder=18)
    plt.text(10000,1000,r"$a_{rh}<a_{dom}$",{'ha': 'center', 'va': 'center'},rotation=35,zorder=19)
    plt.text(m_table[0]/10**3*2,10**8,r"$\alpha_c=0.1$, $\xi_i=0.4$",zorder=20)
if xi==10:
    plt.text(2*10**8,3000,r"$a_{fz}<a_{can}$",rotation=-90,zorder=18)
    plt.text(60000,500000,r"$a_{rh}<5a_{fz}$",{'ha': 'center', 'va': 'center'},rotation=50,zorder=19)
    plt.text(m_table[0]/10**3*2,10**8,r"$\alpha_c=0.1$, $\xi_i=10$",zorder=20)
if xi==1 and alpha_can==0.1:
    plt.text(3*10**7,3000,r"$a_{fz}<a_{can}$",rotation=-90,zorder=18)
    plt.text(10000,20000,r"$a_{rh}<a_{dom}$",{'ha': 'center', 'va': 'center'},rotation=50,zorder=19)
    plt.text(m_table[0]/10**3*2,10**8,r"$\alpha_c=0.1$, $\xi_i=1$",zorder=20)
if xi==1 and alpha_can==0.01:
    plt.text(3*10**4,100,r"$a_{fz}<a_{can}$",rotation=-90,zorder=18)
    plt.text(600,1000,r"$a_{rh}<a_{dom}$",{'ha': 'center', 'va': 'center'},rotation=40,zorder=19)
    plt.text(m_table[0]/10**3*2,10**8,r"$\alpha_c=0.01$, $\xi_i=1$",zorder=20)
plt.xlabel(r"$m$ (GeV)")
plt.ylabel(r"$T_{rh}$ (MeV)",rotation=90, labelpad=5)
# plt.tick_params(axis='x', which='minor', bottom=False)

locmin = LogLocator(base=10.0,subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=17)
ax1.xaxis.set_minor_locator(locmin)
ax1.xaxis.set_minor_formatter(NullFormatter())
ax1.yaxis.set_minor_locator(locmin)
ax1.yaxis.set_minor_formatter(NullFormatter())
Mrh_table=table(lambda x: 10**-4*gsm_s(x)/10*(gsm(x)/10)**-1.5*(x/10)**-3,Trh_table)

def forward2(x):
    return np.interp(x, Trh_table, Mrh_table)

def inverse2(x):
    return np.interp(1/x, 1/Mrh_table, Trh_table)

secax = ax1.secondary_yaxis('right', functions=(forward2, inverse2))
# secax.set_ylabel(r'$M_{rh}$ ($M_{\odot}$)')
secax.yaxis.set_ticklabels([])
ax1.text(1.01, 0.08, r'$10^{-6}$',transform=ax1.transAxes)
ax1.text(1.01, 0.185, r'$10^{-9}$',transform=ax1.transAxes)
ax1.text(1.01, 0.305, r'$10^{-12}$',transform=ax1.transAxes)
ax1.text(1.01, 0.425, r'$10^{-15}$',transform=ax1.transAxes)
ax1.text(1.01, 0.54, r'$10^{-18}$',transform=ax1.transAxes)
ax1.text(1.01, 0.662, r'$10^{-21}$',transform=ax1.transAxes)
ax1.text(1.01, 0.785, r'$10^{-24}$',transform=ax1.transAxes)
ax1.text(1.01, 0.9, r'$10^{-27}$',transform=ax1.transAxes)
secax.set_ylabel(r'$M_{rh}$ ($M_{\odot}$)',labelpad=100, rotation=-90)

#analytical Mpk adom>afz
#Mpk_an=1.74*10**-4*(Trh/10)**(-1)*(m/1000)**(-11/4)*xi**(15/2)*alpha_can**(9/4)*(log(xi**-4.5*alpha_can**-0.75*(m/1000)**0.25))**6
#Analytical Mpk error becomes bad for adom close to asfz as well as for large afz.

#analyticcal T_kpk error stays within factor of 4.