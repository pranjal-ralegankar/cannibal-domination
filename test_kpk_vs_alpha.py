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
# fixing m and xi_i
# =============================================================================
m=30000
xi=1.5#3.330373288903145
Trh=8
alpha_table=10**np.arange(-3,0,0.1)

adom_afz_table=np.zeros(alpha_table.shape[0])
kpk_table2=np.zeros(alpha_table.shape[0])
kpk_dom_table2=np.zeros(alpha_table.shape[0])
kpk_fz_table2=np.zeros(alpha_table.shape[0])
rs_table2=np.zeros(alpha_table.shape[0])

i=0
for alpha_can in alpha_table:
    #running background evolution
    m,xi,alpha_can,afz,adom,arh,rhocan,rhor,X,w,c2s,Ht,rhodm,hor_i,aeq,gamma,H1=bck2(m,xi,alpha_can,Trh,0)
    #m is mass of cannibal in Mev

    csfz=np.sqrt(c2s(20*afz))*20
    lambda1=np.real(3*csfz/2/np.sqrt(2)*(lambertw(2*(0.594/3/csfz*adom/afz)**(2/3)))**(3/2))
    apk=lambda1*afz
    kpk_dom=apk*Ht(apk)
    rs=lambda a: integrate.quad(lambda x: c2s(x)**0.5*(x*Ht(x))**-1/x, 0.01, a)[0]
    
    kpk_fz=(3/2*(1+w(2*afz))/c2s(2*afz))**0.5*(2*afz*Ht(2*afz))/1.4
    if adom>2*afz:
        kpk=kpk_dom
        rs_table2[i]=rs(arh)*hor_i
    else:
        kpk=kpk_fz
        rs_table2[i]=rs(arh)*hor_i

    kpk_table2[i]=kpk/hor_i
    kpk_dom_table2[i]=kpk_dom/hor_i
    kpk_fz_table2[i]=kpk_fz/hor_i
    adom_afz_table[i]=adom/afz/2
    print(alpha_can)
    i=i+1

i_fz=np.where(adom_afz_table<1)[0][0]

# =============================================================================
# Numerical kpk
# =============================================================================
# # #Calculating transfer function
# xi_table2=np.array([0.4,0.6,0.8,1.05,1.6,2.5,4,9])
# # xi_table2=np.array([0.4,])
# kpk_num_table=np.zeros(xi_table2.shape[0])
# kpk_an_table=np.zeros(xi_table2.shape[0])
# kpk_num_lerr=np.zeros(xi_table2.shape[0])
# kpk_num_herr=np.zeros(xi_table2.shape[0])
# n=0
# for xi in xi_table2:
#     m,xi,alpha_can,afz,adom,arh,rhocan,rhor,X,w,c2s,Ht,rhodm,hor_i,aeq,gamma,H1=bck2(m,xi,alpha_can,Trh,0)
#     ddel,ddelsim,ddel_super_sim,ddel_post_rh,ddel_r=pert(rhocan,rhor,rhodm,Ht,c2s,w,gamma)

#     if adom>2*afz:
#         csfz=np.sqrt(c2s(20*afz))*20
#         lambda1=np.real(3*csfz/2/np.sqrt(2)*(lambertw(2*(0.594/3/csfz*adom/afz)**(2/3)))**(3/2))
#         apk=lambda1*afz
#         kpk=apk*Ht(apk)
#         K=10**np.arange(log(kpk/2)/log(10),log(kpk*2)/log(10),0.03)
#         print(K.shape[0])
#         delta_dmf=np.zeros(K.shape[0])
#         delta_dm_primef=np.zeros(K.shape[0])
#         aend=np.zeros(K.shape[0])
#         j=0
#         aosc=np.sqrt(adom/arh*c2s(afz))*afz
#         kosc=aosc*Ht(aosc)*1.5
#         for k in K:
#             if k>1:
#                 ahz=optimize.fsolve(lambda a: a*Ht(a)-k,1/k)[0]
#             else:
#                 ahz=optimize.fsolve(lambda a: a*Ht(a)-k,1)[0]
#             if ahz<arh/100:#For modes entering the horizon before reheating we completely ignore radiation perturbation
#                 astart=min(1,ahz/10)
#                 delta0=[1,-2,k**2/2/astart/Ht(astart),-2/(1+w(astart)),k**2/2/astart/Ht(astart),-2,k**2/2/astart/Ht(astart)]
#                 delta=solve_ivp(lambda a,y: ddel(a,y,k),[astart,100*ahz],delta0,method='BDF',dense_output=False,atol=1e-8,rtol=1e-7)# Complete perturbation equations solving till 100ahz
#                 if k<kosc:#ahz>aosc
#                     deltasim0=[delta.y[0,-1],delta.y[1,-1],delta.y[2,-1],delta.y[3,-1],delta.y[4,-1]]#initial condition  deep in subhorizon
#                     deltasim=solve_ivp(lambda a,y: ddelsim(a,y,k),[delta.t[-1],5*arh],deltasim0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5)#ignore radiation perturbations completely
#                     delta_dmf[j]=deltasim.y[3,-1] #find final DM_density
#                     delta_dm_primef[j]=(deltasim.y[3,-1]-deltasim.y[3,-2])/(deltasim.t[-1]-deltasim.t[-2])#find final derivative of DM density
#                     aend[j]=deltasim.t[-1] #find time when we end the simulation
#                 else:
#                     deltasupersim0=[delta.y[3,-1],delta.y[4,-1]]#initial condition deep in subhorizon for just DM
#                     deltasupersim=solve_ivp(lambda a,y: ddel_super_sim(a,y,k),[delta.t[-1],5*arh],deltasupersim0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5)#ignore radiation, cannibal and metric perturbations completely
#                     delta_dmf[j]=deltasupersim.y[0,-1] #find final DM_density
#                     delta_dm_primef[j]=(deltasupersim.y[0,-1]-deltasupersim.y[0,-2])/(deltasupersim.t[-1]-deltasupersim.t[-2])#find final derivative of DM density
#                     aend[j]=deltasupersim.t[-1] #find time when we end the simulation
#             else: #for modes entering the horizon close to reheating we solve the full equation till reheating occurs and then ignore cannibal
#                 astart=1
#                 a_end=max(100*arh,100*ahz)
#                 delta0=[1,-2,k**2/2,-2/(1+w(astart)),k**2/2,-2,k**2/2]
#                 delta=solve_ivp(lambda a,y: ddel(a,y,k),[astart,5*arh],delta0,method='BDF',dense_output=False,atol=1e-8,rtol=1e-7)#,atol=1e-6,rtol=1e-5
#                 delta_post_rh0=[delta.y[0,-1],delta.y[3,-1],delta.y[4,-1],delta.y[5,-1],delta.y[6,-1]]
#                 delta_post_rh=solve_ivp(lambda a,y: ddel_post_rh(a,y,k),[delta.t[-1],a_end],delta_post_rh0,method='BDF',dense_output=False,atol=1e-7,rtol=1e-6)
#                 delta_dmf[j]=delta_post_rh.y[1,-1]
#                 delta_dm_primef[j]=(delta_post_rh.y[1,-1]-delta_post_rh.y[1,-2])/(delta_post_rh.t[-1]-delta_post_rh.t[-2])
#                 aend[j]=delta_post_rh.t[-1]
#             j=j+1
#             print(j)
#     else:
#         kpk=(3/2*(1+w(2*afz))/c2s(2*afz))**0.5*(2*afz*Ht(2*afz))/1.4
#         K=10**np.arange(log(kpk/2)/log(10),log(kpk*2)/log(10),0.03)
#         print(K.shape[0])
#         delta_dmf=np.zeros(K.shape[0])
#         delta_dm_primef=np.zeros(K.shape[0])
#         aend=np.zeros(K.shape[0])
#         j=0
#         kosc=1*Ht(1)
        
#         for k in K:
#             if k>1:
#                 ahz=optimize.fsolve(lambda a: a*Ht(a)-k,1/k)[0]
#             else:
#                 ahz=optimize.fsolve(lambda a: a*Ht(a)-k,1)[0]
#             if ahz<arh/100:#For modes entering the horizon before reheating we completely ignore radiation perturbation
#                 astart=min(1,ahz/10)
#                 delta_sim0=[1,-2,k**2/2/astart/Ht(astart),-2/(1+1/3),k**2/2/astart/Ht(astart)]
#                 if k<kosc:#will take into account cannibal perturbations
#                     delta_sim=solve_ivp(lambda a,y: ddelsim(a,y,k),[astart,5*arh],delta_sim0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5)#,atol=1e-6,rtol=1e-5
#                     delta_dmf[j]=delta_sim.y[3,-1] #find final DM_density
#                     delta_dm_primef[j]=(delta_sim.y[3,-1]-delta_sim.y[3,-2])/(delta_sim.t[-1]-delta_sim.t[-2])#find final derivative of DM density
#                     aend[j]=delta_sim.t[-1] #find time when we end the simulation
#                 else:# will ignore cannibal perturbations after 100ahz
#                     delta_sim=solve_ivp(lambda a,y: ddelsim(a,y,k),[astart,100*ahz],delta_sim0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5)
#                     deltasupersim0=[delta_sim.y[3,-1],delta_sim.y[4,-1]]#initial condition deep in subhorizon for just DM
#                     deltasupersim=solve_ivp(lambda a,y: ddel_super_sim(a,y,k),[delta_sim.t[-1],5*arh],deltasupersim0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5)#ignore radiation, cannibal and metric perturbations completely
#                     delta_dmf[j]=deltasupersim.y[0,-1] #find final DM_density
#                     delta_dm_primef[j]=(deltasupersim.y[0,-1]-deltasupersim.y[0,-2])/(deltasupersim.t[-1]-deltasupersim.t[-2])#find final derivative of DM density
#                     aend[j]=deltasupersim.t[-1] #find time when we end the simulation
#             else: #for modes entering the horizon close to reheating we solve the full equation till reheating occurs and then ignore cannibal
#                 astart=1
#                 a_end=max(100*arh,100*ahz)
#                 delta0=[1,-2,k**2/2,-2/(1+w(astart)),k**2/2,-2,k**2/2]
#                 delta=solve_ivp(lambda a,y: ddel(a,y,k),[astart,5*arh],delta0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5)#,atol=1e-6,rtol=1e-5
#                 delta_post_rh0=[delta.y[0,-1],delta.y[3,-1],delta.y[4,-1],delta.y[5,-1],delta.y[6,-1]]
#                 delta_post_rh=solve_ivp(lambda a,y: ddel_post_rh(a,y,k),[delta.t[-1],a_end],delta_post_rh0,method='BDF',dense_output=False,atol=1e-6,rtol=1e-5)
#                 delta_dmf[j]=delta_post_rh.y[1,-1]
#                 delta_dm_primef[j]=(delta_post_rh.y[1,-1]-delta_post_rh.y[1,-2])/(delta_post_rh.t[-1]-delta_post_rh.t[-2])
#                 aend[j]=delta_post_rh.t[-1]
#             j=j+1
#             print(j)
#     A=aend*delta_dm_primef
#     B=1/(aend)*np.exp(delta_dmf/A)
#     ahoru=(5*arh)**2*Ht(5*arh)/K #finding horizon entry of DM perturbation in a universe without cannibal
#     relative_delta_DM=A/9.11*(1+(np.log(ahoru/0.594/aend)+delta_dmf/A)/np.log(4*0.594*np.exp(-3)*aeq/ahoru))#finding ratio of DM density wrt t scenaario where there was no cannibal and only radiation domination in the early universe
    
#     i_pk=np.where(abs(relative_delta_DM)==max(abs(relative_delta_DM)))[0][0]
#     i_err=np.where(abs(relative_delta_DM)>0.94*max(abs(relative_delta_DM)))[0]
#     kpk_num_table[n]=K[i_pk]/hor_i
#     kpk_num_lerr[n]=(K[i_pk]-K[i_err[0]])/hor_i
#     kpk_num_herr[n]=(K[i_err[-1]]-K[i_pk])/hor_i
#     kpk_an_table[n]=kpk/hor_i
    
#     # plt.figure(n)
#     # miny=0.1
#     # maxy=5*np.max(abs(relative_delta_DM))
#     # plt.loglog(K,abs(relative_delta_DM))
#     # plt.loglog([kpk,kpk],[miny,maxy],'--k')
#     # plt.text(1.1*kpk,2*miny,r"$k_{pk}$")
#     # plt.loglog([K[i_pk],K[i_pk]],[miny,maxy],'--r')
#     # plt.ylim([miny,maxy])
#     # plt.xlabel(r"$k/k_{hor,i}$")
#     # plt.ylabel(r"$|T(k)|$",rotation=90, labelpad=0)
#     n=n+1

def forward2(x):
    return np.interp(x, alpha_table, adom_afz_table)

def inverse2(x):
    return np.interp(1/x, 1/adom_afz_table, alpha_table)

alpha_dom=inverse2(1)
plt.rcParams.update({'font.size': 20})
fig, ax1 = plt.subplots()
ax1.set_xlabel(r"$\xi_i$")
ax1.set_ylabel(r"$k_{pk}$ (pc${}^{-1}$)")
# ax1.loglog(xi_table,kpk_table2,'-b',label="combined analytical")
ax1.loglog(alpha_table,kpk_dom_table2,'-',color='darkorange',label=r"analytical for $a_{dom}\gg a_{fz}$")
ax1.loglog(alpha_table,kpk_fz_table2,'-g',label=r"analytical for $a_{dom}\ll a_{fz}$")
ax1.loglog(alpha_table,1/rs_table2,'--b',label=r"$r_s(a_{rh})$")
# ax1.scatter(xi_table2,kpk_num_table,label=r"Numerical")
# ax1.errorbar(xi_table2,kpk_num_table,yerr=[kpk_num_lerr, kpk_num_herr],fmt='o',label=r"Numerical")
ax1.loglog([alpha_dom,alpha_dom],[1, 10**2],'--k',alpha=0.3)
ax1.set_ylim(.1, 10**3)
# ax1.text(xi_table[i_fz],4*1.1,r"$a_{dom}=2a_{fz}$")

# from matplotlib.ticker import FixedLocator, FixedFormatter
secax = ax1.secondary_xaxis('top', functions=(forward2, inverse2))
secax.set_xlabel(r'$a_{dom}/2a_{fz}$')
# secax.tick_params(labeltop=False)
# ax1.text(0.125, 1.02, r'$10^1$', fontsize=14,transform=ax1.transAxes)
# ax1.text(0.288, 1.02, r'$10^0$', fontsize=14,transform=ax1.transAxes)
# ax1.text(0.431, 1.02, r'$10^{-1}$', fontsize=14,transform=ax1.transAxes)
# ax1.text(0.56, 1.02, r'$10^{-2}$', fontsize=14,transform=ax1.transAxes)
# secax.set_xlabel('$a_{dom}/2a_{fz}$',labelpad=20)

fig.legend(bbox_to_anchor=(0.98, 0.8))
# ax1.text(xi_table[0],80,r"$m= 300$ GeV, $\alpha_c=1$, $T_{rh}=8$ MeV")

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
plt.gcf().set_size_inches(10, 6,forward=True)
plt.gcf().subplots_adjust(left=0.11, right=0.98, bottom=0.13, top=0.88)