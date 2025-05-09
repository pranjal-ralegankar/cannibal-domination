# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 10:57:30 2019

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
from scipy.special import lambertw
from matplotlib import ticker
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

dof=pd.read_excel('../../pranjal work/git-python/cannibal-base-code/early universe degrees of freedom2.xlsx', sheetname='Sheet1')
ge=np.array(dof.Ge.values)
T_mev=np.array(dof.K_bT_MeV.values)

Mpl=2.435*10**21 #in MeV

lmbda=0.02
alpha_can=3/pi/2**(4/3)*lmbda
xi=1

Trh_table=10**np.arange(log(3)/log(10),log(10**6)/log(10),0.1) #in Mev
m_table=10**np.arange(log(10**3)/log(10),log(10**11)/log(10),0.1) # in MeV
arh_table=np.zeros([Trh_table.shape[0],m_table.shape[0]])
afz_table=np.zeros([Trh_table.shape[0],m_table.shape[0]])
adom_table=np.zeros([Trh_table.shape[0],m_table.shape[0]])
y_table=np.zeros([Trh_table.shape[0],m_table.shape[0]])
Mpk_table=np.zeros([Trh_table.shape[0],m_table.shape[0]])
T_kpk_table=np.zeros([Trh_table.shape[0],m_table.shape[0]])

i=0
print(Trh_table.shape[0])
for Trh in Trh_table:
    print(i)
    j=0
    for m in m_table:
        print("i=",i,"; j=",j)
        #limits of m
        m_afz=8*alpha_can**(3/7)*(Trh/10)**(6/7)*10**3 # arh>afz condition when afzz>adom
        m_adom=6/xi**3*(Trh/10)*10**3# arh>adom condtion when afz<adom
        m_alphaDM=1.2*10**8*(Trh/10)**-1/sqrt(1+np.interp(10*m/xi,np.flip(T_mev),np.flip(ge))/xi**4)*lmbda*10**3 #alpha_DM<1 condition
        m_alphac1=8*10**12*xi**2*alpha_can**3 #adom>afz>100a_i
        m_alphac2=2*10**13*alpha_can**3 #afz>100a_i and afz>adom
        m_mDM=10**8/4*(Trh/10)**-1/sqrt(1+np.interp(10*m/xi,np.flip(T_mev),np.flip(ge))/xi**4)*lmbda**2*10**3 #mDM/m>10
        if m>m_afz/10 and m>m_adom/10 and m>m_mDM/10 and m<m_alphaDM*10 and m<m_alphac1*10 and m<m_alphac2*10:
            #solving background equations
            m,xi,alpha_can,afz,adom,arh,rhocan,rhor,X,w,c2s,Ht,rhodm,hor_i,aeq,gamma,H1=bck2(m,xi,alpha_can,Trh,0)
            
            #yDM
            Y_inf=3.83*10**(-40)*aeq**3/10*(1000/m)**4 #assumed m_dm/m=10 to get a guess of x_f
            xf_guess=optimize.root(lambda x: (x/2/pi)**1.5*exp(-x)-Y_inf,10).x[0]
            sigmavx_DM=H1/100/m**3/10*xf_guess**2/Y_inf#10**(42)*xf_guess**2*H1/m*(3402*aeq)**(-3)*(m/1000)**2 # DM annihilation rate in MeV^-2
            ydm=((sigmavx_DM*128*pi*m**2/lmbda+5*lmbda/6)/3)**0.5 # value of y assuming y^2>lambda^2
            #ydmguess=(1.9*10**-6/lmbda*(Trh/10)*(m/1000)*sqrt(1+np.interp(10*m/xi,np.flip(T_mev),np.flip(ge))/xi**4))**0.5
            #sigmavx_dmguess=10**-8*(Trh/10)*(m/1000)**-1*sqrt(1+np.interp(10*m/xi,np.flip(T_mev),np.flip(ge))/xi**4)*10**-6

            # =============================================================================
            #Solving perturbation equations
            if arh>max(adom,afz):
                if adom>afz:
                    csfz=np.sqrt(c2s(20*afz))*20
                    lambda1=np.real(3*csfz/2/np.sqrt(2)*(lambertw(2*(0.594/3/csfz*adom/afz)**(2/3)))**(3/2))
                    apk=lambda1*afz
                    kpk=apk*Ht(apk)
                else:
                    kpk=(3/2*(1+w(2*afz))/c2s(2*afz))**0.5*(2*afz*Ht(2*afz))/1.2
                
#                if adom>afz:
#                    ahoru_pk=(5*arh)**2*Ht(5*arh)/(apk*Ht(apk))
#                    T_kpk=3/2*np.log(4*0.594*np.exp(-3)*adom/apk)*arh/adom*1.533*(1+np.log(ahoru_pk*1.58/0.594/arh)/np.log(4*0.594*np.exp(-3)*aeq/ahoru_pk))
#                else:
#                    ahoru_pk=(5*arh)**2*Ht(5*arh)/kpk
#                    T_kpk=2.86*arh/afz/9.11*(1+np.log(ahoru_pk*1.55/0.594/arh)/np.log(4*0.594*np.exp(-3)*aeq/ahoru_pk))
                
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
            y_table[i,j]=ydm
#            yguess_table[i,j]=ydmguess
            if arh>max(adom,afz):
                hor_i=aeq*3402*1/H1*(1/1.564*10**-29)# horizon size at a_i in parsecs. aeq/a0=1/zeq=1/3402. 1Mev=2.47*10^27 pc^-1.
                Mpk_table[i,j]=1.62*10**-7*(kpk**-1*hor_i)**3# peak micro halo mass
                T_kpk_table[i,j]=T_kpk
        if m>m_alphaDM*10:
            y_table[i,j]=100
        
        j=j+1
    i=i+1

# =============================================================================
# Plotting m,Trh parameter space
plt.rcParams.update({'font.size': 20})
cs=plt.contourf(m_table/10**3,Trh_table,abs(T_kpk_table),[10,100,10**3,10**4,10**6],locator=ticker.LogLocator(), cmap='YlOrRd', vmin=10**-2, vmax=10**4)
plt.colorbar(cs)
plt.xscale('log')
plt.yscale('log')
plt.ylim([1,10**3])
plt.xlim([10,2*10**4])
cs2=plt.contour(m_table/10**3,Trh_table,log(Mpk_table)/log(10),4,colors='white',linestyles='dashed')
fmt = r'$10^{%d}\ M_{\odot}$'
#fmt.create_dummy_axis()
cl=plt.clabel(cs2, cs2.levels, fmt=fmt, fontsize='smaller', use_clabeltext=True, manual=True)
plt.xlabel(r"$m$ (GeV)")
plt.ylabel(r"$T_{rh}$ (MeV)",rotation=90, labelpad=10)
plt.contourf(m_table/10**3,Trh_table,afz_table,[0,100], colors="white")#painting white over param space where asfz<100a_i
#plt.text(m_table[0]/10**3*2,10**4,r"$a_{sfz}>100a_i$")
plt.contourf(m_table/10**3,Trh_table,y_table,[4*pi,200], colors="white")#painting white over param space where y>4pi.
plt.contourf(m_table/10**3,Trh_table,y_table,[0,10*sqrt(lmbda/3)], colors="white")#painting white over param space where y<10root(lambda/3)
#plt.contourf(m_table/10**3,Trh_table,np.nan_to_num(arh_table/afz_table),[0,5], colors="white")#painting white over param space where arh<5asfz
plt.contourf(m_table/10**3,Trh_table,np.nan_to_num(arh_table/adom_table),[0,1], colors="white")#painting white over param space where arh<5adom

if xi==1:
    plt.text(1.2*10**4,30,r"$a_{sfz}>a_{can}$",rotation=90)
    plt.text(3*10**3,200,r"$y<4\pi$",rotation=-40)
    plt.text(180,140,r"$a_{rh}>a_{dom}$",{'ha': 'center', 'va': 'center'},rotation=40)
    plt.text(50,8,r"$m_{DM}>10m$")
    plt.text(10,7*10**2,r"$\lambda=0.02$, $\xi_i=1$")
if xi==10:
    plt.text(10**6*2,30,r"$a_{sfz}>100a_i$")
    plt.text(3*10**5,2000,r"$\alpha_{DM}<1$")
    plt.text(20,300,r"$a_{rh}>5$max$(a_{dom},a_{sfz})$",{'ha': 'center', 'va': 'center'},rotation=55)
    plt.text(m_table[0]/10**3*2,10**5,r"$\alpha_c=0.02$, $\xi_i=10$, $m_{DM}/m=10$")
plt.gcf().set_size_inches(10, 6,forward=True)
plt.gcf().subplots_adjust(left=0.12, right=1.05, bottom=0.13, top=0.97)
plt.minorticks_on()
