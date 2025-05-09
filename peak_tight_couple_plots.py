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
from perturbation import pert, pert_tk
os.chdir('../../../draft/cannibal-long_paper_figs')
def table(f,t):
    tablef=np.zeros(t.shape[0])
    for i in np.arange(t.shape[0]):
        tablef[i]=f(t[i])
    return tablef

# plotting
plt.rcParams.update({'font.size': 24})

miny=0.01
maxy=5*np.max(abs(relative_delta_DM))
plt.loglog(K_001,abs(relative_delta_DM_001),label=r'decoupled DM',zorder=1)
plt.loglog(K_000,abs(relative_delta_DM_000),label=r'$a_{kd,DM}=a_{rh}/100$',zorder=3)
plt.loglog(K,abs(relative_delta_DM),label=r'$a_{kd,DM}=2a_{rh}$',zorder=2)
plt.loglog([kpk,kpk],[miny,maxy],'--k')
plt.text(1.1*kpk,maxy/2.5,r"$k_{pk}$")
plt.loglog([krh,krh],[miny,maxy],'--b')
plt.text(1.1*krh,maxy/2.5,r"$k_{rh}$")
# plt.loglog([kdom,kdom],[miny,maxy],'--',color='purple')
# plt.text(1.1*kdom,maxy/2,r"$k_{dom}$")
plt.ylim([miny,maxy])
plt.xlabel(r"$k/k_{hor,i}$")
plt.ylabel(r"$|T(k)|$",rotation=90, labelpad=0)
plt.legend(loc='lower center')

plt.gcf().set_size_inches(10, 6,forward=True)
plt.gcf().subplots_adjust(left=0.13, right=0.98, bottom=0.15, top=0.97)
#
#mycwd = os.getcwd()
#os.chdir("..")
#plt.savefig("SM_dom_horz_pk.pdf")
#os.chdir(mycwd)