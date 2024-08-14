# my_package/core.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .phiz_increase import PURELY_COHESIVE_SOIL_WITH_INCREASING_SHEAR_STRENGTH_DET,PURELY_COHESIVE_SOIL_WITH_INCREASING_SHEAR_STRENGTH_PROB
from .infinite import INFINITE_SLOPE_DET,INFINITE_SLOPE_PROB
from .chi_phi import CHI_PHI_SOIL_DET,CHI_PHI_SOIL_PROB
from .purely_cohesive import PURELY_COHESIVE_SOIL_DET,PURELY_COHESIVE_SOIL_PROB


# def PURELY_COHESIVE_SOIL(beta,H,Hw,Hwdash,D,c,lw,l,Ht,q,str,num_simulations=10000):
#     # Check if str is provided
#     if str == 'deterministic':
#         return PURELY_COHESIVE_SOIL_DET(beta,H,Hw,Hwdash,D,c[0],lw,l[0],Ht,q)
#     elif str == 'probabilistic':
#         return PURELY_COHESIVE_SOIL_PROB(beta,H,Hw,Hwdash,D,c,lw,l,Ht,q,num_simulations)
#     else:
#         raise ValueError("Invalid string parameter: must be 'deterministic' or 'probabilistic'")
#     pass

def PURELY_COHESIVE_SOIL(layers,params,str,num_simulations=1000):
    # Check if str is provided
    if str == 'deterministic':
        return PURELY_COHESIVE_SOIL_DET(beta=params['beta'],
                                        H=params['H'],
                                        Hw=params['Hw'],
                                        Hwdash=params['Hwdash'],
                                        D=params['D'],
                                        lw=params['lw'],
                                        Ht=params['Ht'],
                                        q=params['q'],
                                        layers=layers)
    elif str == 'probabilistic':
        return PURELY_COHESIVE_SOIL_PROB(beta=params['beta'],
                                         H=params['H'],
                                         Hw=params['Hw'],
                                         Hwdash=params['Hwdash'],
                                         D=params['D'],
                                         lw=params['lw'],
                                         Ht=params['Ht'],
                                         q=params['q'],
                                         layers=layers,
                                         num_simulations=num_simulations)
    else:
        raise ValueError("Invalid string parameter: must be 'deterministic' or 'probabilistic'")
    pass
    
def CHI_PHI_SOIL(layers,params,str,num_simulations=1000):
    # Check if str is provided
    if str == 'deterministic':
        return CHI_PHI_SOIL_DET(beta=params['beta'],
                                H=params['H'],
                                Hw=params['Hw'],
                                Hc=params['Hc'],
                                Hwdash=params['Hwdash'],
                                D=params['D'],
                                lw=params['lw'],
                                q=params['q'],
                                Ht=params['Ht'],
                                layers=layers)
    elif str == 'probabilistic':
        return CHI_PHI_SOIL_PROB(beta=params['beta'],
                                 H=params['H'],
                                 Hw=params['Hw'],
                                 Hc=params['Hc'],
                                 Hwdash=params['Hwdash'],
                                 D=params['D'],
                                 lw=params['lw'],
                                 q=params['q'],
                                 Ht=params['Ht'],
                                 layers=layers,
                                 num_simulations=num_simulations)
    else:
        raise ValueError("Invalid string parameter: must be 'deterministic' or 'probabilistic'")
    pass

def INFINITE_SLOPE(beta,theeta,H,c,phi,cdash,phdash,l,lw,X,T,str,num_simulations=10000):
    beta=np.radians(beta)
    theeta=np.radians(theeta)
    ph=phi[0]
    phd=phdash[0]
    phi[0]=np.radians(ph)
    phdash[0]=np.radians(phd)

    # Check if str is provided
    if str == 'deterministic':
        return INFINITE_SLOPE_DET(beta,theeta,H,c[0],phi[0],cdash[0],phdash[0],l[0],lw,X,T)
    elif str == 'probabilistic':
        return INFINITE_SLOPE_PROB(beta,theeta,H,c,phi,cdash,phdash,l,lw,X,T,num_simulations)
    else:
        raise ValueError("Invalid string parameter: must be 'deterministic' or 'probabilistic'")
    pass

def PURELY_COHESIVE_SOIL_WITH_INCREASING_SHEAR_STRENGTH(beta,H,H0,cb,l,lb,str,num_simulations=10000):
    # Check if str is provided
    if str == 'deterministic':
        return PURELY_COHESIVE_SOIL_WITH_INCREASING_SHEAR_STRENGTH_DET(beta,H,H0,cb[0],l[0],lb[0])
    elif str == 'probabilistic':
        return PURELY_COHESIVE_SOIL_WITH_INCREASING_SHEAR_STRENGTH_PROB(beta,H,H0,cb,l,lb,num_simulations)
    else:
        raise ValueError("Invalid string parameter: must be 'deterministic' or 'probabilistic'")
    pass

