from GeoSlide import CHI_PHI_SOIL,INFINITE_SLOPE,PURELY_COHESIVE_SOIL,PURELY_COHESIVE_SOIL_WITH_INCREASING_SHEAR_STRENGTH

import matplotlib.pyplot as plt
 
plt.style.use('Rplot')

PURELY_COHESIVE_SOIL(beta=42,
                     H=13,
                     Hw=0,
                     Hwdash=0,
                     D=7,
                     c=[410, 0.2, 'normal'],
                     lw=62.5,
                     l=[120, 0.2, 'normal'],
                     Ht=0,q=220,
                     str='probabilistic',
                     num_simulations=10000)

# CHI_PHI_SOIL(beta=30,
#              H=23,
#              Hw=4,
#              Hc=0,
#              Hwdash=0,
#              D=0,
#              c=[280, 0.2, 'normal'],
#              phi=[17,0.15,'normal'],
#              l=[120, 0.2, 'normal'],
#              lw=62.5,
#              q=40,
#              Ht=0,
#              str='probabilistic',
#              num_simulations=100)

# INFINITE_SLOPE(beta=20,
#                theeta=0,
#                H=12,
#                c=[0, 0.2, 'normal'],
#                phi=[0, 0.15, 'normal'],
#                cdash=[300, 0.2, 'normal'],
#                phdash=[30, 0.15, 'normal'],
#                l=[120, 0.2, 'normal'],
#                lw=62.4,
#                X=8,
#                T=11.3,
#                str='probabilistic',
#                num_simulations=10000)
                                                                                                                                                                                # deterministic
# PURELY_COHESIVE_SOIL_WITH_INCREASING_SHEAR_STRENGTH(beta=45,
#                                                     H=100,
#                                                     H0=15,
#                                                     cb=[1150, 0.2, 'normal'],
#                                                     l = [100, 0.2, 'normal'],
#                                                     lb = [37.6, 0.2, 'normal'],
#                                                     str='probabilistic',
#                                                     num_simulations=10000)