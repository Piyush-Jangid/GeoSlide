# from SlopeStability import CHI_PHI_SOIL,INFINITE_SLOPE,PURELY_COHESIVE_SOIL,PURELY_COHESIVE_SOIL_WITH_INCREASING_SHEAR_STRENGTH
import matplotlib.pyplot as plt
from GeoSlide import solve
 
plt.style.use('Rplot')
plt.rcParams['mathtext.cal'] = 'serif'

# #(1) Phi=0 Soil
layers=[
    {'cohesion': [600,0.2,'normal'], 'friction_angle': [0,0.2,'normal'], 'unit_weight': [120,0.2,'normal'], 'thickness': 12},
    {'cohesion': [400,0.2,'normal'], 'friction_angle': [0,0.2,'normal'], 'unit_weight': [100,0.2,'normal'], 'thickness': 12},
    {'cohesion': [500,0.2,'normal'], 'friction_angle': [0,0.2,'normal'], 'unit_weight': [105,0.2,'normal'], 'thickness': 12},
]
solve('phi=0',params={'beta':50,
                            'H':24,
                            'Hw':8,
                            'Hwdash':8,
                            'D':12,
                            'lw':62.4,
                            'Ht':8,
                            'q':22,
                            },
                            str='probabilistic',
                            layers=layers,
                            num_simulations=1000)

# # (2) c-phi Soil
# layers=[
#     {'cohesion': [800,0.2,'normal'], 'friction_angle': [8,0.2,'normal'], 'unit_weight': [115,0.2,'normal'], 'thickness': 20},
#     {'cohesion': [600,0.2,'normal'], 'friction_angle': [6,0.2,'normal'], 'unit_weight': [110,0.2,'normal'], 'thickness': 20},
#     {'cohesion': [800,0.2,'normal'], 'friction_angle': [0,0.2,'normal'], 'unit_weight': [120,0.2,'normal'], 'thickness': 20},
# ]

# solve('c-phi',params={'beta':33.69,
#                             'H':40,
#                             'Hw':10,
#                             'Hc':0,
#                             'Hwdash':25,
#                             'D':20,
#                             'lw':62.4,
#                             'Ht':5,
#                             'q':220,
#                             },
#                             str='probabilistic',
#                             layers=layers,
#                             num_simulations=1000)

# #(3)Infinite slope
# solve('infinite-slope',params={
#                             'beta':20,
#                             'theeta':0,
#                             'H':12,
#                             'c':[0, 0.2, 'normal'],
#                             'phi':[0, 0.15, 'normal'],
#                             'cdash':[300, 0.2, 'normal'],
#                             'phdash':[30, 0.15, 'normal'],
#                             'l':[120, 0.2, 'normal'],
#                             'lw':62.4,
#                             'X':8,
#                             'T':11.3,
#                             },
#                             str='probabilistic',
#                             num_simulations=10000)

# #(4) phi=0 with Increasing shear strength
# solve('phi=0_increasing_strength',params={
#                             'beta':45,
#                             'H':100,
#                             'H0':15,                            
#                             'cb':[1150, 0.2, 'normal'],
#                             'l':[100, 0.2, 'normal'],
#                             'lb':[37.6, 0.2, 'normal'],
#                             },
#                             str='probabilistic',
#                             num_simulations=10000)
