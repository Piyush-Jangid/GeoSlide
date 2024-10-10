from .core import PURELY_COHESIVE_SOIL,CHI_PHI_SOIL,INFINITE_SLOPE,PURELY_COHESIVE_SOIL_WITH_INCREASING_SHEAR_STRENGTH

def solve(problem,params,str,layers=[],num_simulations=1000):
    if problem=='phi=0':
        return PURELY_COHESIVE_SOIL(layers,
                     params,
                     str=str,
                     num_simulations=num_simulations)
    elif problem=='c-phi':
        return CHI_PHI_SOIL(layers,
                            params,
                            str=str,
                            num_simulations=num_simulations)
    elif problem=='infinite-slope':
        return INFINITE_SLOPE(beta=params['beta'],
               theeta=params['theeta'],
               H=params['H'],
               c=params['c'],
               phi=params['phi'],
               cdash=params['cdash'],
               phdash=params['phdash'],
               l=params['l'],
               lw=params['lw'],
               X=params['X'],
               T=params['T'],
               str=str,
               num_simulations=num_simulations)
    elif problem=='phi=0_increasing_strength':
        return PURELY_COHESIVE_SOIL_WITH_INCREASING_SHEAR_STRENGTH(beta=params['beta'],
                                                    H=params['H'],
                                                    H0=params['H0'],
                                                    cb=params['cb'],
                                                    l = params['l'],
                                                    lb = params['lb'],
                                                    str=str,
                                                    num_simulations=num_simulations)
    else:
        raise ValueError("Invalid string parameter: must be 'phi=0' or 'c-phi' or 'infinite-slope' or 'phi=0_increasing_strength'")
    pass
    