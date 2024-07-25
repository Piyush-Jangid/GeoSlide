import matplotlib.pyplot as plt
import numpy as np

from .utills import getFromDict,getDataFromExcel,generate_samples,interpolate_2d_array

file_path='B.xlsx'
Bru = getDataFromExcel(file_path,'B')

def INFINITE_SLOPE_DET(beta,theeta,H,c,phi,cdash,phdash,l,lw,X,T):
    #Factor Of Safety Determination
    if theeta==beta:
        ru = (X*lw* (np.cos(beta))**2)/(T*l)
    else:
        ru = lw/(l*(1+np.tan(beta)*np.tan(theeta)))

    A=1-ru*(1+(np.tan(beta))**2)
    B=interpolate_2d_array(Bru,1/np.tan(beta))

    FOS1 = A*(np.tan(phdash)/np.tan(beta))+B*(cdash/(l*H))#Effective stress analyses
    FOS2 = (np.tan(phi)/np.tan(beta))+B*(c/(l*H))#Total stress analyses

    print('Factor of safety for infinite slope condition: ',max(FOS1,FOS2))#1.63
    return max(FOS1,FOS2)
    pass

def INFINITE_SLOPE_DET_FOR_PROB(beta,theeta,H,c,phi,cdash,phdash,l,lw,X,T):
    #Factor Of Safety Determination
    if theeta==beta:
        ru = (X*lw* (np.cos(beta))**2)/(T*l)
    else:
        ru = lw/(l*(1+np.tan(beta)*np.tan(theeta)))

    A=1-ru*(1+(np.tan(beta))**2)
    B=interpolate_2d_array(Bru,1/np.tan(beta))

    FOS1 = A*(np.tan(phdash)/np.tan(beta))+B*(cdash/(l*H))#Effective stress analyses
    FOS2 = (np.tan(phi)/np.tan(beta))+B*(c/(l*H))#Total stress analyses
    return max(FOS1,FOS2)
    pass

def INFINITE_SLOPE_PROB(beta,theeta,H,c,phi,cdash,phdash,l,lw,X,T,num_simulations=10000):
    c_samples = generate_samples(*c, num_simulations)
    phi_samples = generate_samples(*phi, num_simulations)
    cdash_samples = generate_samples(*cdash, num_simulations)
    phdash_samples = generate_samples(*phdash, num_simulations)
    l_samples = generate_samples(*l,num_simulations)
    
    mean_beta,std_beta=beta,0
    mean_theeta,std_theeta=theeta,0
    mean_H, std_H = H, 0
    mean_X, std_X = X, 0
    mean_T, std_T = T, 0
    mean_lw,std_lw=lw,0
    
    beta_samples = np.random.normal(mean_beta, std_beta, num_simulations)
    theeta_samples = np.random.normal(mean_theeta, std_theeta, num_simulations)
    H_samples = np.random.normal(mean_H, std_H, num_simulations)
    lw_samples = np.random.normal(mean_lw, std_lw, num_simulations)
    X_samples = np.random.normal(mean_X, std_X, num_simulations)
    T_samples = np.random.normal(mean_T, std_T, num_simulations)
    
    Fos_values = []
    
    for i in range(num_simulations):                                                      #beta,theeta,H,c,phi,cdash,phdash,l,lw,X,T,num_simulations
        fos_values=INFINITE_SLOPE_DET_FOR_PROB(beta_samples[i], theeta_samples[i], H_samples[i],c_samples[i], phi_samples[i],cdash_samples[i],phdash_samples[i], l_samples[i] ,lw_samples[i],X_samples[i], T_samples[i])
        Fos_values.append(fos_values)
    Fos_values.sort()
    Fos_values=np.array(Fos_values)
    # Plot the distribution of FOS
    plt.figure(figsize=(10, 6))
    plt.hist(Fos_values, bins=50, edgecolor='k', alpha=0.7)
    plt.title('Distribution of Factor of Safety (FoS) Infinite slope')
    plt.xlabel('Factor of Safety')
    plt.ylabel('Frequency')
    plt.grid(True)
    # plt.show()
    
    #Determining Probability of failure vs FOS graph
    hist, bin_edges = np.histogram(Fos_values, bins=50, density=True)
    # Compute the CDF from the histogram
    cdf = np.cumsum(hist * np.diff(bin_edges))
    # Compute the probability of failure
    prob_of_failure = 1 - cdf
    fos_range = bin_edges[:-1]
    # Plot the probability of failure vs FOS
    plt.figure(figsize=(10, 6))
    plt.plot(fos_range, prob_of_failure, label='Probability of Failure', color='red')
    plt.xlabel('Factor of Safety')
    plt.ylabel('Probability of Failure')
    plt.title('Probability of Failure vs Factor of Safety for c-phi soil')
    plt.grid(True)
    plt.legend()
    plt.show()
    pass