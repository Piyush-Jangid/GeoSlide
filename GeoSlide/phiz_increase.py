import matplotlib.pyplot as plt
import numpy as np

from .utills import getFromDict,getDataFromExcel,generate_samples

file_path='Phi_0_Increasing_shear_strength.xlsx'
m0 = getDataFromExcel(file_path,'m=0')
m25= getDataFromExcel(file_path,'m=0.25')
m50= getDataFromExcel(file_path,'m=0.5')
m75= getDataFromExcel(file_path,'m=0.75')
m100=getDataFromExcel(file_path,'m=1')
m125=getDataFromExcel(file_path,'m=1.25')
m150=getDataFromExcel(file_path,'m=1.5')
m175=getDataFromExcel(file_path,'m=1.75')
m200=getDataFromExcel(file_path,'m=2')

m_dict={
    0:m0,
    0.25:m25,
    0.5:m50,
    0.75:m75,
    1:m100,
    1.25:m125,
    1.5:m150,
    1.75:m175,
    2:m200
}

def PURELY_COHESIVE_SOIL_WITH_INCREASING_SHEAR_STRENGTH_DET(beta,H,H0,Cb,l,lb):
    M=H0/H
    N = getFromDict(m_dict,M,beta)
    # print(N)
    FOS = N*(Cb/(lb*(H+H0)))# For submerge slope
    print('Factor of safety for Phi=0,Increasing Shear Strength: ',FOS)#1.36
    return FOS
    pass

def PURELY_COHESIVE_SOIL_WITH_INCREASING_SHEAR_STRENGTH_DET_FOR_PROB(beta,H,H0,Cb,l,lb):
    M=H0/H
    N = getFromDict(m_dict,M,beta)
    # print(N)
    FOS = N*(Cb/(lb*(H+H0)))# For submerge slope
    return FOS
    pass

def PURELY_COHESIVE_SOIL_WITH_INCREASING_SHEAR_STRENGTH_PROB(beta,H,H0,cb,l,lb,num_simulations=10000):
    cb_samples = generate_samples(*cb, num_simulations)
    l_samples = generate_samples(*l,num_simulations)
    lb_samples = generate_samples(*lb,num_simulations)
    
    mean_beta,std_beta=beta,0
    mean_H, std_H = H, 0
    mean_H0, std_H0 = H0, 0
    
    beta_samples = np.random.normal(mean_beta, std_beta, num_simulations)
    H_samples = np.random.normal(mean_H, std_H, num_simulations)
    H0_samples = np.random.normal(mean_H0, std_H0, num_simulations)
    
    Fos_values = []
    
    for i in range(num_simulations):                                                      #beta,theeta,H,c,phi,cdash,phdash,l,lw,X,T,num_simulations
        fos_values=PURELY_COHESIVE_SOIL_WITH_INCREASING_SHEAR_STRENGTH_DET_FOR_PROB(beta_samples[i],H_samples[i],H0_samples[i], cb_samples[i],l_samples[i] ,lb_samples[i])
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
