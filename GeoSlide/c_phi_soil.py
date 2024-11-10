import matplotlib.pyplot as plt
import numpy as np

from .utills import getFromDict,getDataFromExcel,generate_samples,DrawSolution
from .adjustment_factor import getSurchargeFactor,getSubmergenceAndSeepageFactor,getSteadySeepageFactor,getTensionCrackFactor
from .nonhomogenous import HOMO

file_path='Stability Number for c-phi soil.xlsx'
lcf0 = getDataFromExcel(file_path,'lcf=0')
lcf05 = getDataFromExcel(file_path,'lcf=0.5')
lcf1 = getDataFromExcel(file_path,'lcf=1')
lcf105 = getDataFromExcel(file_path,'lcf=1.5')
lcf2 = getDataFromExcel(file_path,'lcf=2')
lcf3 = getDataFromExcel(file_path,'lcf=3')
lcf4 = getDataFromExcel(file_path,'lcf=4')
lcf5 = getDataFromExcel(file_path,'lcf=5')
lcf6 = getDataFromExcel(file_path,'lcf=6')
lcf7 = getDataFromExcel(file_path,'lcf=7')
lcf8 = getDataFromExcel(file_path,'lcf=8')
lcf10 = getDataFromExcel(file_path,'lcf=10')
lcf15 = getDataFromExcel(file_path,'lcf=15')
lcf20 = getDataFromExcel(file_path,'lcf=20')
lcf30 = getDataFromExcel(file_path,'lcf=30')
lcf50 = getDataFromExcel(file_path,'lcf=50')
lcf100 = getDataFromExcel(file_path,'lcf1=100')

lcf_dict={
    0:lcf0,
    0.5:lcf05,
    1:lcf1,
    1.5:lcf105,
    2:lcf2,
    3:lcf3,
    4:lcf4,
    5:lcf5,
    6:lcf6,
    7:lcf7,
    8:lcf8,
    10:lcf10,
    15:lcf15,
    20:lcf20,
    30:lcf30,
    50:lcf50,
    100:lcf100
}

xlcf0=getDataFromExcel(file_path,'x_lcf=0')
xlcf2=getDataFromExcel(file_path,'x_lcf=2')
xlcf5=getDataFromExcel(file_path,'x_lcf=5')
xlcf10=getDataFromExcel(file_path,'x_lcf=10')
xlcf20=getDataFromExcel(file_path,'x_lcf=20')
xlcf100=getDataFromExcel(file_path,'x_lcf=100')

xlcf_dict={
    0:xlcf0,
    2:xlcf2,
    5:xlcf5,
    10:xlcf10,
    20:xlcf20,
    100:xlcf100,
}

ylcf0=getDataFromExcel(file_path,'y_lcf=0')
ylcf2=getDataFromExcel(file_path,'y_lcf=2')
ylcf5=getDataFromExcel(file_path,'y_lcf=5')
ylcf10=getDataFromExcel(file_path,'y_lcf=10')
ylcf20=getDataFromExcel(file_path,'y_lcf=20')
ylcf100=getDataFromExcel(file_path,'y_lcf=100')

ylcf_dict={
    0:ylcf0,
    2:ylcf2,
    5:ylcf5,
    10:ylcf10,
    20:ylcf20,
    100:ylcf100,
}

def CHI_PHI_SOIL_DET(beta,H,Hw,Hc,Hwdash,D,lw,q,Ht,layers):
    c,phi,l=HOMO(beta,H,Hw,Hc,Hwdash,lw,q,Ht,layers)
    # D=0
    c=c[0]
    phi=phi[0]
    l=l[0]
    uq = getSurchargeFactor(q,l,H,beta,D,1)# Surcharge adjustment factor
    uw,uwdash = getSubmergenceAndSeepageFactor(Hw,Hwdash,H,beta,D,1) # Submergence and seepage adjustment factor
    ut = getTensionCrackFactor(Ht,H,beta,D,1)  # Tension Crack adjustment factor

    Pd = ((l*H+q)-(lw*Hw))/(uq*uw*ut)
    if Hc!=0:
        Hwdash,uwdash=getSteadySeepageFactor(Hc,H,beta,0,1)

    Pe = ((l*H+q)-(lw*Hwdash))/(uq*uwdash)
    lcf=Pe*(np.tan(np.radians(phi))/c)

    Ncf=getFromDict(lcf_dict,lcf,1/np.tan(np.radians(beta)))
    # print(Ncf)
    FOS = Ncf*(c/Pd)
    print('Factor of safety For c-phi: ',FOS)#1
    x0=H*getFromDict(xlcf_dict,lcf,1/np.tan(np.radians(beta)))
    y0=H*getFromDict(ylcf_dict,lcf,1/np.tan(np.radians(beta)))
    print('x0 = ',x0)
    print('y0 = ',y0)
#     FailureCircle(x0,y0,0,H,beta,1)
    DrawSolution(x0_list=[x0],y0_list=[y0],D_list=[D],H=H,beta=beta,T_list=[1],q=q,Hw=Hw,Hwdash=Hwdash,fos_values=[FOS])
    R= np.sqrt(x0*x0+y0*y0)
    print('Radius of slip circle = ',R)
    print('Toe Circle')
    return [FOS,x0,y0,R,D,1]
    pass

def CHI_PHI_SOIL_DET_EXT(beta,H,Hw,Hc,Hwdash,D,lw,q,Ht,layers):
    c,phi,l=HOMO(beta,H,Hw,Hc,Hwdash,lw,q,Ht,layers)
    # D=0
    c=c[0]
    phi=phi[0]
    l=l[0]
    uq = getSurchargeFactor(q,l,H,beta,D,1)# Surcharge adjustment factor
    uw,uwdash = getSubmergenceAndSeepageFactor(Hw,Hwdash,H,beta,D,1) # Submergence and seepage adjustment factor
    ut = getTensionCrackFactor(Ht,H,beta,D,1)  # Tension Crack adjustment factor

    Pd = ((l*H+q)-(lw*Hw))/(uq*uw*ut)
    if Hc!=0:
        Hwdash,uwdash=getSteadySeepageFactor(Hc,H,beta,0,1)

    Pe = ((l*H+q)-(lw*Hwdash))/(uq*uwdash)
    lcf=Pe*(np.tan(np.radians(phi))/c)

    Ncf=getFromDict(lcf_dict,lcf,1/np.tan(np.radians(beta)))
    FOS = Ncf*(c/Pd)
    print('Deterministic solution for zero standard deviation->')
    print('Factor of safety For c-phi: ',FOS)#1
    x0=H*getFromDict(xlcf_dict,lcf,1/np.tan(np.radians(beta)))
    y0=H*getFromDict(ylcf_dict,lcf,1/np.tan(np.radians(beta)))
    print('x0 = ',x0)
    print('y0 = ',y0)
#     FailureCircle(x0,y0,0,H,beta,1)
    # DrawSolution(x0_list=[x0],y0_list=[y0],D_list=[0],H=H,beta=beta,T_list=[1],q=q,Hw=Hw,Hwdash=Hwdash,fos_values=[FOS])
    R= np.sqrt(x0*x0+y0*y0)
    print('Radius of slip circle = ',R)
    print('Toe Circle')
    return [FOS,x0,y0,R,D,1]
    pass


def CHI_PHI_SOIL_MID(beta,H,Hw,Hc,Hwdash,D,c,phi,l,lw,q,Ht):
    # D=0  
    uq = getSurchargeFactor(q,l,H,beta,D,1)# Surcharge adjustment factor
    uw,uwdash = getSubmergenceAndSeepageFactor(Hw,Hwdash,H,beta,D,1) # Submergence and seepage adjustment factor
    ut = getTensionCrackFactor(Ht,H,beta,D,1)  # Tension Crack adjustment factor

    Pd = ((l*H+q)-(lw*Hw))/(uq*uw*ut)
    if Hc!=0:
        Hwdash,uwdash=getSteadySeepageFactor(Hc,H,beta,0,1)

    Pe = ((l*H+q)-(lw*Hwdash))/(uq*uwdash)
    lcf=Pe*(np.tan(np.radians(phi))/c)

    Ncf=getFromDict(lcf_dict,lcf,1/np.tan(np.radians(beta)))
    FOS = Ncf*(c/Pd)
    
    x0=H*getFromDict(xlcf_dict,lcf,1/np.tan(np.radians(beta)))
    y0=H*getFromDict(ylcf_dict,lcf,1/np.tan(np.radians(beta)))
    R= np.sqrt(x0*x0+y0*y0)
    return [FOS,x0,y0,R,D,1]
    pass

def CHI_PHI_SOIL_DET_FOR_PROB(beta,H,Hw,Hc,Hwdash,D,lw,q,Ht,layers):
    # D=0
    c,phi,l=HOMO(beta,H,Hw,Hc,Hwdash,lw,q,Ht,layers)
    # D=0
    c=c[0]
    phi=phi[0]
    l=l[0]    
    uq = getSurchargeFactor(q,l,H,beta,D,1)# Surcharge adjustment factor
    uw,uwdash = getSubmergenceAndSeepageFactor(Hw,Hwdash,H,beta,D,1) # Submergence and seepage adjustment factor
    ut = getTensionCrackFactor(Ht,H,beta,D,1)  # Tension Crack adjustment factor

    Pd = ((l*H+q)-(lw*Hw))/(uq*uw*ut)
    if Hc!=0:
        Hwdash,uwdash=getSteadySeepageFactor(Hc,H,beta,0,1)

    Pe = ((l*H+q)-(lw*Hwdash))/(uq*uwdash)
    lcf=Pe*(np.tan(np.radians(phi))/c)

    Ncf=getFromDict(lcf_dict,lcf,1/np.tan(np.radians(beta)))
    FOS = Ncf*(c/Pd)
    
    x0=H*getFromDict(xlcf_dict,lcf,1/np.tan(np.radians(beta)))
    y0=H*getFromDict(ylcf_dict,lcf,1/np.tan(np.radians(beta)))
    R= np.sqrt(x0*x0+y0*y0)
    return [FOS,x0,y0,R,D,1]
    pass

# CHI_PHI_SOIL_PROB(beta,H,Hw,Hc,Hwdash,c,phi,l,lw,q,Ht,num_simulations)
def CHI_PHI_SOIL_PROB(beta,H,Hw,Hc,Hwdash,D,lw,q,Ht,layers,num_simulations=1000):
    CHI_PHI_SOIL_DET_EXT(beta,H,Hw,Hc,Hwdash,D,lw,q,Ht,layers)
    # D=0
    c,phi,l=HOMO(beta,H,Hw,Hc,Hwdash,lw,q,Ht,layers)
    c_samples = generate_samples(*c, num_simulations)
    phi_samples = generate_samples(*phi, num_simulations)
    l_samples = generate_samples(*l, num_simulations)
    
    mean_Hw,std_Hw=Hw,0
    mean_Hc,std_Hc=Hc,0
    mean_Hwdash,std_Hwdash=Hwdash,0
    mean_H, std_H = H, 0
    mean_beta, std_beta = beta, 0
    mean_D, std_D = D, 0
    mean_lw,std_lw=lw,0
    mean_Ht,std_Ht=Ht,0
    mean_q,std_q=q,0
    
    Hw_samples = np.random.normal(mean_Hw, std_Hw, num_simulations)
    Hc_samples = np.random.normal(mean_Hc, std_Hc, num_simulations)
    Hwdash_samples = np.random.normal(mean_Hwdash, std_Hwdash, num_simulations)
    H_samples = np.random.normal(mean_H, std_H, num_simulations)
    beta_samples = np.random.normal(mean_beta, std_beta, num_simulations)
    D_samples = np.random.normal(mean_D, std_D, num_simulations)
    lw_samples = np.random.normal(mean_lw, std_lw, num_simulations)
    Ht_samples = np.random.normal(mean_Ht, std_Ht, num_simulations)
    q_samples = np.random.normal(mean_q, std_q, num_simulations)
    
    Fos_values = []
    x0_values = []
    y0_values = []
    D_values = []
    R_values=[]
    type_values=[]
    
    # Calculate FOS for each set of random samples
    for i in range(num_simulations):                                                      #beta,H,Hw,Hc,Hwdash,c,phi,l,lw,q,Ht
        fos_values, x0_val, y0_val,R_val,D_val,type_val = CHI_PHI_SOIL_MID(beta_samples[i], H_samples[i], Hw_samples[i],Hc_samples[i], Hwdash_samples[i],D_samples[i],c_samples[i],phi_samples[i], l_samples[i] ,lw_samples[i],q_samples[i], Ht_samples[i])
        Fos_values.append(fos_values)
        x0_values.append(x0_val)
        y0_values.append(y0_val)
        R_values.append(R_val)
        D_values.append(D_val)
        type_values.append(type_val)
    
        # Combine lists into a list of tuples
    combined = list(zip(Fos_values, x0_values, y0_values, R_values,D_values,type_values))

    # Sort the combined list of tuples based on the first element (Fos_values)
    sorted_combined = sorted(combined)

    # Unzip the sorted list of tuples back into separate lists
    Fos_values_sorted, x0_values_sorted, y0_values_sorted, R_values_sorted,D_values_sorted,type_values_sorted = zip(*sorted_combined)

    # Convert tuples back to lists
    Fos_values = list(Fos_values_sorted)
    x0_values = list(x0_values_sorted)
    y0_values = list(y0_values_sorted)
    R_values = list(R_values_sorted)
    D_values = list(D_values_sorted)
    type_values = list(type_values_sorted)
    
    Fos_values = np.array(Fos_values)
    x0_values = np.array(x0_values)
    y0_values = np.array(y0_values)
    R_values=np.array(R_values)
    D_values = np.array(D_values)
    type_values = np.array(type_values)
    
    DrawSolution(x0_values,y0_values,D_values,H,beta,type_values,q,Hw,Hwdash,Fos_values)
    
    # Create a figure
    plt.figure(figsize=(10, 8))

    # First subplot
    plt.subplot(3, 1, 1)  # (rows, columns, index)
    plt.hist(c_samples, bins=50, edgecolor='k', alpha=0.7)
    plt.title('Distribution of Cohesion')
    plt.xlabel('Cohesion')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Second subplot
    plt.subplot(3, 1, 2)
    plt.hist(phi_samples, bins=50, edgecolor='k', alpha=0.7)
    plt.title('Distribution of Friction Angle')
    plt.xlabel('Friction Angle')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Third subplot
    plt.subplot(3, 1, 3)
    plt.hist(l_samples, bins=50, edgecolor='k', alpha=0.7)
    plt.title('Distribution of Unit weight of soil')
    plt.xlabel('Unit weight of soil')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # plt.show()

    # Plot the distribution of FOS
    plt.figure(figsize=(10, 6))
    plt.hist(Fos_values, bins=50, edgecolor='k', alpha=0.7)
    plt.title('Distribution of Factor of Safety (FoS)')
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
    plt.plot(fos_range, prob_of_failure, label='Probability of Failure', color='red',linewidth=3)
    plt.xlabel('Factor of Safety')
    plt.ylabel('Probability of Failure')
    plt.title('Probability of Failure vs Factor of Safety')
    plt.grid(True)
    plt.legend()
    plt.show()
    return Fos_values
    pass