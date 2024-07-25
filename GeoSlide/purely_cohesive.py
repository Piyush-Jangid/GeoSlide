import matplotlib.pyplot as plt
import numpy as np

from .utills import getFromDict,getDataFromExcel,generate_samples,interpolate_2d_array,DrawSolution
from .adjustment_factor import getSurchargeFactor,getSubmergenceAndSeepageFactor,getSteadySeepageFactor,getTensionCrackFactor

file_path='PHI=0.xlsx'
toe_circle = getDataFromExcel(file_path,'Toe_circle')
d_0 = getDataFromExcel(file_path,'d=0')
d_01 = getDataFromExcel(file_path,'d=0.1')
d_02 = getDataFromExcel(file_path,'d=0.2')
d_03 = getDataFromExcel(file_path,'d=0.3')
d_05 = getDataFromExcel(file_path,'d=0.5')
d_1 = getDataFromExcel(file_path,'d=1')
d_105 = getDataFromExcel(file_path,'d=1.5')
d_2 = getDataFromExcel(file_path,'d=2')
d_3 = getDataFromExcel(file_path,'d=3')
d_inf=5.641

my_dict={
    0:d_0,
    0.1:d_01,
    0.2:d_02,
    0.3:d_03,
    0.5:d_05,
    1:d_1,
    1.5:d_105,
    2:d_2,
    3:d_3
}

x0_all=getDataFromExcel(file_path,'x_all_circle')
x0_d05=getDataFromExcel(file_path,'x_d=0.5')
x0_d0=getDataFromExcel(file_path,'x_d=0')

x0_dict={
    0:x0_d0,
    0.5:x0_d05
}

y0_toe=getDataFromExcel(file_path,'y_toe_circle')
y0_d0=getDataFromExcel(file_path,'y_d=0')
y0_d1=getDataFromExcel(file_path,'y_d=1')
y0_d2=getDataFromExcel(file_path,'y_d=2')
y0_d3=getDataFromExcel(file_path,'y_d=3')

y0_dict={
    0:y0_d0,
    1:y0_d1,
    2:y0_d2,
    3:y0_d3
}



def PURELY_COHESIVE_SOIL_DET(beta,H,Hw,Hwdash,D,c,lw,l,Ht,q):
    #Deep Circle
    uq1 = getSurchargeFactor(q,l,H,beta,D,2)# Surcharge adjustment factor
    uw1,uwdash1 = getSubmergenceAndSeepageFactor(Hw,Hwdash,H,beta,D,2) # Submergence and seepage adjustment factor
    ut1 = getTensionCrackFactor(Ht,H,beta,D,2)  # Tension Crack adjustment factor

    d = D/H
    N1=0
    if d>3:
        N1=getFromDict(my_dict,3,beta)
    else:
        N1 = getFromDict(my_dict,d,beta)
    Pd1 = ((l*H+q)-(lw*Hw))/(uq1*uw1*ut1)
    FOS1 = N1*c/Pd1
    if d>0.5:
        x01=H*interpolate_2d_array(x0_all,beta)
    else:
        x01=H*getFromDict(x0_dict,D/H,beta)
    if d>3:
        y01=H*getFromDict(y0_dict,3,beta)
    else:
        y01=H*getFromDict(y0_dict,D/H,beta)

    #Toe Circle
    uq2 = getSurchargeFactor(q,l,H,beta,D,1)# Surcharge adjustment factor
    uw2,uwdash2 = getSubmergenceAndSeepageFactor(Hw,Hwdash,H,beta,D,1) # Submergence and seepage adjustment factor
    ut2 = getTensionCrackFactor(Ht,H,beta,D,1)  # Tension Crack adjustment factor

    d = D/H

    N2 = interpolate_2d_array(toe_circle,beta)
    Pd2 = ((l*H+q)-(lw*Hw))/(uq2*uw2*ut2)
    FOS2 = N2*c/Pd2
    x02=H*interpolate_2d_array(x0_all,beta)
    y02=H*interpolate_2d_array(y0_toe,beta)
    
    if FOS1<FOS2:
        print('Factor of safety For Deep circle: ',FOS1)
        print('X0 = ',x01)
        print('Y0 = ',y01)
        R=y01+D
        print('Radius of slipe circle = ',R)
        print('Deep Circle')
        DrawSolution(x0_list=[x01],y0_list=[y01],D_list=[D],H=H,beta=beta,T_list=[2],q=q,Hw=Hw,Hwdash=Hwdash,fos_values=[FOS1])
        return [FOS1,x01,y01,R]
    else:
        print('Factor of safety For Toe circle: ',FOS2)
        print('X0 = ',x02)
        print('Y0 = ',y02)
        R= np.sqrt(x02*x02+y02*y02)
        print('Radius of slipe circle = ',R)
        DrawSolution(x0_list=[x02],y0_list=[y02],D_list=[D],H=H,beta=beta,T_list=[1],q=q,Hw=Hw,Hwdash=Hwdash,fos_values=[FOS2])
        return [FOS2,x02,y02,R]
        pass

def PURELY_COHESIVE_SOIL_DET_FOR_PROB(beta,H,Hw,Hwdash,D,c,lw,l,Ht,q):
    #Deep Circle
    uq1 = getSurchargeFactor(q,l,H,beta,D,2)# Surcharge adjustment factor
    uw1,uwdash1 = getSubmergenceAndSeepageFactor(Hw,Hwdash,H,beta,D,2) # Submergence and seepage adjustment factor
    ut1 = getTensionCrackFactor(Ht,H,beta,D,2)  # Tension Crack adjustment factor

    d = D/H
    N1=0
    if d>3:
        N1=getFromDict(my_dict,3,beta)
    else:
        N1 = getFromDict(my_dict,d,beta)
    Pd1 = ((l*H+q)-(lw*Hw))/(uq1*uw1*ut1)
    FOS1 = N1*c/Pd1
    if d>0.5:
        x01=H*interpolate_2d_array(x0_all,beta)
    else:
        x01=H*getFromDict(x0_dict,D/H,beta)
    if d>3:
        y01=H*getFromDict(y0_dict,3,beta)
    else:
        y01=H*getFromDict(y0_dict,D/H,beta)

    #Toe Circle
    uq2 = getSurchargeFactor(q,l,H,beta,D,1)# Surcharge adjustment factor
    uw2,uwdash2 = getSubmergenceAndSeepageFactor(Hw,Hwdash,H,beta,D,1) # Submergence and seepage adjustment factor
    ut2 = getTensionCrackFactor(Ht,H,beta,D,1)  # Tension Crack adjustment factor

    d = D/H

    N2 = interpolate_2d_array(toe_circle,beta)
    Pd2 = ((l*H+q)-(lw*Hw))/(uq2*uw2*ut2)
    FOS2 = N2*c/Pd2
    x02=H*interpolate_2d_array(x0_all,beta)
    y02=H*interpolate_2d_array(y0_toe,beta)
    
    if FOS1<FOS2:
        R=y01+D
        return [FOS1,x01,y01,R,2,D]
    else:
        R= np.sqrt(x02*x02+y02*y02)
        return [FOS2,x02,y02,R,1,D]
    
def PURELY_COHESIVE_SOIL_PROB(beta,H,Hw,Hwdash,D,c,lw,l,Ht,q,num_simulations=10000):
    c_samples = generate_samples(*c, num_simulations)
#     phi_samples = generate_samples(*phi, num_simulatsions)
    l_samples = generate_samples(*l, num_simulations)
    
    mean_Hw,std_Hw=Hw,0
    mean_Hwdash,std_Hwdash=Hwdash,0
    mean_H, std_H = H, 0
    mean_beta, std_beta = beta, 0
    mean_D, std_D = D, 0
    mean_lw,std_lw=lw,0
    mean_Ht,std_Ht=Ht,0
    mean_q,std_q=q,0
    
    Hw_samples = np.random.normal(mean_Hw, std_Hw, num_simulations)
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
    for i in range(num_simulations):
        fos_values, x0_val, y0_val,R_val,type_val,D_val = PURELY_COHESIVE_SOIL_DET_FOR_PROB(beta_samples[i], H_samples[i], Hw_samples[i], Hwdash_samples[i], D_samples[i], c_samples[i], lw_samples[i], l_samples[i], Ht_samples[i], q_samples[i])
        Fos_values.append(fos_values)
        x0_values.append(x0_val)
        y0_values.append(y0_val)
        D_values.append(D_val)
        R_values.append(R_val)
        type_values.append(type_val)
    # Combine lists into a list of tuples
    combined = list(zip(Fos_values, x0_values, y0_values, D_values, R_values, type_values))

    # Sort the combined list of tuples based on the first element (Fos_values)
    sorted_combined = sorted(combined)

    # Unzip the sorted list of tuples back into separate lists
    Fos_values_sorted, x0_values_sorted, y0_values_sorted, D_values_sorted, R_values_sorted, type_values_sorted = zip(*sorted_combined)

    # Convert tuples back to lists
    Fos_values = list(Fos_values_sorted)
    x0_values = list(x0_values_sorted)
    y0_values = list(y0_values_sorted)
    D_values = list(D_values_sorted)
    R_values = list(R_values_sorted)
    type_values = list(type_values_sorted)

    
    Fos_values = np.array(Fos_values)
    x0_values = np.array(x0_values)
    y0_values = np.array(y0_values)
    R_values=np.array(R_values)
    D_values = np.array(D_values)
    type_values = np.array(type_values)
    
    DrawSolution(x0_values,y0_values,D_values,H,beta,type_values,q,Hw,Hwdash,Fos_values)
    
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
    plt.plot(fos_range, prob_of_failure, label='Probability of Failure', color='red')
    plt.xlabel('Factor of Safety')
    plt.ylabel('Probability of Failure')
    plt.title('Probability of Failure vs Factor of Safety')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    