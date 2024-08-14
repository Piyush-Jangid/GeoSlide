import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

from .utills import getFromDict,getDataFromExcel,generate_samples,interpolate_2d_array,DrawSolution
from .adjustment_factor import getSurchargeFactor,getSubmergenceAndSeepageFactor,getSteadySeepageFactor,getTensionCrackFactor
from .nonhomogenous import HOMO_iter
# from .nonhomogenous import HOMO

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

# Worker function to compute FOS for a chunk of simulations
def compute_FOS_chunk(params_chunk):
    beta_chunk, H_chunk, Hw_chunk, Hwdash_chunk, D_chunk,c_chunk,lw_chunk,l_chunk,Ht_chunk,q_chunk = params_chunk
    return [PURELY_COHESIVE_SOIL_DET_FOR_PROB(beta_chunk[i], H_chunk[i], Hw_chunk[i], Hwdash_chunk[i], D_chunk[i],c_chunk[i],lw_chunk[i],l_chunk[i],Ht_chunk[i],q_chunk[i]) for i in range(len(c_chunk))]


def PURELY_COHESIVE_SOIL_DET(beta,H,Hw,Hwdash,D,lw,Ht,q,layers):
    #Deep Circle
    d = D/H
    if d>0.5:
        x01=H*interpolate_2d_array(x0_all,beta)
    else:
        x01=H*getFromDict(x0_dict,D/H,beta)
    if d>3:
        y01=H*getFromDict(y0_dict,3,beta)
    else:
        y01=H*getFromDict(y0_dict,D/H,beta)

    c,phi,l=HOMO_iter(x01,y01,y01+D,H,layers)
    c=c[0]
    l=l[0]
    uq1 = getSurchargeFactor(q,l,H,beta,D,2)# Surcharge adjustment factor
    uw1,uwdash1 = getSubmergenceAndSeepageFactor(Hw,Hwdash,H,beta,D,2) # Submergence and seepage adjustment factor
    ut1 = getTensionCrackFactor(Ht,H,beta,D,2)  # Tension Crack adjustment factor

    N1=0
    if d>3:
        N1=getFromDict(my_dict,3,beta)
    else:
        N1 = getFromDict(my_dict,d,beta)
    
    Pd1 = ((l*H+q)-(lw*Hw))/(uq1*uw1*ut1)
    FOS1 = N1*c/Pd1

    #Toe Circle
    d = D/H
    x02=H*interpolate_2d_array(x0_all,beta)
    y02=H*interpolate_2d_array(y0_toe,beta)

    c,phi,l=HOMO_iter(x02,y02,np.sqrt(x02*x02+y02*y02),H,layers)
    c=c[0]
    l=l[0]

    uq2 = getSurchargeFactor(q,l,H,beta,D,1)# Surcharge adjustment factor
    uw2,uwdash2 = getSubmergenceAndSeepageFactor(Hw,Hwdash,H,beta,D,1) # Submergence and seepage adjustment factor
    ut2 = getTensionCrackFactor(Ht,H,beta,D,1)  # Tension Crack adjustment factor


    N2 = interpolate_2d_array(toe_circle,beta)
    Pd2 = ((l*H+q)-(lw*Hw))/(uq2*uw2*ut2)
    FOS2 = N2*c/Pd2
    
    #Slope Circle
    d = 0
    x03=H*getFromDict(x0_dict,0,beta)
    y03=H*getFromDict(y0_dict,0,beta)

    c,phi,l=HOMO_iter(x03,y03,y03,H,layers)
    c=c[0]
    l=l[0]

    uq3 = getSurchargeFactor(q,l,H,beta,0,3)# Surcharge adjustment factor
    uw3,uwdash3 = getSubmergenceAndSeepageFactor(Hw,Hwdash,H,beta,0,3) # Submergence and seepage adjustment factor
    ut3 = getTensionCrackFactor(Ht,H,beta,0,3)  # Tension Crack adjustment factor

    N3 = getFromDict(my_dict,0,beta)
    Pd3 = ((l*H+q)-(lw*Hw))/(uq3*uw3*ut3)
    FOS3 = N3*c/Pd3

    if FOS1<FOS2 and FOS1<FOS3:
        print('Factor of safety For Deep circle: ',FOS1)
        print('X0 = ',x01)
        print('Y0 = ',y01)
        R=y01+D
        print('Radius of slip circle = ',R)
        print('Deep Circle')
        DrawSolution(x0_list=[x01],y0_list=[y01],D_list=[D],H=H,beta=beta,T_list=[2],q=q,Hw=Hw,Hwdash=Hwdash,fos_values=[FOS1])
        return [FOS1,x01,y01,R]
    elif FOS2<FOS1 and FOS2<FOS3:
        print('Factor of safety For Toe circle: ',FOS2)
        print('X0 = ',x02)
        print('Y0 = ',y02)
        R= np.sqrt(x02*x02+y02*y02)
        print('Radius of slip circle = ',R)
        print('Toe Circle')
        DrawSolution(x0_list=[x02],y0_list=[y02],D_list=[D],H=H,beta=beta,T_list=[1],q=q,Hw=Hw,Hwdash=Hwdash,fos_values=[FOS2])
        return [FOS2,x02,y02,R]
    else:
        print('Factor of safety For Slope circle: ',FOS3)
        print('X0 = ',x03)
        print('Y0 = ',y03)
        R= y03
        print('Radius of slip circle = ',R)
        print('Slope Circle')
        DrawSolution(x0_list=[x03],y0_list=[y03],D_list=[D],H=H,beta=beta,T_list=[3],q=q,Hw=Hw,Hwdash=Hwdash,fos_values=[FOS3])
        return [FOS3,x03,y03,R]
    pass
def PURELY_COHESIVE_SOIL_DET_EXT(beta,H,Hw,Hwdash,D,lw,Ht,q,layers):
    #Deep Circle
    d = D/H
    if d>0.5:
        x01=H*interpolate_2d_array(x0_all,beta)
    else:
        x01=H*getFromDict(x0_dict,D/H,beta)
    if d>3:
        y01=H*getFromDict(y0_dict,3,beta)
    else:
        y01=H*getFromDict(y0_dict,D/H,beta)

    c,phi,l=HOMO_iter(x01,y01,y01+D,H,layers)
    c=c[0]
    l=l[0]
    uq1 = getSurchargeFactor(q,l,H,beta,D,2)# Surcharge adjustment factor
    uw1,uwdash1 = getSubmergenceAndSeepageFactor(Hw,Hwdash,H,beta,D,2) # Submergence and seepage adjustment factor
    ut1 = getTensionCrackFactor(Ht,H,beta,D,2)  # Tension Crack adjustment factor

    N1=0
    if d>3:
        N1=getFromDict(my_dict,3,beta)
    else:
        N1 = getFromDict(my_dict,d,beta)
    
    Pd1 = ((l*H+q)-(lw*Hw))/(uq1*uw1*ut1)
    FOS1 = N1*c/Pd1

    #Toe Circle
    d = D/H
    x02=H*interpolate_2d_array(x0_all,beta)
    y02=H*interpolate_2d_array(y0_toe,beta)

    c,phi,l=HOMO_iter(x02,y02,np.sqrt(x02*x02+y02*y02),H,layers)
    c=c[0]
    l=l[0]

    uq2 = getSurchargeFactor(q,l,H,beta,D,1)# Surcharge adjustment factor
    uw2,uwdash2 = getSubmergenceAndSeepageFactor(Hw,Hwdash,H,beta,D,1) # Submergence and seepage adjustment factor
    ut2 = getTensionCrackFactor(Ht,H,beta,D,1)  # Tension Crack adjustment factor


    N2 = interpolate_2d_array(toe_circle,beta)
    Pd2 = ((l*H+q)-(lw*Hw))/(uq2*uw2*ut2)
    FOS2 = N2*c/Pd2
    
    #Slope Circle
    # D=0
    # d = D/H
    x03=H*getFromDict(x0_dict,0,beta)
    y03=H*getFromDict(y0_dict,0,beta)

    c,phi,l=HOMO_iter(x03,y03,y03,H,layers)
    c=c[0]
    l=l[0]

    uq3 = getSurchargeFactor(q,l,H,beta,D,3)# Surcharge adjustment factor
    uw3,uwdash3 = getSubmergenceAndSeepageFactor(Hw,Hwdash,H,beta,D,3) # Submergence and seepage adjustment factor
    ut3 = getTensionCrackFactor(Ht,H,beta,D,3)  # Tension Crack adjustment factor

    N3 = getFromDict(my_dict,0,beta)
    Pd3 = ((l*H+q)-(lw*Hw))/(uq3*uw3*ut3)
    FOS3 = N3*c/Pd3

    print('Deterministic Solution with Zero standard deviation')
    if FOS1<FOS2 and FOS1<FOS3:
        print('Factor of safety For Deep circle: ',FOS1)
        print('X0 = ',x01)
        print('Y0 = ',y01)
        R=y01+D
        print('Radius of slip circle = ',R)
        print('Deep Circle')
        # DrawSolution(x0_list=[x01],y0_list=[y01],D_list=[D],H=H,beta=beta,T_list=[2],q=q,Hw=Hw,Hwdash=Hwdash,fos_values=[FOS1])
        return [FOS1,x01,y01,R]
    elif FOS2<FOS1 and FOS2<FOS3:
        print('Factor of safety For Toe circle: ',FOS2)
        print('X0 = ',x02)
        print('Y0 = ',y02)
        R= np.sqrt(x02*x02+y02*y02)
        print('Radius of slip circle = ',R)
        print('Toe Circle')
        # DrawSolution(x0_list=[x02],y0_list=[y02],D_list=[D],H=H,beta=beta,T_list=[1],q=q,Hw=Hw,Hwdash=Hwdash,fos_values=[FOS2])
        return [FOS2,x02,y02,R]
    else:
        print('Factor of safety For Slope circle: ',FOS3)
        print('X0 = ',x03)
        print('Y0 = ',y03)
        R= y03
        print('Radius of slip circle = ',R)
        print('Slope Circle')
        # DrawSolution(x0_list=[x03],y0_list=[y03],D_list=[D],H=H,beta=beta,T_list=[3],q=q,Hw=Hw,Hwdash=Hwdash,fos_values=[FOS3])
        return [FOS3,x03,y03,R]
    pass


def PURELY_COHESIVE_SOIL_MID(beta,H,Hw,Hwdash,D,lw,Ht,q,layers):
    #Deep Circle
    d = D/H
    if d>0.5:
        x01=H*interpolate_2d_array(x0_all,beta)
    else:
        x01=H*getFromDict(x0_dict,D/H,beta)
    if d>3:
        y01=H*getFromDict(y0_dict,3,beta)
    else:
        y01=H*getFromDict(y0_dict,D/H,beta)

    c,phi,l=HOMO_iter(x01,y01,y01+D,H,layers)
    c=c[0]
    l=l[0]
    uq1 = getSurchargeFactor(q,l,H,beta,D,2)# Surcharge adjustment factor
    uw1,uwdash1 = getSubmergenceAndSeepageFactor(Hw,Hwdash,H,beta,D,2) # Submergence and seepage adjustment factor
    ut1 = getTensionCrackFactor(Ht,H,beta,D,2)  # Tension Crack adjustment factor

    N1=0
    if d>3:
        N1=getFromDict(my_dict,3,beta)
    else:
        N1 = getFromDict(my_dict,d,beta)
    
    Pd1 = ((l*H+q)-(lw*Hw))/(uq1*uw1*ut1)
    FOS1 = N1*c/Pd1

    #Toe Circle
    d = D/H
    x02=H*interpolate_2d_array(x0_all,beta)
    y02=H*interpolate_2d_array(y0_toe,beta)

    c,phi,l=HOMO_iter(x02,y02,np.sqrt(x02*x02+y02*y02),H,layers)
    c=c[0]
    l=l[0]

    uq2 = getSurchargeFactor(q,l,H,beta,D,1)# Surcharge adjustment factor
    uw2,uwdash2 = getSubmergenceAndSeepageFactor(Hw,Hwdash,H,beta,D,1) # Submergence and seepage adjustment factor
    ut2 = getTensionCrackFactor(Ht,H,beta,D,1)  # Tension Crack adjustment factor


    N2 = interpolate_2d_array(toe_circle,beta)
    Pd2 = ((l*H+q)-(lw*Hw))/(uq2*uw2*ut2)
    FOS2 = N2*c/Pd2
    
    #Slope Circle
    x03=H*getFromDict(x0_dict,0,beta)
    y03=H*getFromDict(y0_dict,0,beta)

    c,phi,l=HOMO_iter(x03,y03,y03,H,layers)
    c=c[0]
    l=l[0]

    uq3 = getSurchargeFactor(q,l,H,beta,0,3)# Surcharge adjustment factor
    uw3,uwdash3 = getSubmergenceAndSeepageFactor(Hw,Hwdash,H,beta,0,3) # Submergence and seepage adjustment factor
    ut3 = getTensionCrackFactor(Ht,H,beta,D,3)  # Tension Crack adjustment factor

    N3 = getFromDict(my_dict,0,beta)
    Pd3 = ((l*H+q)-(lw*Hw))/(uq3*uw3*ut3)
    FOS3 = N3*c/Pd3

    if FOS1<FOS2 and FOS1<FOS3:
        R=y01+D
        return [FOS1,x01,y01,R,2,D]
    elif FOS2<FOS1 and FOS2<FOS3:
        R= np.sqrt(x02*x02+y02*y02)
        return [FOS2,x02,y02,R,1,D]
    else:
        R= y03
        return [FOS3,x03,y03,R,3,0]
    pass

def PURELY_COHESIVE_SOIL_DET_FOR_PROB(beta,H,Hw,Hwdash,D,c,lw,l,Ht,q):
    #Deep Circle
    d = D/H
    if d>0.5:
        x01=H*interpolate_2d_array(x0_all,beta)
    else:
        x01=H*getFromDict(x0_dict,D/H,beta)
    if d>3:
        y01=H*getFromDict(y0_dict,3,beta)
    else:
        y01=H*getFromDict(y0_dict,D/H,beta)

    uq1 = getSurchargeFactor(q,l,H,beta,D,2)# Surcharge adjustment factor
    uw1,uwdash1 = getSubmergenceAndSeepageFactor(Hw,Hwdash,H,beta,D,2) # Submergence and seepage adjustment factor
    ut1 = getTensionCrackFactor(Ht,H,beta,D,2)  # Tension Crack adjustment factor

    N1=0
    if d>3:
        N1=getFromDict(my_dict,3,beta)
    else:
        N1 = getFromDict(my_dict,d,beta)
    
    Pd1 = ((l*H+q)-(lw*Hw))/(uq1*uw1*ut1)
    FOS1 = N1*c/Pd1

    #Toe Circle
    d = D/H
    x02=H*interpolate_2d_array(x0_all,beta)
    y02=H*interpolate_2d_array(y0_toe,beta)

    uq2 = getSurchargeFactor(q,l,H,beta,D,1)# Surcharge adjustment factor
    uw2,uwdash2 = getSubmergenceAndSeepageFactor(Hw,Hwdash,H,beta,D,1) # Submergence and seepage adjustment factor
    ut2 = getTensionCrackFactor(Ht,H,beta,D,1)  # Tension Crack adjustment factor

    N2 = interpolate_2d_array(toe_circle,beta)
    Pd2 = ((l*H+q)-(lw*Hw))/(uq2*uw2*ut2)
    FOS2 = N2*c/Pd2
    
    #Slope Circle
    x03=H*getFromDict(x0_dict,0,beta)
    y03=H*getFromDict(y0_dict,0,beta)

    uq3 = getSurchargeFactor(q,l,H,beta,D,3)# Surcharge adjustment factor
    uw3,uwdash3 = getSubmergenceAndSeepageFactor(Hw,Hwdash,H,beta,D,3) # Submergence and seepage adjustment factor
    ut3 = getTensionCrackFactor(Ht,H,beta,D,3)  # Tension Crack adjustment factor

    N3 = getFromDict(my_dict,0,beta)
    Pd3 = ((l*H+q)-(lw*Hw))/(uq3*uw3*ut3)
    FOS3 = N3*c/Pd3

    if FOS1<FOS2 and FOS1<FOS3:
        R=y01+D
        return [FOS1,x01,y01,R,2,D]
    elif FOS2<FOS1 and FOS2<FOS3:
        R= np.sqrt(x02*x02+y02*y02)
        return [FOS2,x02,y02,R,1,D]
    else:
        R= y03
        return [FOS3,x03,y03,R,3,0]
    pass

def PURELY_COHESIVE_SOIL_PROB(beta,H,Hw,Hwdash,D,lw,Ht,q,layers,num_simulations=1000):
    PURELY_COHESIVE_SOIL_DET_EXT(beta,H,Hw,Hwdash,D,lw,Ht,q,layers)
    a = PURELY_COHESIVE_SOIL_MID(beta,H,Hw,Hwdash,D,lw,Ht,q,layers)
    d=D/H
    if a[4]==1:
        x02=H*interpolate_2d_array(x0_all,beta)
        y02=H*interpolate_2d_array(y0_toe,beta)
        c,phi,l=HOMO_iter(x02,y02,np.sqrt(x02*x02+y02*y02),H,layers)
    elif a[4]==2:
        if d>0.5:
            x01=H*interpolate_2d_array(x0_all,beta)
        else:
            x01=H*getFromDict(x0_dict,D/H,beta)
        if d>3:
            y01=H*getFromDict(y0_dict,3,beta)
        else:
            y01=H*getFromDict(y0_dict,D/H,beta)
        c,phi,l=HOMO_iter(x01,y01,y01+D,H,layers)
    elif a[4]==3:
        x03=H*getFromDict(x0_dict,0,beta)
        y03=H*getFromDict(y0_dict,0,beta)
        c,phi,l=HOMO_iter(x03,y03,y03,H,layers)
        
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
    y0_values =[] 
    D_values =[] 
    R_values=[]
    type_values=[]
    
    # Calculate FOS for each set of random samples
    for i in range(num_simulations):
        fos_values, x0_val, y0_val,R_val,type_val,D_val = PURELY_COHESIVE_SOIL_DET_FOR_PROB(beta_samples[i], H_samples[i], Hw_samples[i], Hwdash_samples[i], D_samples[i],c_samples[i], lw_samples[i],l_samples[i],Ht_samples[i], q_samples[i])
        Fos_values.append(fos_values)
        x0_values.append(x0_val)
        y0_values.append(y0_val)
        D_values.append(D_val)
        R_values.append(R_val)
        type_values.append(type_val)
    # print(len(np.array(fos_values)))
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
    # print(np.mean(Fos_values))
    
    DrawSolution(x0_values,y0_values,D_values,H,beta,type_values,q,Hw,Hwdash,Fos_values)
    
    # Create a figure
    plt.figure(figsize=(10, 8))

    # First subplot
    plt.subplot(2, 1, 1)  # (rows, columns, index)
    plt.hist(c_samples, bins=50, edgecolor='k', alpha=0.7)
    plt.title('Distribution of Cohesion')
    plt.xlabel('Cohesion')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Second subplot
    plt.subplot(2, 1, 2)
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
