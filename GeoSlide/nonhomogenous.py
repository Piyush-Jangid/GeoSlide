from SlopeStability.adjustment_factor import getSteadySeepageFactor, getSurchargeFactor, getTensionCrackFactor, getSubmergenceAndSeepageFactor
from SlopeStability.utills import getDataFromExcel,getFromDict
import math
import numpy as np


file_path='Stability Number for c-phi soil.xlsx'
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


def find_intersection_circle_horizontal(xc, yc, r, y):
    delta = r**2 - (y - yc)**2
    if delta < 0:
        return []  # No intersection points
    x1 = xc + math.sqrt(delta)
    x2 = xc - math.sqrt(delta)
    return [(x1, y), (x2, y)]

def calculate_central_angle(xc, yc, r, x1, y1, x2, y2):
    angle1 = math.atan2(y1 - yc, x1 - xc)
    angle2 = math.atan2(y2 - yc, x2 - xc)
    central_angle = abs(math.degrees(angle2 - angle1))
    if central_angle > 180:
        central_angle = 360 - central_angle
    return central_angle

def weighted_average(values, weights):
    total_weight = sum(weights)
    if total_weight == 0:
        return 0
    return sum(v * w for v, w in zip(values, weights)) / total_weight

def average_parameters(layers, angles,H):
    layer1,layer2=func(layers,H)
    cohesions = [layer['cohesion'][0] for layer in layers]

    friction_angles = [layer['friction_angle'][0] for layer in layers]
    unit_weights = [layer['unit_weight'][0] for layer in layer1]
    thicknesses = [layer['thickness'] for layer in layer1]

    # Apply math.radians to each element in the list
    radian_angles = [math.radians(angle) for angle in friction_angles]

    # Apply math.tan to each radian value
    friction_angles = [math.tan(angle) for angle in radian_angles]

    c_avg = weighted_average(cohesions, angles)
    phi_avg = math.degrees(math.atan(weighted_average(friction_angles, angles)))
    gamma_avg = weighted_average(unit_weights, thicknesses)
    
    return c_avg, phi_avg, gamma_avg
def func(layers, H):
    layer1 = []
    layer2 = []
    cumulative_thickness = 0
    for layer in layers:
        # print(layer)
        # Calculate the remaining height above the toe
        remaining_height_above_toe = max(H - cumulative_thickness,0)

        if layer['thickness'] <= remaining_height_above_toe:
            # If the whole layer is above the toe
            # print(layer)
            temp_layer = {
                    'cohesion': layer['cohesion'],
                    'friction_angle': layer['friction_angle'],
                    'unit_weight': layer['unit_weight'],
                    'thickness': layer['thickness']
                }
            layer1.append(temp_layer)
            cumulative_thickness += layer['thickness']
        else:
            # Split the layer into two parts
            if remaining_height_above_toe > 0:
                # Part of the layer above the toe
                layer_above_toe = {
                    'cohesion': layer['cohesion'],
                    'friction_angle': layer['friction_angle'],
                    'unit_weight': layer['unit_weight'],
                    'thickness': remaining_height_above_toe
                }
                layer1.append(layer_above_toe)

            # Part of the layer below the toe
            thickness_below_toe = layer['thickness'] - remaining_height_above_toe
            if thickness_below_toe > 0:
                layer_below_toe = {
                    'cohesion': layer['cohesion'],
                    'friction_angle': layer['friction_angle'],
                    'unit_weight': layer['unit_weight'],
                    'thickness': thickness_below_toe
                }
                layer2.append(layer_below_toe)
            cumulative_thickness += layer['thickness']

    return layer1, layer2

def HOMO_iter(xc,yc,r,H,layers):
    # Example usage
    layer1,layer2=func(layers,H)
    # print(layer1)
    # print(layer2)

    angles = []

    # Determine intersection points and angles for each layer above the toe level
    y = H
    for i, layer in enumerate(layer1):
        points = find_intersection_circle_horizontal(xc, yc, r, y)
        if not points:
            continue  # No intersection points
        # Choose the point with the greater x value
        if points[0][0] > points[1][0]:
            x1, y1 = points[0]
        else:
            x1, y1 = points[1]
        
        y -= layer['thickness']
        points = find_intersection_circle_horizontal(xc, yc, r, y)
        if not points:
            continue  # No intersection points
        # Choose the point with the greater x value
        if points[0][0] > points[1][0]:
            x2, y2 = points[0]
        else:
            x2, y2 = points[1]

        angle = calculate_central_angle(xc, yc, r, x1, y1, x2, y2)
        angles.append(angle)

    # For layers below the toe level
    y = 0
    for i, layer in enumerate(layer2):
        y1 -= layer['thickness']
        if yc + abs(y1) >= r:
            points = find_intersection_circle_horizontal(xc, yc, r, y)
            angle = calculate_central_angle(xc, yc, r, points[0][0], points[0][1], points[1][0], points[1][1])
            angles.append(angle)
            break
        points = find_intersection_circle_horizontal(xc, yc, r, y)
        if not points:
            continue  # No intersection points
        # Choose the point with the greater x value
        if points[0][0] > points[1][0]:
            x1, y1 = points[0]
        else:
            x1, y1 = points[1]
        
        y_next = y - layer['thickness']
        if yc + abs(y_next) >= r:
            # Last layer, calculate the central angle using both intersection points
            angle = calculate_central_angle(xc, yc, r, points[0][0], points[0][1], points[1][0], points[1][1])
        else:
            # Choose the point with the greater x value for the next layer
            points_next = find_intersection_circle_horizontal(xc, yc, r, y_next)
            if points_next:
                if points_next[0][0] > points_next[1][0]:
                    x2, y2 = points_next[0]
                else:
                    x2, y2 = points_next[1]
                angle = calculate_central_angle(xc, yc, r, x1, y1, x2, y2)
            else:
                angle = 0  # No intersection points in the next layer

        angles.append(angle)
    # print(angles)
    c_avg, phi_avg, gamma_avg = average_parameters(layer1+layer2, angles,H)
    return [c_avg, layers[0]['cohesion'][1], layers[0]['cohesion'][2]],[phi_avg, layers[0]['friction_angle'][1], layers[0]['friction_angle'][2]] ,[gamma_avg, layers[0]['unit_weight'][1], layers[0]['unit_weight'][2]]
    pass

def average_above_toe(layers,H):
    layer1,layer2=func(layers,H)
    cohesions = [layer['cohesion'][0] for layer in layer1]
    friction_angles = [layer['friction_angle'][0] for layer in layer1]
    unit_weights = [layer['unit_weight'][0] for layer in layer1]
    thicknesses = [layer['thickness'] for layer in layer1]

    c_avg = weighted_average(cohesions, thicknesses)
    phi_avg = weighted_average(friction_angles, thicknesses)
    gamma_avg = weighted_average(unit_weights, thicknesses)
    
    return c_avg, phi_avg, gamma_avg

def HOMO(beta,H,Hw,Hc,Hwdash,lw,q,Ht,layers):
    D=0
    c,phi,l=average_above_toe(layers,H)
    uq = getSurchargeFactor(q,l,H,beta,D,1)# Surcharge adjustment factor
    uw,uwdash = getSubmergenceAndSeepageFactor(Hw,Hwdash,H,beta,D,1) # Submergence and seepage adjustment factor
    ut = getTensionCrackFactor(Ht,H,beta,D,1)  # Tension Crack adjustment factor

    if Hc!=0:
        Hwdash,uwdash=getSteadySeepageFactor(Hc,H,beta,0,1)

    Pe = ((l*H+q)-(lw*Hwdash))/(uq*uwdash)
    # lcf=Pe*(np.tan(np.radians(phi))/c)
    i=1
    x=0
    y=0
    r=0
    rp=-100000
    while (r-rp)>1e-2:
        if i!=1:
            c,phi,l=HOMO_iter(x,y,r,H,layers)
            c=c[0]
            phi=phi[0]
            l=l[0]
            rp=r
        lcf=Pe*(np.tan(np.radians(phi))/c)
        x=H*getFromDict(xlcf_dict,lcf,1/np.tan(np.radians(beta)))
        y=H*getFromDict(ylcf_dict,lcf,1/np.tan(np.radians(beta)))
        r=np.sqrt(x*x+y*y)
        # print(x,y)
        # print(r)
        # print(c)
        # print(i,np.tan(np.radians(phi)))
        i=i+1
    return [c, layers[0]['cohesion'][1], layers[0]['cohesion'][2]],[phi, layers[0]['friction_angle'][1], layers[0]['friction_angle'][2]] ,[l, layers[0]['unit_weight'][1], layers[0]['unit_weight'][2]]
    pass


layers=[
    {'cohesion': [800,0.2,'normal'], 'friction_angle': [8,0.2,'normal'], 'unit_weight': [115,0.2,'normal'], 'thickness': 20},
    {'cohesion': [600,0.2,'normal'], 'friction_angle': [6,0.2,'normal'], 'unit_weight': [110,0.2,'normal'], 'thickness': 20},
    {'cohesion': [800,0.2,'normal'], 'friction_angle': [0,0.2,'normal'], 'unit_weight': [120,0.2,'normal'], 'thickness': 20},
]
# print(calculate_central_angle(26.056, 59.7,65.138353 , 0, 0, 52.112, 0))
# c_avg, phi_avg, gamma_avg=HOMO(33.69,40,0,0,0,62.4,0,0,layers)
# print(np.tan(np.radians(phi_avg[0])))
# print(f"Average Cohesion: {c_avg} kPa")
# print(f"Average Friction Angle: {phi_avg} degrees")
# print(f"Average Unit Weight: {gamma_avg} kN/m^3")

