import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from M_star_from_distribution import *




data = pd.read_csv('new_catalogue.csv')


effective_radius= data['radii 0.007_0.015_0.017'] #This is where the effective radius is saved in unis log10(kpc)
print(effective_radius.size)


central_mask=data['upid']==-1

satellite_mask=data['upid']!=-1

print(sum(central_mask))
print(sum(satellite_mask))


print(np.amax(effective_radius))


print(len(effective_radius))



stellar_mass=np.zeros(16689195)

print("MAXIMUM CENTRAL RADIUS " + str(max(effective_radius[central_mask])))
print("MINIMUM CENTRAL RADIUS " + str(min(effective_radius[central_mask])))


print("MAXIMUM SATELLITE RADIUS " + str(max(effective_radius[satellite_mask])))
print("MINIMUM SATELLITE RADIUS " + str(min(effective_radius[satellite_mask])))

Re_centrals= np.arange(-2.0,2.2,0.1)
Re_centrals=np.around(Re_centrals,1)

print(Re_centrals)




stellar_mass.fill(-999)

for i in range(41):
    
    l=i+1

    print("Upper Bound " + str(Re_centrals[l]))
    print("lower Bound " + str(Re_centrals[i]))
    radius_mask1=effective_radius <Re_centrals[l]
    radius_mask2=effective_radius >=Re_centrals[i]
    central_mask=data['upid']==-1
    radius_mask=np.logical_and(radius_mask1,radius_mask2)
    radius_mask=np.logical_and(radius_mask,central_mask)
    total=np.sum(radius_mask)
    print("TOTAL " + str(total))

    stellar_mass[radius_mask]=Mstar_from_size_central(Re_centrals[i],Re_centrals[l],total)
    print(effective_radius[radius_mask])
    print(stellar_mass[radius_mask])
    print(total)
    

Re_satellites= Re_centrals= np.arange(-0.3,1.7,0.1)
Re_satellites=np.around(Re_satellites,1)

print(Re_satellites)


print(stellar_mass.size)
print(stellar_mass)

for i in range(19):
    
    l=i+1

    print("Upper Bound " + str(Re_satellites[l]))
    print("lower Bound " + str(Re_satellites[i]))
    radius_mask1=effective_radius <Re_satellites[l]
    radius_mask2=effective_radius >=Re_satellites[i]
    satellite_mask=data['upid']!=-1
    radius_mask=np.logical_and(radius_mask1,radius_mask2)
    radius_mask=np.logical_and(radius_mask,satellite_mask)
    total=np.sum(radius_mask)
    print(radius_mask)
    print(total)
    print(stellar_mass[radius_mask])
    stellar_mass[radius_mask]=Mstar_from_size_satellites(Re_satellites[i],Re_satellites [l],total)
    print(effective_radius[radius_mask])
    print(stellar_mass[radius_mask])
    print(total)
    
real_mask=stellar_mass!=-999


data['mass_radii 0.007_0.015_0.017']=stellar_mass #Save stellar mass here



data.to_csv("new_catalogue.csv",index=False)
