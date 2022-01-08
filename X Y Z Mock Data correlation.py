import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from Corrfunc.theory import wp

data=pd.read_csv('Central Stellar Masses.csv')

mass=data['stellar_mass']
print(len(mass))
mass_mask1=mass>11.3
mass_mask2=mass<11.7

mass_mask=np.logical_and(mass_mask1,mass_mask2)

data_x = data['x']
data_y = data['y']
data_z = data['z']

period=500


pimax=40
threads = 4


Rbins = np.linspace(-0.25,1,15)
rbins = 10**Rbins # for the wp calculation


data_x = data_x[mass_mask]
data_y = data_y[mass_mask]
data_z = data_z[mass_mask]

wp_results = wp(period, pimax, threads, rbins, data_x, data_y, data_z, verbose=True) #Calculate Projected correlation from X,Y,Z 
wp1 = wp_results['wp']


np.savetxt("wp_11.3-11.7_Grylls.txt",wp1)

