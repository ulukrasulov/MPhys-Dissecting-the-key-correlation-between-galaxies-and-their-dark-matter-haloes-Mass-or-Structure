import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Halo_catalogue.csv')


A_k1=0.020
A_k2=0.016
A_k3=0.009
h=0.6777

virial_radius=data['Virial_radius']


vir_mask=virial_radius>69

virial_radius=virial_radius[vir_mask]
central_mask=data['upid']==-1

virial_radius=virial_radius[central_mask]



length=len(data['Virial_radius'])
print(virial_radius.size)

virial_radius = virial_radius/h  #Virial radius in kpc
print(virial_radius[0:20])

log_effective_radii=np.zeros(len(virial_radius))


mask1=virial_radius<160
mask2=virial_radius>120



radius_mask1=virial_radius>160


radius_mask2=np.logical_and(mask1,mask2)

radius_mask3=virial_radius<120


log_effective_radii[radius_mask1]=np.log10(A_k1)+np.log10(virial_radius[radius_mask1]) 
log_effective_radii[radius_mask2]=np.log10(A_k2)+np.log10(virial_radius[radius_mask2]) 
log_effective_radii[radius_mask3]=np.log10(A_k3)+np.log10(virial_radius[radius_mask3]) 

print('HELLOO')

print(len(virial_radius))
log_effective_radii_scatter=np.zeros(len(log_effective_radii))
log_effective_radii_scatter[radius_mask1]=np.random.normal(log_effective_radii[radius_mask1], scale=0.1) #add scatter 0.2
log_effective_radii_scatter[radius_mask2]=np.random.normal(log_effective_radii[radius_mask2], scale=0.15) #add scatter 0.2
log_effective_radii_scatter[radius_mask3]=np.random.normal(log_effective_radii[radius_mask3], scale=0.2) #add scatter 0.2




print((log_effective_radii[0:20]))
print((log_effective_radii_scatter[0:20]))

#print(len(log_effective_radii))

Vol = (500/0.6777)**3
binwidth = 0.1
bins3 = np.arange(-0.6,1.6,binwidth)
SMF = np.histogram(log_effective_radii, bins=bins3)[0]/Vol/0.1
SMF_scatter = np.histogram(log_effective_radii_scatter, bins=bins3)[0]/Vol/0.1



df1=pd.read_csv('/home/uluk/Downloads/New_Catalog_SDSS_complete.dat',delim_whitespace=True)

print(df1.columns)

df1 = df1.query('Vmaxwt>0')

skycov=8000.

fracper1 = len(df1)/670722
fracsky1=(skycov*fracper1)/(4*np.pi*(180./np.pi)**2.)

SMF_BinWidth = 0.1
SMF_Bins = np.arange(-0.6,1.6,SMF_BinWidth)
All1 = np.histogram(df1.query('NewLCentSat==1')['logReSerExp'], weights=df1.query('NewLCentSat==1')['Vmaxwt'], bins=SMF_Bins)[0]/fracsky1/SMF_BinWidth
Num1 = np.histogram(df1.query('NewLCentSat==1')['logReSerExp'], bins=SMF_Bins)[0]/fracsky1/SMF_BinWidth

error1=All1/np.sqrt(Num1)*10 




plt.plot(bins3[1:]-0.1/2,SMF,label="Effective radius without scatter 0.010 (<120 vir) _0.016 (120-160) _0.021 (>160 vir)")
plt.plot(bins3[1:]-0.1/2,SMF_scatter,label="Mock Galaxies with scatter  0.010 (<120 vir 0.18) _0.016 (120-160 0.15) _0.021 (>160 vir 0.05)")
plt.errorbar(SMF_Bins[1:]-SMF_BinWidth/2,All1,yerr=error1,fmt='o',label="SDSS Central Galaxies ")
plt.yscale('log')
plt.legend(loc='upper right')
plt.show()

#data['scatter radii 0.010_0.016_0.021']=log_effective_radii_scatter
#data.to_csv("/media/uluk/Seagate Expansion Drive/Dissertation/Triple Kravstov/Radii 0.008 0.018 0.024/0.021 0.016 0.010 (scatter 0.05,0.15,0.18)/Halo_catalogue.csv")
