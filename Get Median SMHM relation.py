import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import DescrStatsW
import sklearn

plt.rcParams['ytick.minor.visible']=True
plt.rcParams['xtick.minor.visible']=True
plt.rcParams['axes.linewidth']=5
plt.rcParams['xtick.major.size'] =15
plt.rcParams['ytick.major.size'] =15
plt.rcParams['xtick.minor.size'] =10
plt.rcParams['ytick.minor.size'] =10
plt.rcParams['xtick.major.width'] =5
plt.rcParams['ytick.major.width'] =5
plt.rcParams['xtick.minor.width'] =5
plt.rcParams['ytick.minor.width'] =5
plt.rcParams['axes.titlepad'] = 10

plt.rcParams['figure.figsize']=(12,16)

h=0.6777
data=pd.read_csv('Central Stellar Masses.csv')
mass=data['stellar_mass']
mass_mask1=mass>11.3
mass_mask2=mass<11.7
mass_mask=np.logical_and(mass_mask1,mass_mask2)


print(sum(mass_mask))

central_mask=data['upid']==-1
data1=data[central_mask]


data1['mvir']=np.log10(data1['mvir']) - np.log10(h)

    

def Mstellar_meanAndScatter(SM_bins):

    means = []
    scatters = []
    width = SM_bins[1]-SM_bins[0]
    for b in SM_bins[::-1][1:][::-1]:
        df_ = data1.query('mvir>{} & mvir<{}'.format(b, b+width))
        Re = df_['stellar_mass'].values
        print(b,len(Re))
        if len(Re) < 5 :
            means.append(np.nan)
            scatters.append(np.nan)
        elif np.isnan(np.sum(Re))==True:
            print("full of NaN")
            means.append(np.nan)
            scatters.append(np.nan)    
        else:
            ds = DescrStatsW(Re, weights=None)
            print(Re[0:10])
            print(sum(Re))
            perc = ds.quantile([0.16,0.84], return_pandas=False)
            Re = Re[ (Re>perc[0]) & (Re<perc[1])]
            means.append(np.median(Re))
            scatters.append(np.std(Re))
        
    return SM_bins[::-1][1:][::-1]+width/2, np.array(means), np.array(scatters)


Mstar2,mean2,scatter2 = Mstellar_meanAndScatter(SM_bins=np.arange(11,16.0,0.1))


params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)

np.savetxt('Grylls_Halo_Mass_centrals.txt',Mstar2)
np.savetxt('Grylls_stellar_Mass_median_centrals.txt',mean2)
np.savetxt('Grylls_stellar_Mass_scatter_centrals.txt',scatter2)


Mstar2=np.loadtxt('Grylls_Halo_Mass_centrals.txt')
mean2=np.loadtxt('Grylls_stellar_Mass_median_centrals.txt')
scatter2=np.loadtxt('Grylls_stellar_Mass_scatter_centrals.txt')

line2=plt.errorbar(Mstar2,mean2,yerr=scatter2,label="SMHM Relation For Central Haloes and Galaxies",color='red',linewidth=3)

plt.ylabel('$log_{10} M_* [M\odot]$',fontsize=20)
plt.xlabel('$log_{10} M_{halo} [M\odot]$',fontsize=20)
plt.legend(loc='lower right',fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()



