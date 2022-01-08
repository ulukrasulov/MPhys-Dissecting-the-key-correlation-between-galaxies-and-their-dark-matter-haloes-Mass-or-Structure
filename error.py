import numpy as np
from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
from Corrfunc.utils import convert_rp_pi_counts_to_wp
from Corrfunc.utils import convert_3d_counts_to_cf
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('Random_data_mass_11.0-11.2.csv' , delimiter = ",")
data = pd.read_csv('data_mass_11.0-11.2.csv',delimiter=',')
print(df.columns)
print(df.head(10))

print(data.columns)
print(data.head(10))

cz_rand=df['z']
ra_rand=df['ra']
dec_rand=df['dec']
radial_weights_rand=df['radial_weight']




DEC=data['dec']
Z=data['z']*300000
RA=data['ra']
WEIGHTS=data['radial_weight']

sample_size = len(DEC)
ind = np.arange(0,len(RA))

N_bootstrap=1000
print(Z[10])

cosmology = 1

pimax=20
nthreads = 4




Rbins = np.linspace(-1,1.0,50)
rbins = 10**Rbins # for the wp calculation




N=len(RA)
rand_N=3*N

# Auto pair counts in DD
autocorr=1
DD_counts = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, rbins,RA1=RA, DEC1=DEC, CZ1=Z,weights1=WEIGHTS)
#Cross pair counts in DR
autocorr=0
DR_counts = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, rbins,RA1=RA, DEC1=DEC, CZ1=Z,weights1=WEIGHTS,RA2=ra_rand, DEC2=dec_rand, CZ2=cz_rand,weights2=radial_weights_rand)
# Auto pairs counts in RR
autocorr=1
RR_counts = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, rbins,ra_rand, dec_rand, cz_rand,weights2=radial_weights_rand)
# All the pair counts are done, get the angular correlation function

print(rbins)

nbins=len(rbins)-1

clustering1 = convert_rp_pi_counts_to_wp(N, N, rand_N, rand_N,DD_counts, DR_counts,DR_counts, RR_counts, nbins, pimax)


diff=np.diff(rbins)

plt.plot(np.log10(rbins[1:]-diff/2),np.log10(clustering1),color='red',label='original')
#plt.errorbar(np.log10(rbins[1:]-diff/2),(wp),yerr=std, fmt='o',label='mean')
plt.xlabel('log10(rbins)')
plt.ylabel('log10(wp)')
plt.legend(loc='upper right')
plt.show()

"""
clustering=clustering1



for k in range(N_bootstrap):
    indices = np.random.choice(ind, len(RA), replace=True)
    print(indices[0:20])
    z_new = Z[indices]
    ra_new = RA[indices]
    dec_new= DEC[indices]
    weight_new=WEIGHTS[indices]
    print(z_new[0:5])

    cosmology = 1

    pimax=20
    nthreads = 4





    Rbins = np.linspace(-1,1.5,50)
    rbins = 10**Rbins # for the wp calculation


    nbins=len(rbins)-1

    N=len(RA)
    rand_N=3*N

    # Auto pair counts in DD
    autocorr=1
    DD_counts = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, rbins,RA1=ra_new, DEC1=dec_new, CZ1=z_new,weights1=weight_new)
    #Cross pair counts in DR
    autocorr=0
    DR_counts = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, rbins,RA1=ra_new, DEC1=dec_new, CZ1=z_new,weights1=weight_new,RA2=ra_rand, DEC2=dec_rand, CZ2=cz_rand,weights2=radial_weights_rand)
    # Auto pairs counts in RR
    autocorr=1
    RR_counts = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, rbins,ra_rand, dec_rand, cz_rand,weights2=radial_weights_rand)
    # All the pair counts are done, get the angular correlation function


    wp = convert_rp_pi_counts_to_wp(N, N, rand_N, rand_N,DD_counts, DR_counts,DR_counts, RR_counts, nbins, pimax)


    
    clustering=np.vstack((clustering,wp))
    print(clustering)


np.savetxt("10.4-10.6 new_bins_clustering10.txt",clustering)




data=np.loadtxt('10.4-10.6 new_bins_clustering10.txt')
Rbins = np.linspace(-1,1.5,50)
rbins = 10**Rbins # for the wp calculation
print(data.shape)

wp=np.zeros(49)
std=np.zeros(49)
for i in range(49):
    wp[i]=np.mean(np.log10(data[:,i]))
    std[i]=np.std(np.log10(data[:,i]))



print(len(data[0,:]))    
diff=np.diff(rbins)

plt.plot(np.log10(rbins[1:]-diff/2),np.log10(data[0,:]),color='red',label='original')
plt.errorbar(np.log10(rbins[1:]-diff/2),(wp),yerr=std, fmt='o',label='mean')
plt.xlabel('log10(rbins)')
plt.ylabel('log10(wp)')
plt.legend(loc='upper right')
plt.show()





df = pd.DataFrame({"bins" : ((rbins[1:]-diff/2)), "mean_wp" : wp, "error_wp" : std})
df.to_csv("/media/uluk/Seagate Expansion Drive/Dissertation/Viola wp errors/Results/10.4-10.6_with_erros.csv", index=False)
"""

















