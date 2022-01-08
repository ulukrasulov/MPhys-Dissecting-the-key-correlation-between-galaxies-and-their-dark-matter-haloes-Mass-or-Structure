# Get SDSS Clustering

import numpy as np
from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
from Corrfunc.utils import convert_rp_pi_counts_to_wp
from Corrfunc.utils import convert_3d_counts_to_cf
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def getSDSSClustering(actual_filename, random_filename,
                      Rbins = np.linspace(-1, 1.0, 50),
             path="/media/uluk/Seagate Expansion Drive/Dissertation/Viola wp errors/",
                   store_directory="Processed_Data/", pimax = 20, centrals=False):

    rbins = 10**Rbins # for the wp calculation
    bins = Rbins

    store_name = actual_filename[:-4] + "_" +  random_filename[:-4] \
        + "_" + str(Rbins[0]) + "_" + str(Rbins[-1]) + "_" + str(len(Rbins)) + ".csv"

    print("Attempting to read data from file")
    try:
        final_data = np.loadtxt(path + store_directory + store_name)
        print("Data Found")
    except OSError as e:
        print("Data not found, computing. This will take some time.")
        # Input files

        print("Reading File (randoms): {}".format(random_filename))
        print("Reading File (SDSS): {}".format(actual_filename))

        random_df = pd.read_csv(path + random_filename, sep = "\s+|,|\s+,|,\s+", engine='python') #delim_whitespace=True)
        data_df = pd.read_csv(path + actual_filename, delimiter=',')

        print("Random Fields:", list(random_df), "File:", random_filename)
        print("Data Fields:", list(data_df), "File:", actual_filename)

        # Random Data
        cz_rand = random_df['z']

        if centrals:
            cz_rand *= 300000

        ra_rand = random_df['ra']
        dec_rand = random_df['dec']
        radial_weights_rand = random_df['radial_weight']

        # Input Data
        dec = data_df['dec']
        z = data_df['z']*300000
        ra = data_df['ra']
        weights = data_df['radial_weight']

        #sample_size = len(DEC)
        ind = np.arange(0, len(ra))

        N_bootstrap = 10

        cosmology = 2
        nthreads = 8

        print("Computing Auto pair counts in DD")

        # Auto pair counts in DD
        autocorr = 1
        DD_counts = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, 10**bins,
                                 RA1=ra, DEC1=dec, CZ1=z, weights1=weights, verbose=True)

        print("Computing Cross pair counts in DR")

        # Cross pair counts in DR
        autocorr = 0
        DR_counts = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, 10**bins,
                         RA1=ra, DEC1=dec, CZ1=z, weights1=weights,
                         RA2 = ra_rand, DEC2 = dec_rand, CZ2 = cz_rand, weights2 = radial_weights_rand, verbose=True)


        print("Computing Pair counts in RR")
        # Auto pairs counts in RR
        autocorr = 1
        RR_counts = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, 10**bins, ra_rand, dec_rand, cz_rand,
                         weights1=radial_weights_rand, verbose=True)

        N = len(ra)
        rand_N = 3*N

        clustering = convert_rp_pi_counts_to_wp(N, N, rand_N, rand_N, DD_counts, DR_counts,
                                DR_counts, RR_counts, len(bins)-1, pimax)


        for k in tqdm(range(N_bootstrap)):
            # Subsample
            indices = np.random.choice(ind, len(ra), replace=True)
            z_new = z[indices]
            ra_new = ra[indices]
            dec_new = dec[indices]
            weight_new=weights[indices]

            # Auto pair counts in DD
            autocorr = 1
            DD_counts = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, rbins,
                RA1 = ra_new, DEC1 = dec_new, CZ1 = z_new, weights1 = weight_new)
            #Cross pair counts in DR
            autocorr = 0
            DR_counts = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, rbins,
                RA1 = ra_new, DEC1 = dec_new, CZ1 = z_new, weights1 = weight_new,
                RA2 = ra_rand, DEC2 = dec_rand, CZ2 = cz_rand, weights2 = radial_weights_rand)
            # Auto pairs counts in RR
            autocorr = 1
            RR_counts = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, rbins,
                ra_rand, dec_rand, cz_rand,weights2 = radial_weights_rand)

            wp = convert_rp_pi_counts_to_wp(N, N, rand_N, rand_N, DD_counts, DR_counts,
                DR_counts, RR_counts, len(bins)-1, pimax)

            clustering=np.vstack((clustering, wp))

            final_data = clustering

        np.savetxt(path + store_directory + store_name, clustering)

    wp = np.log10(np.mean(final_data, axis = 0))
    std = np.std(final_data, axis = 0)
    std = np.log10(10**wp + std) - wp

    spacing = Rbins[0]-Rbins[1]
    
    plt.errorbar(Rbins[:-1] + spacing/2,(wp),yerr=std, fmt='o',label='11.0-11.2 Bootstrap=10')

    plt.ylabel('log10(wp)')

    plt.legend(loc='upper right')

    plt.show()


    return wp, std, Rbins[:-1] + spacing/2





if __name__ == "__main__":
    wp, std, Rbins = getSDSSClustering("data_mass_11.0-11.2.csv", "Random_data_mass_11.0-11.2.csv",
                          Rbins = np.linspace(-1, 1.0, 50),
                          path="/media/uluk/Seagate Expansion Drive/Dissertation/Viola wp errors/",
                          store_directory="Processed_Data/", pimax = 20)
