# Imports - Generic
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz, trapz
from scipy.special import erfc

# Imports - Astro
from colossus.cosmology import cosmology
from colossus.lss import mass_function

cosmo = cosmology.setCosmology("planck18")

class VPEAKSM_Generator:
    def __init__(self):
        # starting things
        pass

    def Scatter(self, hm):
        return 0.1

    def HMF(self, z, bins = 0.05, cum_var=1.0):
        """ Function to generate the Halo Mass function.
        Params:
            z (float), redshift [dimensionless]
        returns:
            (np array), the mass range log10 [Msun]
            (np array), the hmf itself log10 [mpc^-1 dex^-1]
        """
        width = bins
        M = np.arange(9., 15.5+width, width) # Sets the HMF mass range
        phi = mass_function.massFunction((10.0**M)*cosmo.h, z , mdef = '200m' , model = 'tinker08' , q_out="dndlnM") * np.log(10) * cosmo.h**3.0 * cum_var # Produces the Tinker et al. (2008) HM

        if True:
            a = 1./(1.+z) #Constants for Behroozi et al. (2013) Appendix G, eqs. G6, G7 and G8 subhalos correction
            C = np.power(10., -2.415 + 11.68*a - 28.88*a**2 + 29.33*a**3 - 10.56*a**4)
            logMcutoff = 10.94 + 8.34*a - 0.36*a**2 - 5.08*a**3 + 0.75*a**4
            correction = phi * C * (logMcutoff - M)
            return M , np.log10(phi + correction)

        return M, np.log10(phi)

    def derivative(self, x, y):
        """ Returns the derivative of any input function
        """
        func = interp1d(x, y, fill_value="extrapolate")
        dx = 0.1
        x_calc = np.arange(x[0],x[-1]+dx,dx)
        y_calc = func(x_calc)
        dydx = np.diff(y_calc)/np.diff(x_calc)

        dydx = np.append(dydx, dydx[-1]) #Preventing boundary abnormalities in the returned function
        dydx_low = np.mean(dydx[:10])
        dydx[0] = dydx_low
        dydx[1] = dydx_low
        dydx_high = np.mean(dydx[-10:])

        return interp1d(x_calc, dydx, fill_value=(dydx_low, dydx_high), bounds_error = False)

    def integrals(self, SMF, HMF, sig):
        """ Function to return the integrals of the stellar mass function (SMF) and HMF
            as calculated in Aversa et al. (2015) eq. 37
        Params:
            - SMF ((2, N) np array), The stellar mass function, where the first element is the mass bins,
            the second is the SMF itself
            - HMF ((2, N), np array), The Halo Mass function, formatted as the SMF.
            - sig (float or array like), the scatter, sigma. This is
        """

        # Extract the respective arrays from HMF and SMF
        M_s , phi_s = SMF[0] , SMF[1]
        M_h , phi_h = HMF[0] , HMF[1]
        phi_s , phi_h = 10.0**phi_s , 10.0**phi_h # No longer in log10

        I_phi_s = np.flip(cumtrapz(np.flip(phi_s), M_s)) # Integral of SMF

        I_phi_h = np.array([]) # Empty
        for i, m in enumerate(M_h): # Iterate through Mhalo

            I = np.trapz(phi_h*0.5*erfc((m-M_h)/(np.sqrt(2)*sig)) , M_h) # Some fancy formula.

            I_phi_h = np.append(I_phi_h,I)

        M_s , M_h = M_s[:-1]+0.025 , M_h+0.025

        return I_phi_s, I_phi_h , M_s , M_h

    def SMFfromSMHM(self, M_s, vmax_h, sig_0, z):
        """ Reconstructs the SMF from the Stellar Mass Halo mass relationship
        params:
            M_s
            M_h
            sig_0
            z
        """
        bins = 0.1
        volume = (500*cosmo.h)**3 # Need to check if this is okay

        #M_hmf, phi = self.HMF(z, bins, cum_var = volume*bins) # Generate HMF

        #self.vmax_h
        #self.phi_h

        #M_hmf = M_hmf[20:] # Cutting the low mass end of the HMF for increased speed
        #phi = 10**phi[20:]

        phi = np.log10( (10**self.phi_h)*(volume*bins) )

        cum_phi = np.cumsum( phi )
        max_number = np.floor(np.max(cum_phi))

        print("Max_number:", max_number)

        if (np.random.uniform(0, 1) > np.max(cum_phi)-max_number): # Calculating number of halos to compute
            max_number += 1

        int_cum_phi = interp1d(cum_phi, self.Mpeak_h)
        range_numbers = np.random.uniform(np.min(cum_phi), np.max(cum_phi), int(max_number))
        halo_masses = int_cum_phi(range_numbers)

        M_smf = np.arange(8.5, 12.2 , 0.1) #SMF histogram bins
        SMHMinterp = interp1d(vmax_h , M_s , fill_value="extrapolate")
        stellar_masses = SMHMinterp(halo_masses) + np.random.normal(0., self.Scatter(halo_masses), halo_masses.size) # Calculating stellar masses using SMHM with scatter
        phi_smf = np.histogram(stellar_masses , bins = M_smf)[0]/0.1/volume #SMF histogram

        return M_smf[:-1]+0.05 , np.log10(phi_smf)

    def GetSMHM(self):

        SMF = np.array([[9.054204518687811, -2.1774689491408887],
                    [9.255406446355702, -2.2668877840095236],
                    [9.456659303162063, -2.309248094931914],
                    [9.654646328964038, -2.3692362539072196],
                    [9.852658819335248, -2.405695150909403],
                    [10.053930774568535, -2.4304085153519517],
                    [10.255240926655675, -2.419827986834817],
                    [10.456506515746653, -2.4504236667706465],
                    [10.654474443121702, -2.528058772225794],
                    [10.852365976789045, -2.6762816636003075],
                    [11.056674581903604, -2.89513053774804],
                    [11.254406962013228, -3.190411316454568],
                    [11.45530968099261, -3.556298979507388],
                    [11.652812879979118, -4.063343115972014],
                    [11.853441854839222, -4.682170345235896],
                    [12.053892577714683, -5.4657024083116355],
                    [12.250625473481835, -6.68450671946321]])

        z = 0.1
        Mstar_s, phi_s = SMF[:,0] , SMF[:,1]

        abun = np.loadtxt("ExampleMpeakSM.txt")

        Mpeak_h = np.array([10.54591837, 10.6377551,  10.72959184, 10.82142857, 10.91326531, 11.00510204,
                                 11.09693878, 11.18877551, 11.28061224, 11.37244898, 11.46428571, 11.55612245,
                                 11.64795918, 11.73979592, 11.83163265, 11.92346939, 12.01530612, 12.10714286,
                                 12.19897959, 12.29081633, 12.38265306, 12.4744898,  12.56632653, 12.65816327,
                                 12.75,       12.84183673, 12.93367347, 13.0255102,  13.11734694, 13.20918367,
                                 13.30102041, 13.39285714, 13.48469388, 13.57653061, 13.66836735, 13.76020408,
                                 13.85204082, 13.94387755, 14.03571429, 14.12755102, 14.21938776, 14.31122449,
                                 14.40306122, 14.49489796, 14.58673469, 14.67857143, 14.77040816, 14.8622449,
                                 14.95408163] )

        phi_h = np.array(   [-1.64569902, -1.36365886, -1.33626109, -1.34874622, -1.5084095,  -1.51250842,
                             -1.60379076, -1.67710181, -1.76842011, -1.84906203, -1.91653181, -2.00576751,
                             -2.07859653, -2.170284,   -2.24726163, -2.33655068, -2.41156324, -2.49539311,
                             -2.5785239,  -2.65962856, -2.74160643, -2.82539033, -2.90935057, -2.99264816,
                             -3.07517799, -3.16118488, -3.24481785, -3.32873704, -3.41653127, -3.5057285,
                             -3.59352738, -3.68548197, -3.77905903, -3.86933119, -3.97037248, -4.07184525,
                             -4.17943214, -4.27603782, -4.39110549, -4.51207324, -4.62978577, -4.74930429,
                             -4.89189413, -5.0199272,  -5.17711354, -5.35941545, -5.53802729, -5.73882319,
                             -5.94439055] )

        #phi_h = np.log10(10**phi_h * 0.7**3.0) # Not sure if we need this or not.

        #Mpeak_h, phi_h = self.HMF(z)

        self.Mpeak_h = Mpeak_h
        self.phi_h = phi_h

        deri = self.derivative(abun[:,0], abun[:,1])(Mpeak_h)
        n = 0
        e = 1.
        while n < 3:
            # Calls the integrals function and matches integral values of SMF to interpolated values in the integral of HMF, iterating multiple times
            I_phi_s , I_phi_h , Mstar_s_temp, Mpeak_h_temp = self.integrals(np.array([Mstar_s , phi_s]),
                                                                           np.array([Mpeak_h , phi_h]),
                                                                           self.Scatter(Mpeak_h)/deri)
            int_I_phi_h = interp1d(I_phi_h, Mpeak_h_temp, fill_value="extrapolate")
            Mpeak_h_match = np.array([])
            for m in range(Mstar_s_temp.size):
                Mpeak_h_match = np.append(Mpeak_h_match , int_I_phi_h(I_phi_s[m]))

            Mstar_s_iter, phi_s_iter = self.SMFfromSMHM(Mstar_s_temp, Mpeak_h_match, self.Scatter(Mpeak_h_match), z) # Reconstructing SMF
            int_phi_s_iter = interp1d(Mstar_s_iter , phi_s_iter, fill_value="extrapolate")
            e_temp = max((phi_s - int_phi_s_iter(Mstar_s))/phi_s) # Calculating relative error between reconstructed SMF and the input SMF

            if e_temp < e:
                # Only accepts iterations that decrease the relative error
                e = e_temp
                deri = self.derivative(Mpeak_h_match, Mstar_s_temp)(Mpeak_h)
            n += 1

        self.mstar = Mstar_s_temp
        self.mhalo = Mpeak_h_match

    def HMSM(self, HM, scatter = False):

        func = interp1d(self.mhalo, self.mstar, bounds_error = False, fill_value="extrapolate")
        sm = func(HM)
        if scatter:
            scat = np.random.normal(scale=self.Scatter(HM), size=np.shape(sm))
            sm += scat
        return sm
