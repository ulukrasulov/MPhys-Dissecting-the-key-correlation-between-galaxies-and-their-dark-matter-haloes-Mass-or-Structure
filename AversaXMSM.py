# Chris's (much improved!) verison of the code written by Max and Hao that
# generates the XMSM relation from HMFS

# Imports - Generic
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz, trapz
from scipy.special import erfc

# Imports - Astro
from colossus.cosmology import cosmology
from colossus.lss import mass_function

# Set cosmology. We need to be careful how we use this.
cosmo = cosmology.setCosmology("planck18")

# Note that we sometimes use mhalo or mpeak as a variable name (even though
# this will be done with other variables). If anyone feels like refactoring
# this, crack on! It should work regardless.

# Let's do this.
class AversaXMSM:
    def __init__(self, z = 0.1, XMF = "Default", SMF="Default", scatterFunc=None):
        """ Class to derive, store and calculate the X to SM (where X is some property),
        e.g. halo mass, peak mass etc. The constructor will calculate the best X2SM
        relation given the supplied variables.

        Params:
            z (float), the redshift in question [dimensionless].
            XMF (nx2 np array), the function of the variable in question. The first
                column is the property bins, the second is phi. Default uses Mpeak
                from MultiDark at z=0.
            SMF (nx2 np array), the stellar mass function, formatted as XMF.
                Default is the SMF from the SDSS, for centrals.
            scatterFunc (pointer), an function that takes a single
                variable (X), which returns the value of the scatter at that variable.
                This is so the user can override the default behavior, which is
                a scatter of 0.1, independent of X.
        """
        ########################################################################
        # Part one - Variable management
        ########################################################################

        self.z = z

        # + XMF +

        if str(XMF) == "Default":
            # This is the MPeak function derived from the MultiDark at z=0
            self.Mpeak_h = np.array([10.54591837, 10.6377551,  10.72959184, 10.82142857, 10.91326531, 11.00510204,
                                     11.09693878, 11.18877551, 11.28061224, 11.37244898, 11.46428571, 11.55612245,
                                     11.64795918, 11.73979592, 11.83163265, 11.92346939, 12.01530612, 12.10714286,
                                     12.19897959, 12.29081633, 12.38265306, 12.4744898,  12.56632653, 12.65816327,
                                     12.75,       12.84183673, 12.93367347, 13.0255102,  13.11734694, 13.20918367,
                                     13.30102041, 13.39285714, 13.48469388, 13.57653061, 13.66836735, 13.76020408,
                                     13.85204082, 13.94387755, 14.03571429, 14.12755102, 14.21938776, 14.31122449,
                                     14.40306122, 14.49489796, 14.58673469, 14.67857143, 14.77040816, 14.8622449,
                                     14.95408163] )
            self.phi_h = np.array([-1.64569902, -1.36365886, -1.33626109, -1.34874622, -1.5084095,  -1.51250842,
                                     -1.60379076, -1.67710181, -1.76842011, -1.84906203, -1.91653181, -2.00576751,
                                     -2.07859653, -2.170284,   -2.24726163, -2.33655068, -2.41156324, -2.49539311,
                                     -2.5785239,  -2.65962856, -2.74160643, -2.82539033, -2.90935057, -2.99264816,
                                     -3.07517799, -3.16118488, -3.24481785, -3.32873704, -3.41653127, -3.5057285,
                                     -3.59352738, -3.68548197, -3.77905903, -3.86933119, -3.97037248, -4.07184525,
                                     -4.17943214, -4.27603782, -4.39110549, -4.51207324, -4.62978577, -4.74930429,
                                     -4.89189413, -5.0199272,  -5.17711354, -5.35941545, -5.53802729, -5.73882319,
                                     -5.94439055] )
        else:
            self.Mpeak_h, self.phi_h = XMF[:, 0], XMF[:, 1]

        # + SMF +

        if str(SMF) == "Default":
            # Bernardi's SMF. Should probs read this in from a file.
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

        self.Mstar_s, self.phi_s = SMF[:,0], SMF[:,1]

        #  + Scatter +
        # This lets you externally assign your own scatter function if you want.
        if scatterFunc is not None:
            self.Scatter = scatterFunc
        else:
            self.Scatter = self.InternalScatter

        ########################################################################
        # Part 2 - Generating the initial iteration
        ########################################################################

        self.GenerateXMSM(1.0) # Generate this with no derivative.
        # This assigns an initial guess to the self.mhalo, self.mstar buffers.

        ########################################################################
        # Part 3 - Generating the final iteration
        ########################################################################

        # First we need the derivative.
        deri = self.derivative(self.mhalo, self.mstar)(self.Mpeak_h)

        self.GenerateXMSM(deri)


    def InternalScatter(self, var, z):
        return np.ones_like(var)*0.1

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
        M_s, phi_s = SMF[0], SMF[1]
        M_h, phi_h = HMF[0], HMF[1]
        phi_s, phi_h = 10.0**phi_s, 10.0**phi_h

        I_phi_s = np.flip(cumtrapz(np.flip(phi_s), M_s)) # Integral of SMF

        I_phi_h = np.array([])
        for i, m in enumerate(M_h): # Iterate through Mhalo
            I = np.trapz(phi_h*0.5*erfc((m-M_h)/(np.sqrt(2)*sig)) , M_h) # Some fancy formula.
            I_phi_h = np.append(I_phi_h,I)

        M_s , M_h = M_s[:-1]+0.025 , M_h+0.025
        return I_phi_s, I_phi_h, M_s, M_h

    def SMFfromSMHM(self, M_s, vmax_h, sig_0, z):
        """ Reconstructs the SMF from the Stellar Mass Halo mass relationship
        """
        bins = 0.1
        volume = (500/cosmo.h)**3 # 'sufficently large' volume

        phi = np.log10( (10**self.phi_h)*(volume*bins) )

        cum_phi = np.cumsum( phi )
        max_number = np.floor(np.max(cum_phi))

        if (np.random.uniform(0, 1) > np.max(cum_phi)-max_number): # Calculating number of halos to compute
            max_number += 1

        int_cum_phi = interp1d(cum_phi, self.Mpeak_h)
        range_numbers = np.random.uniform(np.min(cum_phi), np.max(cum_phi), int(max_number))
        halo_masses = int_cum_phi(range_numbers)

        M_smf = np.arange(8.5, 12.2 , 0.1) #SMF histogram bins
        SMHMinterp = interp1d(vmax_h , M_s , fill_value="extrapolate")
        stellar_masses = SMHMinterp(halo_masses) + np.random.normal(0., self.Scatter(halo_masses, z), halo_masses.size) # Calculating stellar masses using SMHM with scatter
        phi_smf = np.histogram(stellar_masses , bins = M_smf)[0]/0.1/volume #SMF histogram

        return M_smf[:-1]+0.05 , np.log10(phi_smf)

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

    def GenerateXMSM(self, deri):
        """ Function doing most of the 'heavy lifting'. This is run twice - first
            to get an initial 'guess' and then to work out the actual result.
        """

        n, e = 0, 1.
        while n < 3:
            # Calls the integrals function and matches integral values of SMF
            # to interpolated values in the integral of HMF, iterating multiple times
            I_phi_s, I_phi_h, Mstar_s_temp, Mpeak_h_temp = self.integrals( np.array([self.Mstar_s, self.phi_s]),
                                                                           np.array([self.Mpeak_h , self.phi_h]),
                                                                           self.Scatter(self.Mpeak_h, self.z)/deri)
            int_I_phi_h = interp1d(I_phi_h, Mpeak_h_temp, fill_value="extrapolate")
            Mpeak_h_match = np.array([])
            for m in range(Mstar_s_temp.size):
                Mpeak_h_match = np.append(Mpeak_h_match , int_I_phi_h(I_phi_s[m]))

            Mstar_s_iter, phi_s_iter = self.SMFfromSMHM(Mstar_s_temp, Mpeak_h_match, self.Scatter(Mpeak_h_match, self.z), self.z) # Reconstructing SMF
            int_phi_s_iter = interp1d(Mstar_s_iter , phi_s_iter, fill_value="extrapolate")
            e_temp = max((self.phi_s - int_phi_s_iter(self.Mstar_s))/self.phi_s) # Calculating relative error between reconstructed SMF and the input SMF

            if e_temp < e:
                # Only accepts iterations that decrease the relative error
                e = e_temp
                deri = self.derivative(Mpeak_h_match, Mstar_s_temp)(self.Mpeak_h)
            n += 1

        self.mstar = Mstar_s_temp
        self.mhalo = Mpeak_h_match

    def X2SM(self, HM, scatter = True, z = 0.0):
        func = interp1d(self.mhalo, self.mstar, bounds_error = False, fill_value="extrapolate")
        sm = func(HM)
        if scatter:
            scat = np.random.normal(scale=self.Scatter(HM, z), size=np.shape(sm))
            sm += scat
        return sm

if __name__ == "__main__":
    Obj = AversaXMSM()
    Obj.X2SM(12.0)
