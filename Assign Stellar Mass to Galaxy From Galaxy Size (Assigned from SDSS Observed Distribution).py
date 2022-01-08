import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy import integrate
import random

from scipy import interpolate

def Mstar_from_size_central(Re_low,Re_up,total):
    
    data=np.loadtxt("/home/uluk/Desktop/New/Meeting 29.02.21/Distribution data new/data between " + str(Re_low) + " " + str(Re_up) + ".txt")
    
    mass_bins=np.arange(7,14.0,0.1)
    mass_bins=np.delete(mass_bins,0)
    sds=data*mass_bins
    normalisation=sum(data*mass_bins) #normalise histogram
    data=data/normalisation
    #plt.plot(mass_bins,data) #Plot histogram
    #plt.show()
    print(sum(mass_bins*data)) #check its normalised
    cdf=np.zeros(len(data))
    """Calculate the Cumulative Distribtuion function looks like a sigmoid"""
    for i in range(len(data)-1):
        cdf[0]=data[0]
        i=i+1
        cdf[i]=cdf[i-1]+data[i]*mass_bins[i]

    print(cdf)
    #plot cdf
    #plt.plot(mass_bins, cdf, 'ro')
    #plt.show()



    """Create a continous CDF function"""

    f = interpolate.interp1d(mass_bins,cdf) 

    """Interpolate the reverse CDF function i.e. pick y value 0 to 1 and select corresponding mass"""

    reverse=interpolate.interp1d(cdf,mass_bins) 


    y_new = f(mass_bins)   # use interpolation function returned by `interp1d`

    #plt.plot(mass_bins, cdf, 'o', mass_bins, y_new, '-') #Plot continius cdf and discrete one comare

    #plt.show()

    y_reverse=reverse(cdf)#Inverse CDF function Mass on y axis

    #plt.plot(cdf,mass_bins,'o',cdf,y_reverse, '-')#Plot Inverse continious cdf and discrete one comare

    #plt.show()

    masses=np.zeros(total)

    """Random number generattor uniform 0-1 assign masses from the inverse CDF function. Should reproduce the original data"""
    for i in range(total): 
        
        x=random.random()
        mass=reverse(x)
        masses[i]=mass

    print(masses)


    histogram=np.histogram(masses,np.arange(7,14.0,0.1))
    print(histogram[0])
    print(histogram[0].shape)
    histogram_values=np.asarray(histogram[0])  
    normalisation1=sum(histogram_values*mass_bins)

    #histogram_values=histogram_values/normalisation1
    """Plot original data and the reproduced mass distribtuion to see if it all works"""
    #line1=plt.plot(mass_bins,histogram_values,label="new distribution",color='red')
    #line2=plt.plot(mass_bins,data,label='original data',lw=2,color='blue',alpha=0.5)
    #plt.legend(loc='upper right')
    #plt.show()
    return masses



def Mstar_from_size_satellites(Re_low,Re_up,total):
    
    data=np.loadtxt("/home/uluk/Desktop/New/Meeting 29.02.21/Satellite Distribution data/data between " + str(Re_low) + " " + str(Re_up) + ".txt")
    
    mass_bins=np.arange(7,14.0,0.1)
    mass_bins=np.delete(mass_bins,0)
    sds=data*mass_bins
    normalisation=sum(data*mass_bins) #normalise histogram
    data=data/normalisation
    #plt.plot(mass_bins,data) #Plot histogram
    #plt.show()
    print(sum(mass_bins*data)) #check its normalised
    cdf=np.zeros(len(data))
    """Calculate the Cumulative Distribtuion function looks like a sigmoid"""
    for i in range(len(data)-1):
        cdf[0]=data[0]
        i=i+1
        cdf[i]=cdf[i-1]+data[i]*mass_bins[i]

    print(cdf)
    #plot cdf
    #plt.plot(mass_bins, cdf, 'ro')
    #plt.show()



    """Create a continous CDF function"""

    f = interpolate.interp1d(mass_bins,cdf) 

    """Interpolate the reverse CDF function i.e. pick y value 0 to 1 and select corresponding mass"""

    reverse=interpolate.interp1d(cdf,mass_bins) 


    y_new = f(mass_bins)   # use interpolation function returned by `interp1d`

    #plt.plot(mass_bins, cdf, 'o', mass_bins, y_new, '-') #Plot continius cdf and discrete one comare

    #plt.show()

    y_reverse=reverse(cdf)#Inverse CDF function Mass on y axis

    #plt.plot(cdf,mass_bins,'o',cdf,y_reverse, '-')#Plot Inverse continious cdf and discrete one comare

    #plt.show()

    masses=np.zeros(total)

    """Random number generattor uniform 0-1 assign masses from the inverse CDF function. Should reproduce the original data"""
    for i in range(total): 
        
        x=random.random()
        mass=reverse(x)
        masses[i]=mass

    print(masses)


    histogram=np.histogram(masses,np.arange(7,14.0,0.1))
    print(histogram[0])
    print(histogram[0].shape)
    histogram_values=np.asarray(histogram[0])  
    normalisation1=sum(histogram_values*mass_bins)

    histogram_values=histogram_values/normalisation1
    """Plot original data and the reproduced mass distribtuion to see if it all works"""
    #line1=plt.plot(mass_bins,histogram_values,label="new distribution",color='red')
    #line2=plt.plot(mass_bins,data,label='original data',lw=2,color='blue',alpha=0.5)
    #plt.legend(loc='upper right')
    #plt.show()
    return masses

