import batman
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import corner
import emcee

from copy import deepcopy
from isochrones.mist.bc import MISTBolometricCorrectionGrid
from isochrones import get_ichrone
import time
import scipy
from scipy import optimize
import radvel
import radvel.likelihood
from astroquery.mast import Catalogs
import sys

################################################################

# This is a global model to fit the RV, SED, and light curve   #
# data for TOI-3261. It will save the chains, which can then   #
# be plotted using another script.                             #
# This fit assumes no TTVs.                                    #

################################################################

#turn off automatic parallelization to avoid problems with later parallelization method
os.environ["OMP_NUM_THREADS"] = "1"

################################################
##### initializing paths to required files #####
################################################

filepath = '/home/u1153471/toi3261/scripts/'             #folder containing initialization files
out_folder = '/home/u8015661/emma/toi3261/fitresults/'   #folder to store fit results

priorcsv = 'toi3261_prior.csv'         #name of prior csv

#loading in data from csv file
data = pd.read_csv(filepath+'toi3261_detrended_lc',comment='#', header=0) #reading in light curve file

obj_id = "TIC 358070912"

t = np.array(data["time"]) + 2457000
y = np.array(data["flux"])
yerr = np.array(data["err"]) #error

        
####################################
######### Defining Classes #########
####################################

class Params:
    def __init__(self):
         self.data = []

    def add(self, x):           #function for adding free parameters to a list
        self.data.append(x)

    def unpack(self):           #unpacking list of free parameter names (to be used in theta) -> duplicates from shared parameters removed
        return [x.name for x in self.data]
    
    def __print__(self):
        return '%s' % (self.data)    #%s = print as string, %f = print as float, %d = print as integer

    def __repr__(self):
        return '%s' % (self.data)    #%s = print as string, %f = print as float, %d = print as integer

    def __getitem__(self, item):
         return self.data[item]
    def initialize_theta(self):
        theta_test = []
        pos = []
        spr = []

        for x in self.data:
            initial_guess = f'{x.name}_test'       #this will make a variable called p1_test, p2_test, etc.
            locals()[initial_guess] = x.value      #p1 = number, p2 = number, ...

            theta_test.append(locals()[initial_guess])  #appends to initial theta list
        
            nwalkers = (len(self.unpack())*2) + 10
            ndim = len(self.unpack())
            
            if x.prior.priortype == 'gaussian':
                #spread = 0.1 * x.err                                         #10% of 1-sigma width
                spread = float(x.prior.spread) 
                
            elif x.prior.priortype == 'box':
                #spread = 0.2 * (x.prior.upper_bound - x.prior.lower_bound)   #20% of box spread
                spread = float(x.prior.spread)
            
            position = locals()[initial_guess] + spread * np.random.randn(nwalkers)      #number of walkers, number of free parameters
             
            while ((position < x.prior.lower_bound).any() == True) or ((position > x.prior.upper_bound).any() == True):
                position[position < x.prior.lower_bound] = locals()[initial_guess] + spread * np.random.randn()
                position[position > x.prior.upper_bound] = locals()[initial_guess] + spread * np.random.randn()
            
            pos.append(position)

        return theta_test, nwalkers, ndim, pos

class Variable:
    def __init__(self, name, objno, value, err=0, prior='none', flag='fixed'):
        self.name = name
        self.objno = objno        #object it belongs to -> 0 = star, 1 = planet 1, etc.
        self.value = value
        self.err = err
        self.flag = flag
        self.prior = prior

    def apply_prior(self,x):
        return self.prior(x)

    def __print__(self):
        #return "teststring"
        return '%s:%f:%f:%f:%f:%f:%s' %(self.name, self.value, self.err,self.prior.lower_bound,self.prior.center_value,self.prior.upper_bound, self.flag)    #%s = print as string, %f = print as float, %d = print as integer
    def __repr__(self):
        return '%s:%f:%f:%f:%f:%f:%s' %(self.name, self.value, self.err,self.prior.lower_bound,self.prior.center_value,self.prior.upper_bound, self.flag)    #%s = print as string, %f = print as float, %d = print as integer

class TransitTime:
    def __init__(self, name, objno, value, epoch, err=0, prior='none', flag='fixed'):
        self.name = name
        self.objno = objno        #object it belongs to -> 0 = star, 1 = planet 1, etc.
        self.value = value
        self.err = err
        self.flag = flag
        self.prior = prior
        self.epoch = epoch

    def apply_prior(self,x):
        return self.prior(x)

    def __print__(self):
        #return "teststring"
        return '%s:%f:%f:%f:%f:%f:%f:%s' %(self.name, self.value, self.err,self.epoch,self.prior.lower_bound,self.prior.center_value,self.prior.upper_bound, self.flag)    #%s = print as string, %f = print as float, %d = print as integer
    def __repr__(self):
        return '%s:%f:%f:%f:%f:%f:%f:%s' %(self.name, self.value, self.err,self.epoch,self.prior.lower_bound,self.prior.center_value,self.prior.upper_bound, self.flag)    #%s = print as string, %f = print as float, %d = print as integer

class PriorFunction:
    def __init__(self, priortype, center_value= 'none', error = 'none', upper_bound = 'none', lower_bound = 'none',spread='none'):
        self.priortype = priortype
        self.center_value = center_value
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.spread = spread
        self.error = error

    def __call__(self, x):           #function for calling different types of priors
        if self.priortype == 'box':
            if x > self.upper_bound:
                return -np.inf
            elif x < self.lower_bound:
                return -np.inf
            else:
                return 0
        if self.priortype == 'gaussian':
            return float(-.5 * (x - self.center_value)**2/(self.error)**2)
        if self.priortype == 'none':
            return 0

##########################################################
############# Initializing Stellar Fitting ###############
##########################################################

#Querying photometry/astrometry from TIC and Gaia via MAST
photometry = Catalogs.query_object(obj_id,catalog="Tic")
#gaia_info = Catalogs.query_object(obj_id,catalog="Gaia")


G = 6.67e-8
msun = 1.989e33
rsun = 6.96e10
lsun = 3.828e33
sigma = 5.67e-5

Jmag = photometry['Jmag'][0]
Jmagerr = photometry['e_Jmag'][0]

Hmag = photometry['Hmag'][0]
Hmagerr = photometry['e_Hmag'][0]

Kmag = photometry['Kmag'][0]
Kmagerr = photometry['e_Kmag'][0]

Gmag = 13.046638
Gmagerr = .03

BPmag = 13.509357
BPmagerr = .03                       #use error floor of .03 for gaia bands

RPmag = 12.427166
RPmagerr = .03

W1mag = photometry['w1mag'][0]
W1magerr = photometry['e_w1mag'][0]

W2mag = photometry['w2mag'][0]
W2magerr = photometry['e_w2mag'][0]

bc_grid = MISTBolometricCorrectionGrid(['J', 'H', 'K', 'G', 'BP', 'RP', 'WISE_W1', 'WISE_W2'])
mist = get_ichrone('mist')

##########################################################
################## Defining Functions ####################
##########################################################

def batman_model(input_dict, star_dict, time):
    
    aAU = star_dict['mstar'].value**(1/3)*(input_dict['p'].value/365.25)**(2/3)  #a in AU
    aors = aAU*215/star_dict['rstar'].value                  #a over r_star
    inc = 180/np.pi*np.arccos(input_dict['b'].value/aors)
 
    sqrte_cosw = input_dict['sqrte_cosw'].value
    sqrte_sinw = input_dict['sqrte_sinw'].value
    ecc = sqrte_cosw**2 +sqrte_sinw**2
    
    
    u1 = 2*np.sqrt(star_dict['q1'].value)*star_dict['q2'].value
    u2 = np.sqrt(star_dict['q1'].value)*(1 - (2 * star_dict['q2'].value))

    #if e > 1, batman returns nans
    if (ecc > 0.8):
        flux = np.zeros(len(time))
        flux[:] = np.nan
        return flux
    else:
        w_rad = np.arctan2(sqrte_sinw,sqrte_cosw)
        w = w_rad * 180 / np.pi

        params = batman.TransitParams()
        params.t0 = input_dict['t0'].value                          #time of inferior conjunction
        params.per = input_dict['p'].value                          #orbital period
        params.rp = input_dict['rprs'].value                        #planet radius (in units of stellar radii)
        params.a = aors                                             #semi-major axis (in units of stellar radii)
        params.inc = inc                                            #orbital inclination (in degrees)
        params.ecc = ecc                                            #eccentricity
        params.w = w                                                #longitude of periastron (in degrees)
        params.u = [u1,u2]                                          #limb darkening coefficients [u1, u2]
        params.limb_dark = "quadratic"                              #limb darkening model
        
        if np.min(np.abs(np.ediff1d(time))) > 0.0007:                        #supersample for long cadence data - if it's above 1 minute, then it's long cadence
            m = batman.TransitModel(params,time,supersample_factor=7,exp_time=.001389) #for tess
        else:
            m = batman.TransitModel(params, time)                   #initializes model
        flux = m.light_curve(params)
        return flux

#### MIST Isochrone Fitting ####

def sed_likelihood(params):
    age = params['age_Gyr'].value*1.e9
    
    logg = np.log10(G*params['mstar'].value*msun/(params['rstar'].value*rsun)**2.)
    
    try:
        eep = mist.get_eep(params['mstar'].value, np.log10(age), params['feh'].value, accurate =True)
    except RuntimeError:
        return -np.inf
    except ValueError:
        return -np.inf
    mist_teff, mist_logg, mist_rad = mist.interp_value([eep, np.log10(age), params['feh'].value], ['Teff', 'logg', 'radius'])
    
    if mist_rad<0.5 or mist_rad>2.5: # *** Needs to be changed later ***
        return -np.inf
    
    teff = params['teff'].value
    rad = params['rstar'].value
    par = params['parallax'].value
    av = params['av'].value
    errscale = params['errscale'].value
    feh = params['feh'].value
    
    chi2_mist = np.log(2*np.pi*(0.03*mist_teff)**2.) + np.log(2*np.pi*(0.03*mist_rad)**2.)+np.log(2*np.pi*(0.03*mist_logg)**2.)+(teff-mist_teff)**2./2./(0.03*mist_teff)**2. + (rad-mist_rad)**2./2./(0.03*mist_rad)**2. + (logg-mist_logg)**2./2./(0.03*mist_logg)**2. 
    
    lumlsun = 4*np.pi*sigma*teff**4.*(rad*rsun)**2./lsun
    mbolabs = 4.74-2.5*np.log10(lumlsun)
    mbolapp = mbolabs + 5*np.log10(1./par/1.e-3) - 5 
    bc = bc_grid.interp([teff, logg, feh, av])
    
    Jsed, Hsed, Ksed, Gsed, Bpsed, Rpsed, W1sed, W2sed = mbolapp - bc
    chi2_sed = np.log(8*2*np.pi*errscale**2.) + 0.5*((W1sed-W1mag)**2./errscale**2./W1magerr**2.+(W2sed-W2mag)**2./errscale**2./W2magerr**2. +  (Jsed-Jmag)**2./errscale**2./Jmagerr**2. + (Hsed-Hmag)**2./errscale**2./Hmagerr**2. + (Ksed-Kmag)**2./errscale**2./Kmagerr**2. + (Gsed-Gmag)**2./errscale**2./Gmagerr**2. + (Bpsed-BPmag)**2./errscale**2./BPmagerr**2. + (Rpsed-RPmag)**2./errscale**2./(RPmagerr**2.))
    
    prob = chi2_mist+chi2_sed
    #print(mstar, logg, rstar, feh, teff, av, par, np.log10(age), mist_rad, mist_teff,mist_logg,Jsed, Hsed, Ksed, Gsed, Bpsed, Rpsed, chi2_mist, chi2_sed, chi2_harps, prob)
    return -prob

#### RadVel Fitting ####

#read in rv data
jd_ESPRESSO, rv_ESPRESSO, erv_ESPRESSO = np.loadtxt('WASP-47_ESPRESSO.vels', unpack=True, usecols=(0,1,2))

def initialize_model(planet_dict):
    #initial positions based on current theta; note e = 0 bc of usp
    p = planet_dict['p'].value
    t0 = planet_dict['t0'].value
    w =  planet_dict['w'].value
    k = planet_dict['k'].value   #guess here

    nplanets = len(system_list)-1
    planet_letters = {1:'b'}
    #stellar = dict(mstar=system_list[0]['mstar'].value, mstar_err=system_list[0]['mstar'].err)

    #Set a time base that's located in the middle of your RV time series
    time_base = 0.5*(jd_ESPRESSO.min() + jd_ESPRESSO.max())

    #Add starting values for the orbital parameters, taken either from the literature
    #or from your transit fit -- these don't constrain the fit, they just give it somewhere
    #to start from
    anybasis_params = radvel.Parameters(nplanets,basis='per tc e w k', planet_letters=planet_letters)
    anybasis_params['per1'] = radvel.Parameter(value=p)              #Orbital period
    anybasis_params['tc1'] = radvel.Parameter(value=t0)              #Time of conjunction
    anybasis_params['e1'] = radvel.Parameter(value=0.0)              #orbital eccentricity
    anybasis_params['w1'] = radvel.Parameter(value=w)                #longitude of periastron -- this is rarely well constrained, so just start with pi/2
    anybasis_params['k1'] = radvel.Parameter(value=k)                #RV semi-amplitude in m/s
    anybasis_params['dvdt'] = radvel.Parameter(value=0.0)            #RV slope: (If rv is m/s and time is days then [dvdt] is m/s/day)
    anybasis_params['curv'] = radvel.Parameter(value=0.0)            #RV curvature: (If rv is m/s and time is days then [curv] is m/s/day^2)

    #Add an offset term for each of your data sets, these start as 0 and will 
    #be fit to align the data sets with one another     

    # Convert input orbital parameters into the fitting basis
    fitting_basis = 'per tc secosw sesinw k' #There are other options specified in the RadVel paper, but this one is pretty common
    params = anybasis_params.basis.to_any_basis(anybasis_params,fitting_basis)

    mod = radvel.RVModel(params, time_base=time_base)
    
    return mod

def rv_likelihood(planet_dict,star_dict, rv_t, rv, rv_e):

    rv_model = initialize_model(planet_dict)
    rv_model_points = rv_model(rv_t) - star_dict['gamma'].value
    erv2 = rv_e**2 + star_dict['jitter'].value**2 #rv errors here + jitter in quadrature
    return -0.5 * np.sum((rv - rv_model_points) ** 2 / erv2 + np.log(erv2))


###### Log Likelihood ######

"""
This function generates a likelihood for each step of the MCMC.
It stores and updates the values of theta in the correct planet/star 
dictionary, as these dictionaries are called in the various modeling 
functions. It returns the combined likelihoods from light curve, SED,
and RV modeling.
"""

def log_likelihood(theta, freeparams, fixedparams, system_list):
    
    free_param_list = freeparams.unpack                             #creates set of parameter variable names for each planet
    locals()[free_param_list] = theta                               #list of names = list for initial conditions of each free parameter

    fixed_param_list = fixed_params.unpack
    locals()[fixed_param_list] = fixedparams                        #creating fixed parameter list


    #updating planet dictionaries by packing in new theta
    for i in range(len(system_list)-1):                         #for each planet,
        for m in range(len(theta)):                             #and each free parameter that we go through,                                                                                  #if it's not a transit time, just pack it as normal
            if int(freeparams[m].objno) == 0:
                continue
            elif int(freeparams.unpack()[m][-1]) == (i+1):                               #if the subscript (ex. p1, p2) matches the current planet we're looking at
                system_list[i+1][freeparams.unpack()[m][:-1]].value = theta[m]           #evaluate variable and store it in its respective planet dictionary
    
    #updating stellar dictionary from new theta
    for i in range(len(theta)):
        if (int(freeparams[i].objno) == 0) :  #if these are stellar parameters, update star dictionary
            system_list[0][freeparams[i].name].value = theta[i]
        else:
            pass


    #### fitting t0's using residual - perform operations for each transit separately  ####
    planet_likelihood_value = 0
    model = np.zeros(len(t))

    #making batman model for generic multiplanet system
    for i in range(len(system_list)-1):
        model_i = batman_model(system_list[i+1],system_list[0],t) - 1
        model += model_i    
    model += 1

    #calculating likelihood
    likelihood = np.sum((y - model)**2/yerr**2)
    planet_likelihood_value += (-0.5*likelihood)
    
    #if batman returns nans, have the log likelihood instead be -inf
    if (np.isnan(model).all() == True) or (np.sum(np.abs(model-1))<len(model)*1e-5):
        return -np.inf

    #combine with sed and rv likelihoods
    sed_likelihood_value = sed_likelihood(system_list[0])
    #rv_likelihood_value = rv_likelihood(system_list[1],system_list[0], jd_ESPRESSO, rv_ESPRESSO, erv_ESPRESSO)
    #print(rv_likelihood_value)    
        
    return planet_likelihood_value + sed_likelihood_value #+ rv_likelihood_value


####### Log Prior #######

def log_prior(theta, freeparams):
    log_prior = 0

    for i in range(len(theta)):
        prior_i = freeparams[i].apply_prior(theta[i])
        log_prior += prior_i

    return log_prior
 
####### Log Probability #######

def log_probability(theta, freeparams, fixedparams, system_list):
    lp = log_prior(theta, freeparams)

    if not np.isfinite(lp):
        return -np.inf
    probability = lp + log_likelihood(theta, freeparams, fixedparams, system_list)
    #debugline = ''
    #for i in range(len(theta)):
    #    debugline+="%f," % theta[i]
    #debugline+="%f,%f" % (lp, probability)
    
    return probability

####### Phase-Folding #######

def phasefold(t,t0,p):
    return ( ( (t-t0) / p ) - np.floor( (t-t0)/p ) )

####### Binning #######

def bin_data(x,data, bins, xmin, xmax):
    
    binned_data = np.zeros(bins) + np.nan
    centers = np.zeros(bins) + np.nan
    l_bin = (xmax - xmin) / bins  #length of each bin
    
    #define intervals based on number of bins, take average of that interval and then plot that
    for i in range(bins):
        mask1 = xmin+(l_bin*i) < x 
        mask2 = x < xmin+(l_bin*(i+1))
        
        interval = data[mask1 * mask2]
        
        bincenter = np.nanmedian(interval)
        centers[i] = bincenter
        
        mean = np.nanmean(interval)
        binned_data[i] = mean
        
    return binned_data, centers

##### Converting to Variable #####

def convert_to_variable(input_row):
    var_prior = PriorFunction(input_row['prior'], center_value=input_row['value'], error = input_row['error'], upper_bound= input_row['upper_bound'], lower_bound=input_row['lower_bound'],spread=input_row['spread'])
    var_dict  = Variable(input_row['name'],input_row['objectno'],input_row['value'],input_row['error'], prior = var_prior, flag = input_row['flag'])

    return var_dict

##### Reading input csv #####

"""
This function will read the config csv, which has all the info about
each free parameter and the priors on them. It will sort the input params
into free vs fixed param lists, and return:

1. 'System List': contains dictionaries of variables belonging to each
member of the system. (star dictionary, planet 1 dictionary, etc.)
2. 'Free params': list containing the names of all free parameters
3. 'Fixed params': list containing names of all fixed parameters
4. 'True values': list of the initial values of each variable, for corner plotting

"""
def read_fit_param_csv(input_csv):
    exofop_data = pd.read_csv(input_csv, comment='#',header=0)  #prior data
    nplanet = exofop_data['objectno'].max()

    #### Adding transit times to prior dataframe ####
    #exofop_data = pd.concat([exofop_data,t0_df], ignore_index=True)

    system_list = []
    
    #################################################
    #### creating individual planet dictionaries ####
    #################################################

    for obj in range(nplanet+1):                                          #for each object in the system
        if obj == 0:                                                      #if it's a star, add to separate star dictionary
            exofop_data_i = exofop_data[exofop_data['objectno'] == obj]   #only look at part of csv that applies to the object you're looking at
            dict_name = 'star_dict'
            star_dict = {}

            for index, row in exofop_data_i.iterrows():
                v = convert_to_variable(row)
                star_dict[row['name']] = v
            system_list.append(star_dict)

        else:
            exofop_data_i = exofop_data[exofop_data['objectno'] == obj]   #all parameters for specific planet + its different transit times
            
            #transit_times = np.array(exofop_data_m['epoch'][exofop_data_m['name']=='t0'])                   #list showing all of the epochs of data you have
            #ntransits = len(transit_times)

            #list_name = f'planet{obj}_list'                               #'master' planet list that will hold all dictionaries for different transits of the same planet (different t0's)
            #locals()[list_name] = []
            dict_name = f'planet{obj}_dict'
            
            #for i in range(ntransits):                                #iterate through each transit to create transit dictionaries for a given planet
            #    epoch = transit_times[i]
            #    dict_name = f'transit{epoch}_dict'
            locals()[dict_name] = {}

                #condition = (exofop_data_m['name']=='t0') * (exofop_data_m['epoch'] != epoch)
                #exofop_data_i = exofop_data_m.drop(exofop_data_m[condition].index)     #drop all rows with transit times not pertaining to the one you want to look at

            for index, row in exofop_data_i.iterrows():                       #assign variables to proper keys in individual planet or star dictionary
                v = convert_to_variable(row)
                locals()[dict_name][row['name']] = v                          #example: planet1_dict['p'] = p variable with all associated attributes (name, error, etc)

            system_list.append(eval(dict_name))

    #uncertainty floors on stellar parameters
    if system_list[0]['teff'].err < 100:
        system_list[0]['teff'].err = 100
    if system_list[0]['feh'].err <.05:
        system_list[0]['feh'].err = .05

    #########################################################
    #### sorting variables into free vs fixed parameters ####
    #########################################################

    freeparams = Params()
    fixed_params = Params()
    true_values = []

    for i in range(len(system_list)):                           #for each member of the system...
        for key in (system_list[i]):                            #and each parameter of that planet...
            if system_list[i][key].flag == 'free':              #decide if it's free (based on flag), and if so, add the NAME of the parameter to a list
                if i == 0:                                      #don't change the name of stellar parameters because they're shared
                    freeparams.add(system_list[i][key])
                    true_values.append(system_list[i][key])
                else:
                    temp = deepcopy(system_list[i][key])        #creates temporary variable (keeps same properties as original) that will be added to free parameter list
                    temp.name = f'{key}{i}'
                    freeparams.add(temp)
                    true_values.append(temp)                    #adds true value to separate "truth" list to be used in corner plot before it's overwritten
                
            elif system_list[i][key].flag == 'fixed':
                if i == 0:
                    fixed_params.add(system_list[i][key])
                else:
                    temp = deepcopy(system_list[i][key])        #same creation of temporary variable
                    temp.name = f'{key}{i}'
                    fixed_params.add(temp)  

            #for transit in range(1,len(system_list[i])):         #iterate through each transit to add specific t0 -> t0 for each transit will be treated as its own free parameter
            #    epoch = system_list[i][transit]['t0'].epoch
                
            #    temp = deepcopy(system_list[i][transit]['t0'])
            #    temp.name = f't0{i}_{epoch}'
                
            #    if system_list[i][transit]['t0'].flag == 'free':
            #        freeparams.add(temp)
            #        true_values.append(temp)

            #    elif system_list[i][transit]['t0'].flag == 'free':
            #        fixed_params.add(temp)
                
            
    return system_list, freeparams, fixed_params, true_values

##### Functions used for detrending #####

def polynomial(t,c0,c1,c2,c3):
    return c3*t**3 + c2*t**2 + c1*t + c0

def normalize(t):
    return (t - np.nanmean(t))/len(t)

def least_sq(theta_p, mask, residual, lc_err):
    c0,c1,c2,c3 = theta_p[0],theta_p[1],theta_p[2],theta_p[3]
    model = polynomial(normalize(t[mask]), c0, c1, c2, c3)
    #print(np.sum((residual - model)**2 / lc_err**2))

    return (residual - model) / lc_err

##########################################################
#################### Setting Up Data  ####################
##########################################################
if __name__ == '__main__':

    system_list, freeparams, fixed_params, true_values = read_fit_param_csv(filepath+priorcsv)  #koi134_prior_notdvs.csv


    ##########################################################
    ###################### Running MCMC  #####################
    ##########################################################

    #initialize walkers (define their starting positions)

    theta_test, nwalkers, ndim, pos2 = freeparams.initialize_theta()
    pos = np.swapaxes(pos2,0,1)

    #log_prob = log_probability(theta_test, freeparams, fixed_params)

    print(log_probability(theta_test,freeparams,fixed_params,system_list))
    exit()

    output = 'ascii'
    

    ########## Parallelization ##########
   
    from multiprocessing import Pool

    with Pool(processes=60) as pool:
        if output == 'h5':
            filename = "chains.h5"
            backend = emcee.backends.HDFBackend(filename)
            backend.reset(nwalkers, ndim)

            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(freeparams, fixed_params, system_list),backend=backend, pool=pool)
            max_n = 75000

            # We'll track how the average autocorrelation time estimate changes
            index = 0
            autocorr = np.empty(max_n)

            # This will be useful to testing convergence
            old_tau = np.inf
    
            # Now we'll sample for up to max_n steps
            for sample in sampler.sample(pos, iterations=max_n, progress=True):
                # Only check convergence every 100 steps
                if sampler.iteration % 100:
                    continue

                # Compute the autocorrelation time so far
                # Using tol=0 means that we'll always get an estimate even
                # if it isn't trustworthy
                tau = sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index += 1

                # Check convergence
                converged = np.all(tau * 100 < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                if converged:
                    break
                old_tau = tau
        elif output == 'ascii':
            max_n = 75000

            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(freeparams,fixed_params,system_list),pool=pool)
            os.system("rm -rf "+out_folder+"mcmc_tested_params_nottvs")
            os.system("rm -rf "+out_folder+ "testparams_step_nottvs")

            header2 = freeparams.unpack()
            header2.append('likelihood')
            header = np.array(header2)

            np.savetxt(out_folder+'testparams_step_nottvs', header.reshape(1,header.shape[0]),fmt='%s')
            os.system('cat '+out_folder+'testparams_step >> '+ out_folder+'mcmc_tested_params_nottvs')

            master_pos = np.ones([100*nwalkers,ndim+1])
            counter = 0
            
            for sample in sampler.sample(pos, iterations=max_n, store=False):
                position = sample.coords
                probability=sample.log_prob
                
                #print(len(position))
                #print(master_pos.shape)
                #print(len(probability))
                for i in range(len(position)):
                    master_pos[counter] = np.array(list(position[i])+[probability[i]])
                    counter += 1
                    
                    if counter >= 100 * nwalkers:
                        np.savetxt(out_folder+"testparams_step_nottvs", master_pos, fmt='%.10f')
                        os.system("cat "+ out_folder +"testparams_step >> " + out_folder + "mcmc_tested_params_nottvs")
                        counter = 0
