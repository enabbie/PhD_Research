import batman
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import corner
import emcee
from IPython.display import display, Math
from copy import deepcopy
from isochrones.mist.bc import MISTBolometricCorrectionGrid
from isochrones import get_ichrone
import time


#turn off automatic parallelization to avoid problems with later parallelization method
os.environ["OMP_NUM_THREADS"] = "1"

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
        return '%s:%f:%f:%s' %(self.name, self.value, self.err, self.flag)    #%s = print as string, %f = print as float, %d = print as integer
    def __repr__(self):
        return '%s:%f:%f:%s' %(self.name, self.value, self.err, self.flag)    #%s = print as string, %f = print as float, %d = print as integer

class PriorFunction:
    def __init__(self, priortype, center_value= 'none', upper_bound = 'none', lower_bound = 'none'):
        self.priortype = priortype
        self.center_value = center_value
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def __call__(self, x):           #function for calling different types of priors
        if self.priortype == 'box':
            if x > self.upper_bound:
                return -np.inf
            elif x < self.lower_bound:
                return -np.inf
            else:
                return 0
        if self.priortype == 'gaussian':
            return float(-.5 * (x - self.center_value)**2/(self.upper_bound - self.center_value)**2)
        if self.priortype == 'none':
            return 0

##########################################################
############# Initializing Stellar Fitting ###############
##########################################################

G = 6.67e-8
msun = 1.989e33
rsun = 6.96e10
lsun = 3.828e33
sigma = 5.67e-5

Jmag = 9.043
Jmagerr =.018

Hmag = 8.845
Hmagerr =.015

Kmag = 8.777
Kmagerr = .018

Gmag =  9.945831
Gmagerr = .03

BPmag = 10.222494
BPmagerr = .03

RPmag = 9.519123
RPmagerr = .03

W1mag = 8.712
W1magerr = 0.03

W2mag = 8.743
W2magerr = .03

gaia_par =  6.224005486281034
par_err = .022274612

harps_teff =  6217
harps_teff_err = 125

harps_feh = .17
harps_feh_err =.05  

bc_grid = MISTBolometricCorrectionGrid(['J', 'H', 'K', 'G', 'BP', 'RP', 'WISE_W1', 'WISE_W2'])
mist = get_ichrone('mist')

##########################################################
################## Defining Functions ####################
##########################################################

def batman_model(input_dict, star_dict, time):
    aAU = star_dict['mstar'].value**(1/3)*(input_dict['p'].value/365.25)**(2/3)  #a in AU
    aors = aAU*215/star_dict['rstar'].value                  #a over r_star
    inc = 180/np.pi*np.arccos(input_dict['b'].value/aors)

    params = batman.TransitParams()
    params.t0 = input_dict['t0'].value                          #time of inferior conjunction
    params.per = input_dict['p'].value                          #orbital period
    params.rp = input_dict['rprs'].value                        #planet radius (in units of stellar radii)
    params.a = aors                                             #semi-major axis (in units of stellar radii)
    params.inc = inc                                            #orbital inclination (in degrees)
    params.ecc = 0                        #eccentricity
    params.w = 90                            #longitude of periastron (in degrees)
    params.u = [star_dict['u1'].value, star_dict['u2'].value]   #limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic"                              #limb darkening model

    m = batman.TransitModel(params, time)                       #initializes model
    flux = m.light_curve(params)
    return flux

#### MIST Isochrone Fitting ####

def sed_likelihood(params):
    #mass, age, feh, rad, teff, av, par, errscale = params
    #mstar, age_Gyr, feh, rstar, teff, par, errscale = params
    #av = 0
    age = params['age_Gyr'].value*1.e9
    
    logg = np.log10(G*params['mstar'].value*msun/(params['rstar'].value*rsun)**2.)
    
    try:
        eep = mist.get_eep(params['mstar'].value, np.log10(age), params['feh'].value, accurate =True)
    except RuntimeError:
        return -np.inf
    except ValueError:
        return -np.inf
    mist_teff, mist_logg, mist_rad = mist.interp_value([eep, np.log10(age), params['feh'].value], ['Teff', 'logg', 'radius'])
    
    if mist_rad<0.5 or mist_rad>2.5: #not this -> keep
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


###### Log Likelihood ######

def log_likelihood(theta,t,y,yerr, fixedparams):
    free_param_list = freeparams.unpack                         #creates set of parameter variable names for each planet
    locals()[free_param_list] = theta                           #list of names = list for initial conditions of each free parameter

    fixed_param_list = fixed_params.unpack
    locals()[fixed_param_list] = fixedparams                    #creating fixed parameter list

    #updating planet dictionaries by packing in new theta
    for i in range(len(system_list)-1):                         #for each planet
        for m in range(len(theta)):                             #and each free parameter that we go through
            if freeparams.unpack()[m][-1] == (i+1):             #if the subscript (ex. p1, p2) matches the current planet we're looking at
                system_list[i+1][freeparams.unpack[m][:-1]].value = eval(freeparams.unpack[m])   #evaluate the variable and store it in its respective planet dictionary

    for i in range(len(theta)):
        if (freeparams[i].objno == 0) :  #if these are stellar parameters, update star dictionary
            system_list[0][freeparams[i].name].value = theta[i]
        else:
            system_list[int(freeparams[i].name[-1])][freeparams[i].name[:-1]].value = theta[i]  #if they're planet parameters, match them to correct planet

    model = np.zeros(len(t))
    for i in range(len(system_list)-1):
        model_i = batman_model(system_list[i+1], system_list[0],t) - 1   #model for multiplanet system using superposition of individual planet models
        model += model_i
    model += 1
    
    sed_likelihood_value = sed_likelihood(system_list[0])

    sigma2 = yerr**2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2)) + sed_likelihood_value


####### Log Prior #######

def log_prior(theta, freeparams):
    log_prior = 0

    for i in range(len(theta)):
        prior_i = freeparams[i].apply_prior(theta[i])
        #print(theta[i])
        #print('prior_i=',prior_i)

        log_prior += prior_i
    
    #chi2_harps = 0.5*((teff-harps_teff)**2./harps_teff_err**2. + (feh-harps_feh)**2./harps_feh_err**2.)    (format csv so this happens if the teff and feh are gaussian priors)(harps = center value)
    # (par-gaia_par)**2./(par_err)**2.) (gaia parallax = center value)

    return log_prior
 
####### Log Probability #######

def log_probability(theta, t, y, yerr, freeparams, fixedparams):
    lp = log_prior(theta, freeparams)
    if not np.isfinite(lp):
        return -np.inf
    probability = lp + log_likelihood(theta,t,y,yerr,fixedparams)
    debugline = ''
    for i in range(len(theta)):
        debugline+="%f," % theta[i]
    debugline+="%f,%f" % (lp, probability)
    
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

#### Converting to Variable ####
def convert_to_variable(input_row):
    var_prior = PriorFunction(input_row['prior'], center_value=input_row['value'], upper_bound= input_row['upper_bound'], lower_bound=input_row['lower_bound'])
    var_dict  = Variable(input_row['name'],input_row['objectno'],input_row['value'],input_row['error'], prior = var_prior, flag = input_row['flag'])

    return var_dict

def read_fit_param_csv(input_csv):
    exofop_data = pd.read_csv(input_csv, header=0)  #prior data
    
    nplanet = exofop_data['objectno'].max()
    
    system_list = []
    
    #################################################
    #### creating individual planet dictionaries ####
    #################################################

    for obj in range(nplanet+1):                                          #for each object in the system
        if obj == 0:                                                      #if it's a star, add to separate star dictionary
            exofop_data_i = exofop_data[exofop_data['objectno'] == obj]
            dict_name = 'star_dict'
            star_dict = {}
        else:
            exofop_data_i = exofop_data[exofop_data['objectno'] == obj]
            dict_name = f'planet{obj}_dict'
            locals()[dict_name] = {}

        for index, row in exofop_data_i.iterrows():                       #assign variables to proper keys in individual planet or star dictionary
            v = convert_to_variable(row)
            locals()[dict_name][row['name']] = v
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

    for i in range(len(system_list)):                           #for each planet...
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
                    fixed_params.add(temp)                      #add fixed parameter name to separate list
    return system_list, freeparams, fixed_params, true_values

##########################################################
#################### Setting Up Data  ####################
##########################################################
if __name__ == '__main__':
    #loading in data from csv file

    #data = pd.read_csv("/home/enabbie/4342_lightcurve.csv", header=0)  #light curve
    data = pd.read_csv("/home/enabbie/5126_lightcurve.csv",comment='#', header=0)
    exofop_data = pd.read_csv("/home/enabbie/exofop_prior.csv", header=0)  #data compiled from exofop

    t = np.array(data["time"])
    flux = np.array(data["flux"])
    err = np.array(data["fluxerr"]) #error

    #system_list, freeparams, fixed_params, true_values = read_fit_param_csv("/home/enabbie/exofop_prior.csv")
    system_list, freeparams, fixed_params, true_values = read_fit_param_csv("/home/enabbie/5126_prior.csv")


    ##########################################################
    ###################### Running MCMC  #####################
    ##########################################################

    #initialize walkers (define their starting positions)


    theta_test = []

    for i in range(len(freeparams.unpack())):                  #add free parameters to test list one by one
        initial_guess = f'{freeparams.unpack()[i]}_test'       #this will make a variable called p1_test, p2_test, etc.
        locals()[initial_guess] = freeparams.data[i].value     #p1 = number, p2 = number, ...

        theta_test.append(locals()[initial_guess])

        log_probability(theta_test, t, flux, err, freeparams, fixed_params)

        #print(log_probability(theta_test,t,flux,err,freeparams,fixed_params))

        pos = theta_test + 1e-4 * np.random.randn(40, len(theta_test))      #number of walkers, number of free parameters
        nwalkers, ndim = pos.shape

    filename = "chains.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)


    ########## Parallelization ##########

    from multiprocessing import Pool

    with Pool(processes=5) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(t,flux,err, freeparams, fixed_params),backend=backend, pool=pool)
        max_n = 65000

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

