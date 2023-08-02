import batman
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import corner
import emcee
#from IPython.display import display, Math
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

 
#turn off automatic parallelization to avoid problems with later parallelization method
os.environ["OMP_NUM_THREADS"] = "1"

################################################
##### initializing paths to required files #####
################################################

filepath = '/home/u1153471/koi134_scripts/'
out_folder = '/home/u8015661/emma/koi134_fitresults/toi5143/'
#filepath = '/home/u8015661/emma/koi134_scripts/toi5143/'

priorcsv = str(sys.argv[1])           #name of prior csv

 #loading in data from csv file
data = pd.read_csv(filepath+'toi5143_lc',comment='#', header=0) #reading in light curve file
ntransit_vs_center = pd.read_csv(filepath+'transittimes_TOI5143.csv', comment='#',header=0)  #csv of approximate transit times and durations

#obj_id = "TIC 271772050"
obj_id = "TIC 281837575"

t = np.array(data["time"])
y = np.array(data["flux"])
yerr = np.array(data["err"]) #error

ntransit = np.array(ntransit_vs_center['epoch'])
duration = np.array(ntransit_vs_center['duration'])
#kepler_time_arr = ntransit_vs_center['time']
#tess_time_arr = kepler_time_arr + 2454833 - 2457000
tess_time_arr = ntransit_vs_center['time']
time_error = ntransit_vs_center['err']

ntransit_vs_center['tess_time'] = tess_time_arr
center = ntransit_vs_center['tess_time']

objectno = np.array(ntransit_vs_center['objectno'])
namearr = np.array(['t0']*len(center))
priorarr = np.array(['box']*len(center))
flagarr = np.array(['free']*len(center))
spread_arr = np.array([.0002]*len(center))

t0_df = pd.DataFrame.from_dict({'name':namearr,'objectno':objectno,'epoch':ntransit,'value':center,'error':time_error*10,'lower_bound':(center-(time_error*10)),'upper_bound':(center+(time_error*10)),'prior': priorarr,'spread': spread_arr,'flag':flagarr})

time_mask = []

for j in range(np.max(objectno)):
    ntransit_j = np.array(t0_df[t0_df['objectno']==(j+1)]['epoch'])
    center_j = np.array(t0_df[t0_df['objectno']==(j+1)]['value'])
    duration_j = duration[t0_df['objectno']==(j+1)]

    fit = np.polyfit(ntransit_j, center_j, 1)  #linear fit just to get prediction of p and t0 for masking purposes
    period = fit[0]
    t0_original = fit[1]

    for i in range(len(ntransit_j)):
        t0_i = t0_original + period*ntransit_j[i]
        time_mask_i = (t>(t0_i-6*duration_j[i])) & (t < (t0_i+6*duration_j[i]))  #creates time masks to cut out each individual transit
        time_mask.append(time_mask_i)
        
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

#Gmag = photometry['GAIAmag'][0]
Gmag = 13.619514
Gmagerr = .03

#BPmag = photometry['gaiabp'][0]
BPmag = 13.92255
BPmagerr = .03                       #use error floor of .03 for gaia bands

#RPmag = photometry['gaiarp'][0]
RPmag = 13.157321
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
            #m = batman.TransitModel(params,time,supersample_factor=7,exp_time=.02041667) #for kepler
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
    jitter = planet_dict['jitter'].value
    gamma = (rv_ESPRESSO.max()+rv_ESPRESSO.min())/2 #offset that you need to subtract off to compare models

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
    anybasis_params['gamma_ESPRESSO'] = radvel.Parameter(value=gamma)      

    anybasis_params['jit_ESPRESSO'] = radvel.Parameter(value=jitter)

    # Convert input orbital parameters into the fitting basis
    fitting_basis = 'per tc secosw sesinw k' #There are other options specified in the RadVel paper, but this one is pretty common
    params = anybasis_params.basis.to_any_basis(anybasis_params,fitting_basis)

    mod = radvel.RVModel(params, time_base=time_base)
    
    return mod

def rv_likelihood(theta, rv_t, rv, rv_e):

    rv_model = initialize_model(theta)
    rv_model_points = rv_model(rv_t) - rv_model.params['gamma_ESPRESSO'].value
    erv2 = rv_e**2 #rv errors here
    return -0.5 * np.sum((rv - rv_model_points) ** 2 / erv2 + np.log(erv2))


###### Log Likelihood ######

def log_likelihood(theta, freeparams, fixedparams, system_list):
    
    free_param_list = freeparams.unpack                             #creates set of parameter variable names for each planet
    locals()[free_param_list] = theta                               #list of names = list for initial conditions of each free parameter

    fixed_param_list = fixed_params.unpack
    locals()[fixed_param_list] = fixedparams                        #creating fixed parameter list


    #updating planet dictionaries by packing in new theta
    for i in range(len(system_list)-1):                             #for each planet,
        for tn in range(len(system_list[i+1])):                     #and each transit of that planet,
            for m in range(len(theta)):                             #and each free parameter that we go through,
                if freeparams.unpack()[m][:2] == 't0':              #if this is a transit time, we have to treat it differently
           
                    word = freeparams.unpack()[m]
                    var_name = word.split('_')[0]                   #omits everything after '_' -> ex. t01_3 becomes t01
                    transit_number = eval(word.split('_')[1])       #saves transit number as an integer (so we still know, for example, that t01_3 corresponds to transit 3)
                    var_subscript = var_name[-1]                    #do this to get the object number
                    
                    if (int(var_subscript) == (i+1)) & (int(transit_number) == int(system_list[i+1][tn]['t0'].epoch)):    #must match object AND transit number          
                        system_list[i+1][tn][var_name[:-1]].value = theta[m]                                #evaluate the variable and store it in its respective planet dictionary

                else:                                                                                       #if it's not a transit time, just pack it as normal
                    if freeparams[m].objno == 0:
                        continue
                    if int(freeparams.unpack()[m][-1]) == (i+1):                                            #if the subscript (ex. p1, p2) matches the current planet we're looking at
                        system_list[i+1][tn][freeparams.unpack()[m][:-1]].value = theta[m]                  #evaluate variable and store it in its respective planet dictionary

    for i in range(len(theta)):
        if (freeparams[i].objno == 0) :  #if these are stellar parameters, update star dictionary
            system_list[0][freeparams[i].name].value = theta[i]
        else:
            #system_list[int(freeparams[i].name[-1])][freeparams[i].name[:-1]].value = theta[i]  #if they're planet parameters, match them to correct planet
            pass

    

    #### fitting t0's using residual - perform operations for each transit separately  ####
    planet_likelihood_value = 0
    mask_counter = 0

    for i in range(len(system_list)-1):
        ntransit_number = len(system_list[i+1])  #number of sub-dictionaries in the planet dictionary = # transits

        for m in range(ntransit_number):    #generate likelihood of each transit separately
            #model = np.zeros(len(t))
            #model_i = batman_model(system_list[i+1], system_list[0],t) - 1   #model for multiplanet system using superposition of individual planet models
            #model += model_i
            #model += 1
    
            theta_p = [1,0.01,0.01,0.01] #initial guesses for polynomial coefficients [c0, c1, c2, c3]
            
            t_mask = time_mask[mask_counter]
            restricted_flux = y[t_mask]

            model = batman_model(system_list[i+1][m],system_list[0],t[t_mask])

            residual = restricted_flux - model
            
            detrend_lsq = scipy.optimize.leastsq(least_sq, theta_p, args=(t_mask,residual,yerr[t_mask]))
            
            c0_new = detrend_lsq[0][0]
            c1_new = detrend_lsq[0][1]
            c2_new = detrend_lsq[0][2]
            c3_new = detrend_lsq[0][3]
            
            poly_model = polynomial(normalize(t[t_mask]),c0_new,c1_new,c2_new,c3_new)
            
            likelihood_i = np.sum((residual - poly_model)**2/yerr[t_mask]**2)
            planet_likelihood_value += (-0.5*likelihood_i)
            mask_counter+=1
    
            #if batman returns nans, have the log likelihood be -inf
            if (np.isnan(model).all() == True) or (np.sum(np.abs(model-1))<len(model)*1e-5):
                return -np.inf

    sed_likelihood_value = sed_likelihood(system_list[0])
    #rv_likelihood_value = rv_likelihood(system_list[1])

    #sigma2 = yerr**2
    #return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2)) + sed_likelihood_value
        
        
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
    if input_row['name'] == 't0':
        var_prior = PriorFunction(input_row['prior'], center_value=input_row['value'], error = input_row['error'], upper_bound= input_row['upper_bound'], lower_bound=input_row['lower_bound'],spread=input_row['spread'])
        var_dict  = TransitTime(input_row['name'],input_row['objectno'],input_row['value'],input_row['epoch'], input_row['error'], prior = var_prior, flag = input_row['flag'])
    else:
        var_prior = PriorFunction(input_row['prior'], center_value=input_row['value'], error = input_row['error'], upper_bound= input_row['upper_bound'], lower_bound=input_row['lower_bound'],spread=input_row['spread'])
        var_dict  = Variable(input_row['name'],input_row['objectno'],input_row['value'],input_row['error'], prior = var_prior, flag = input_row['flag'])

    return var_dict

def read_fit_param_csv(input_csv):
    exofop_data = pd.read_csv(input_csv, comment='#',header=0)  #prior data
    nplanet = exofop_data['objectno'].max()

    #### Adding transit times to prior dataframe ####
    exofop_data = pd.concat([exofop_data,t0_df], ignore_index=True)

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
            exofop_data_m = exofop_data[exofop_data['objectno'] == obj]   #all parameters for specific planet + its different transit times
            
            transit_times = np.array(exofop_data_m['epoch'][exofop_data_m['name']=='t0'])                   #list showing all of the epochs of data you have
            ntransits = len(transit_times)

            list_name = f'planet{obj}_list'                               #'master' planet list that will hold all dictionaries for different transits of the same planet (different t0's)
            locals()[list_name] = []
            
            for i in range(ntransits):                                #iterate through each transit to create transit dictionaries for a given planet
                epoch = transit_times[i]
                dict_name = f'transit{epoch}_dict'
                locals()[dict_name] = {}

                condition = (exofop_data_m['name']=='t0') * (exofop_data_m['epoch'] != epoch)
                exofop_data_i = exofop_data_m.drop(exofop_data_m[condition].index)     #drop all rows with transit times not pertaining to the one you want to look at

                for index, row in exofop_data_i.iterrows():                       #assign variables to proper keys in individual planet or star dictionary
                    v = convert_to_variable(row)
                    locals()[dict_name][row['name']] = v                          #example: planet1_dict['p'] = p variable with all associated attributes (name, error, etc)
                locals()[list_name].append(locals()[dict_name])

            system_list.append(eval(list_name))

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
        if i == 0:                                              #add only one copy of the stellar parameters to be fit
            for key in (system_list[i]):
                if system_list[i][key].flag == 'free':
                    freeparams.add(system_list[i][key])
                    true_values.append(system_list[i][key])
                
                elif system_list[i][key].flag == 'fixed':
                    fixed_params.add(system_list[i][key])
        else:
            for key in (system_list[i][0]):                        #and each parameter of that planet...
                if system_list[i][0][key].flag == 'free':              #decide if it's free (based on flag), and if so, add the NAME of the parameter to a list
                    
                    temp = deepcopy(system_list[i][0][key])        #creates temporary variable (keeps same properties as original) that will be added to free parameter list
                    if key == 't0':
                        epoch = system_list[i][0][key].epoch
                        temp.name = f'{key}{i}_{epoch}'             #naming scheme = t01_1, t01_3 for transits 1 and 3 of planet 1
                    else:
                        temp.name = f'{key}{i}'
                    freeparams.add(temp)
                    true_values.append(temp)                    #adds true value to separate "truth" list to be used in corner plot before it's overwritten
           
                elif system_list[i][0][key].flag == 'fixed':
                    temp = deepcopy(system_list[i][0][key])        #same creation of temporary variable
                    if key == 't0':
                        epoch = system_list[i][0][key].epoch
                        temp.name = f'{key}{i}_{epoch}'
                    else:
                        temp.name = f'{key}{i}'
                        fixed_params.add(temp)                  #add fixed parameter name to separate list
            
            for transit in range(1,len(system_list[i])):         #iterate through each transit to add specific t0 -> t0 for each transit will be treated as its own free parameter
                epoch = system_list[i][transit]['t0'].epoch
                
                temp = deepcopy(system_list[i][transit]['t0'])
                temp.name = f't0{i}_{epoch}'
                
                if system_list[i][transit]['t0'].flag == 'free':
                    freeparams.add(temp)
                    true_values.append(temp)

                elif system_list[i][transit]['t0'].flag == 'free':
                    fixed_params.add(temp)
                
            
    return system_list, freeparams, fixed_params, true_values

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

    #print(log_probability(theta_test,freeparams,fixed_params,system_list))
    
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
