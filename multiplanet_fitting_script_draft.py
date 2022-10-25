import batman
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import corner
import emcee
from IPython.display import display, Math


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

class Variable:
    def __init__(self, name, value, err=0, prior='none', flag='fixed'):
        self.name = name
        self.value = value
        self.err = err
        self.flag = flag
        self.prior = prior

    def apply_prior(self,x):
        return self.prior(x)

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
            if x < self.lower_bound:
                return -np.inf
        if self.priortype == 'gaussian':
            return -.5 * (x - self.center_value)**2/(self.upper_bound - self.center_value)**2
        if self.priortype == 'none':
            return 0


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
    params.ecc = input_dict['ecc'].value                        #eccentricity
    params.w = input_dict['w'].value                            #longitude of periastron (in degrees)
    params.u = [star_dict['u1'].value, star_dict['u2'].value]   #limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic"                              #limb darkening model

    m = batman.TransitModel(params, time)                       #initializes model
    flux = m.light_curve(params)
    return flux

###### Log Likelihood ######

def log_likelihood(theta,t,y,yerr, fixedparams):
    free_param_list = freeparams.unpack                         #creates set of parameter variable names for each planet
    locals()[free_param_list] = theta                           #list of names = list for initial conditions of each free parameter

    fixed_param_list = fixed_params.unpack
    locals()[fixed_param_list] = fixedparams                    #creating fixed parameter list

    #updating planet dictionaries by packing in new theta
    for i in range(len(system_list)-1):                         #for each planet
        for m in range(len(theta)):                             #and each free parameter that we go through
            if freeparams.unpack[m][-1] == (i+1):               #if the subscript (ex. p1, p2) matches the current planet we're looking at
                system_list[i+1][freeparams.unpack[m][:-1]].value = eval(freeparams.unpack[m])   #evaluate the variable and store it in its respective planet dictionary

    for i in range(len(theta)):
        if (freeparams[i].name == 'mstar') or (freeparams[i].name == 'rstar') or (freeparams[i].name == 'u1') or (freeparams[i].name == 'u2'):  #if these are stellar parameters, update star dictionary
            system_list[0][freeparams[i].name].value = theta[i]
        else:
            system_list[int(freeparams[i].name[-1])][freeparams[i].name[:-1]].value = theta[i]  #if they're planet parameters, match them to correct planet

    model = np.zeros(t)
    for i in range(len(system_list)-1):
        model_i = batman_model(system_list[i+1], system_list[0],t) - 1   #model for multiplanet system using superposition of individual planet models
        model += model_i
    model += 1

    sigma2 = yerr**2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))


####### Log Prior #######

def log_prior(theta, freeparams):
    log_prior = 0

    for i in range(len(theta)):
        prior_i = freeparams[i].apply_prior(theta[i])

        log_prior += prior_i

    return log_prior
 
####### Log Probability #######

def log_probability(theta, t, y, yerr, freeparams, fixedparams):
    lp = log_prior(theta, freeparams)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, t, y, yerr,fixedparams)


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



##########################################################
#################### Setting Up Data  ####################
##########################################################

#loading in data from csv file

data = pd.read_csv("/home/enabbie/4342_lightcurve.csv", header=0)  #light curve
exofop_data = pd.read_csv("C:/Users/enabb/OneDrive/Documents/exofop_prior.csv", header=0)  #data compiled from exofop

t = np.array(data["time"])
flux = np.array(data["flux"])
err = np.array(data["fluxerr"]) #error

nplanet = len(exofop_data['planet'])

#System list to hold all the planet dictionaries -> build one planet at a time
system_list = [] 

#individual planet dictionaries
for i in range(nplanet):

    #read in exofop data and store for use as prior values -> build parameter classes one at a time

    #period
    p = Variable(exofop_data['period'][i],exofop_data['period_e'][i], flag = exofop_data['p_flag'][i],prior = exofop_data['p_prior'][i])
    
    #t0
    t0 = Variable((exofop_data['t0'][i]- 2457000), exofop_data['t0_e'][i],prior = exofop_data['t0_prior'][i],flag = exofop_data['t0_flag'][i])
    
    #u1
    u1 = Variable(.1, exofop_data['u1_e'][i],prior = exofop_data['u1_prior'][i],flag = exofop_data['u1_flag'][i])


    #u2
    u2 = Variable(.1, exofop_data['u2_e'][i],prior = exofop_data['u2_prior'][i],flag = exofop_data['u2_flag'][i])
    
    #rstar
    rstar = Variable(exofop_data['rstar'][i], exofop_data['rstar_e'][i], prior = exofop_data['rstar_prior'][i], flag = exofop_data['rstar_flag'])
    
    #mstar
    mstar = Variable(exofop_data['mstar'][i], exofop_data['mstar_e'][i], prior = exofop_data['mstar_prior'][i], flag = exofop_data['mstar_flag'])

    #rprs
    #rprs = (exofop_data['rp_re'][i] * 0.00916794) / exofop_data['rstar'][i]                   #ratio of earth radius to solar radius, divided by rstar (in solar radii)
    #rprs_dict = {'value': rprs, 'err': exofop_data['rp_e'][i],
    #            'prior': exofop_data['rprs_prior'][i], 'flag': exofop_data['rprs_flag'][i]}   #** do conversion for rprs in csv file instead
    rprs = Variable(exofop_data['rprs'][i], exofop_data['rprs_e'][i],prior = exofop_data['rprs_prior'][i], flag = exofop_data['rprs_flag'][i])
    
    #b
    b = Variable(exofop_data['b'][i], exofop_data['b_e'][i], prior = exofop_data['b_prior'][i], flag = exofop_data['b_flag'][i])

    if i == 0:
        star_dict = {'u1': u1, 'u2': u2, 'mstar': mstar, 'rstar': rstar}
        system_list.append(star_dict)
    else:
        dict_name = f'planet{i}_dict'                                                #naming scheme: planet1_dict, etc.
        locals()[dict_name] = {'p': p, 't0': t0, 'rprs': rprs, 'b': b}               #creates dictionary for each planet using assigned name

        system_list.append(locals()[dict_name])                                      #appending dictionary to list


##########################################################
##############     Sorting variables     #################
############ into free vs fixed parameters ###############
##########################################################

freeparams = Params()
fixed_params = Params()
true_values = []

for i in range(len(system_list)):                           #for each planet...
    for key in (system_list[i]):                            #and each parameter of that planet...
        if system_list[i][key].flag == 'free':              #decide if it's free (based on flag), and if so, add the NAME of the parameter to a list
            if i == 0:                                      #don't change the name of stellar parameters because they're shared
                freeparams.add(system_list[i][key])
            else:
                temp = system_list[i][key].deepcopy         #creates temporary variable (keeps same properties as original) that will be added to free parameter list
                temp.name = f'{key}{i}'
                freeparams.add(temp)
                true_values.append(temp)                    #adds true value to separate "truth" list to be used in corner plot before it's overwritten
        
        elif system_list[i][key].flag == 'fixed':
            if i == 0:
                fixed_params.add(system_list[i][key])
            else:
                temp = system_list[i][key].deepcopy         #same creation of temporary variable
                temp.name = f'{key}{i}'
                fixed_params.add(temp)                      #add fixed parameter name to separate list



##########################################################
###################### Running MCMC  #####################
##########################################################

#initialize walkers (define their starting positions)


theta_test = []  #possible alternative: np.zeros with the length being a predetermined amount of free parameters (might need flag dictionary for this)

for i in range(len(freeparams.unpack)):                  #add free parameters to test list one by one
    initial_guess = f'{freeparams.unpack[i]}_test'       #this will make a variable called p1_test, p2_test, etc.
    locals()[initial_guess] = freeparams.data[i].value   #p1 = number, p2 = number, ...

    theta_test.append(locals()[initial_guess])

log_probability(theta_test, t, flux, err, freeparams, fixed_params)

pos = theta_test + 1e-4 * np.random.randn(40, len(theta_test))      #number of walkers, number of free parameters
nwalkers, ndim = pos.shape

#sampler = emcee.EnsembleSampler(
#    nwalkers, ndim, log_probability, args=(t, flux, .001,fixedparams)
#)
#sampler.run_mcmc(pos, 10000, progress=True);


########## Parallelization ##########

from multiprocessing import Pool

with Pool(processes=5) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(t,flux,err, freeparams, fixed_params), pool=pool)
    sampler.run_mcmc(pos, 10000, progress=True)

#discard the initial 100 steps, thin by 15 steps, and flatten the chain to get a flat list of samples (taken from emcee documentation)
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

tau = sampler.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)

#plotting log probability vs samples
plt.plot(samples[:,0],log_prob_samples)
plt.xlabel("samples")
plt.ylabel("log probability")

#plotting chains
fig, axes = plt.subplots(10, figsize=(10, 7), sharex=True) #CHANGE NUMBER OF SUBPLOTS AND LABELS
samples = sampler.get_chain()
#labels = ["p1","p2","t01","t02","u1","u2", "rprs1","rprs2", "b1","b2"]
labels = [freeparams[i].name for i in range(len(freeparams))]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");


#############################################################
###########         Obtain results, then          ###########
########### re-pack and store parameters in their ###########
###########      proper planet dictionaries       ###########
#############################################################

#displaying best fit parameters

for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.7f}_{{-{1:.7f}}}^{{{2:.7f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    display(Math(txt))

bf_system_list = []   #create new system list for best fit parameters

#for i in range(nplanet):
#    if i == 0:
#         bf_star_dict = {'u1':, 'u2':, 'mstar':, 'rstar':}
#         bf_system_list.append(bf_star_dict)
#    else: 
#        for m in range(ndim):
#            dict_name = f'bf_planet{i}_dict'
#            locals()[dict_name] = {}
#
#            if labels[m][-1] == i:
#                bestfit = np.percentile(flat_samples[:,i],[50]).item()
#            bf_system_list.append(bestfit)

#print(bf_system_list)

### Updating dictionary with best fit values ###

for i in range(ndim):
    if (labels[i] == 'mstar') or (labels[i] == 'rstar') or (labels[i] == 'u1') or (labels[i] == 'u2'):  #if these are stellar parameters, update star dictionary
        system_list[0][labels[i]].value = np.percentile(flat_samples[:,i],[50]).item()
    else:
        system_list[int(labels[i][-1])][labels[i][:-1]].value = np.percentile(flat_samples[:,i],[50]).item()  #if they're planet parameters, match them to correct planet
        system_list[int(labels[i][-1])][labels[i][:-1]].err = np.percentile(flat_samples[:,i],[84]).item()    #also add error **CHECK THIS**

########################################
########### plotting results ###########
########################################

bf_model = np.zeros(t)
for i in range(len(system_list)-1):
        bf_model_i = batman_model(system_list[i+1], system_list[0],t) - 1   #model of system using best fit parameters
        bf_model += bf_model_i
bf_model += 1

plt.plot(t, flux,'o', label="data")                               #data
#plt.plot(t, flux-bestfitresults,'o', label="simulated data")     #residual
#plt.plot(t, bestfitresults, "k", label="model")                  #batman model using best fit results
plt.legend(loc="lower right")
#plt.ylim(0.996, 1.005)
#plt.xlim(-.5,.5)
plt.xlabel("t")
plt.ylabel("flux");


#phase-folded plots of individual planets with best fit data
for i in range(len(system_list)-1):

    phase = phasefold(t,system_list[i+1]['t0'].value,system_list[i+1]['p'].value)
    binned_phase, center = bin_data(phase,phase,500,0,1)

    phase_mask = phase > .5
    phase[phase_mask] -= 1

    edited_phase, center = bin_data(phase,phase,500,-.5,.5)
    binned_flux, center = bin_data(phase,flux,500,-.5,.5)


    plt.title(f"lightcurve of planet {i+1} with best fit model")
    plt.xlim(-.1,.1)
    plt.scatter(edited_phase,binned_flux, label = "binned data")
    plt.plot(edited_phase, batman_model(bf_system_list[i+1],bf_system_list[0],t),c='red',label="model")
    plt.legend()


#phase-folded plot with planet 2 best fit data
#phase = phasefold(t,t02_bf,p2_bf)
#binned_phase, center = bin_data(phase,phase,500,0,1)

#phase_mask = phase > .5
#phase[phase_mask] -= 1

#edited_phase, center = bin_data(phase,phase,500,-.5,.5)
#binned_flux, center = bin_data(phase,flux,500,-.5,.5)


#plt.title("lightcurve of planet 2 with best fit model")
#plt.xlim(-.1,.1)
#plt.scatter(edited_phase,binned_flux, label = "binned data")
#plt.plot(edited_phase, batman_model(p2_bf,0,u1_bf, u2_bf, rprs2_bf, b2_bf,edited_phase*p2_bf,mstar,rstar),c='red',label="model")
#plt.legend()


#histogram
#plt.hist(flux-bestfitresults,bins=20) 

#corner plot
fig = corner.corner(
    flat_samples, labels=labels, truths=[true_values.value[i] for i in range(len(true_values))])