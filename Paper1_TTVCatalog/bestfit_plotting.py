from multiplanet_fitting_script import batman_model, Variable, Params, read_fit_param_csv, phasefold, bin_data
import numpy as np
import matplotlib.pyplot as plt
import batman
import pandas as pd
import corner
import emcee
from IPython.display import display, Math

data = pd.read_csv("/home/enabbie/5126_lightcurve.csv", header=0)  #light curve
t = np.array(data['time'])
flux = np.array(data["flux"])
err = np.array(data["fluxerr"]) #error


#read in chains
reader = emcee.backends.HDFBackend('chains.h5')


#discard the initial 100 steps, thin by 15 steps, and flatten the chain to get a flat list of samples (taken from emcee documentation)
flat_samples = reader.get_chain(discard=100, thin=15, flat=True)

tau = reader.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
burnin=100
samples = reader.get_chain(discard=burnin,thin=thin, flat=True) #add back thin=thin
log_prob_samples = reader.get_log_prob(discard=burnin,thin=thin, flat=True) #same here

#plotting log probability vs samples
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(samples[:,0],log_prob_samples,linestyle='None')
plt.xlabel("samples")
plt.ylabel("log probability")

fig.savefig('/home/enabbie/5126_fitresults/log_prob_plot.png',dpi=300)

#plotting chains
fig, axes = plt.subplots(10, figsize=(10, 7), sharex=True) #CHANGE NUMBER OF SUBPLOTS AND LABELS
samples = reader.get_chain()

system_list, freeparams, fixed_params, true_values = read_fit_param_csv('/home/enabbie/exofop_prior.csv')
ndim = len(freeparams.unpack())

labels = [freeparams[i].name for i in range(len(freeparams.unpack()))]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");

fig.savefig('/home/enabbie/5126_fitresults/chains_plot.png',dpi=300)

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
    #print(txt)

### Updating dictionary with best fit values ###

for i in range(ndim):
    if (freeparams[i].objno == 0 ):  #if these are stellar parameters, update star dictionary
        system_list[0][labels[i]].value = np.percentile(flat_samples[:,i],[50]).item()
    else:
        system_list[int(labels[i][-1])][labels[i][:-1]].value = np.percentile(flat_samples[:,i],[50]).item()  #if they're planet parameters, match them to correct planet
        system_list[int(labels[i][-1])][labels[i][:-1]].err = np.percentile(flat_samples[:,i],[84]).item()    #also add error **CHECK THIS**


########################################
########### plotting results ###########
########################################
bf_model = np.zeros(len(t))
for i in range(len(system_list)-1):
    bf_model_i = batman_model(system_list[i+1], system_list[0],t) - 1   #model of system using best fit parameters
    bf_model += bf_model_i
bf_model += 1

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(t, flux,'o', label="data")                               #data
#plt.plot(t, flux-bestfitresults,'o', label="simulated data")     #residual
plt.plot(t, bf_model, "k", label="model")                  #batman model using best fit results
plt.legend(loc="lower right")
#plt.ylim(0.996, 1.005)
#plt.xlim(1650,1670)
plt.xlabel("t")
plt.ylabel("flux");

fig.savefig('/home/enabbie/5126_fitresults/bestfit_system_lightcurve.png',dpi=300)


#phase-folded plots of individual planets with best fit data
for i in range(len(system_list)-1):

    phase = phasefold(t,system_list[i+1]['t0'].value,system_list[i+1]['p'].value)
    binned_phase, center = bin_data(phase,phase,500,0,1)

    phase_mask = phase > .5
    phase[phase_mask] -= 1

    edited_phase, center = bin_data(phase,phase,500,-.5,.5)
    binned_flux, center = bin_data(phase,flux,500,-.5,.5)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.title(f"lightcurve of planet {i+1} with best fit model")
    plt.xlim(-.1,.1)
    plt.scatter(edited_phase,binned_flux, label = "binned data")
    plt.plot(edited_phase, batman_model(system_list[i+1],system_list[0],(edited_phase*system_list[i+1]['p'].value)+system_list[i+1]['t0'].value),c='red',label="model")
    plt.legend()

    fig.savefig(f'/home/enabbie/5126_fitresults/planet{i+1}_bestfit_lightcurve.png',dpi=300)

#histogram
#plt.hist(flux-bestfitresults,bins=20) 

print(true_values)
print(flat_samples.size)

#corner plot
fig = corner.corner(
    flat_samples, labels=labels, truths=[true_values[i].value for i in range(len(true_values))])
fig.savefig('/home/enabbie/5126_fitresults/cornerplot.png',dpi=300)
