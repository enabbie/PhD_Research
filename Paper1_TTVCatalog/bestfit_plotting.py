from multiplanet_fitting_script import batman_model, Variable, Params, read_fit_param_csv, phasefold, bin_data, log_probability
import numpy as np
import matplotlib.pyplot as plt
import batman
import pandas as pd
import corner
import emcee
from IPython.display import display, Math


data = pd.read_csv("/home/enabbie/KOI134_seg.csv", header=0)  #light curve
t = np.array(data['time'])
flux = np.array(data["flux"])
err = np.array(data["err"]) #error


#read in chains
#reader = emcee.backends.HDFBackend('chains.h5')

chains = np.loadtxt('mcmc_tested_params')
chain_df = pd.DataFrame(chains).tail(10000)


#discard the initial 100 steps, thin by 15 steps, and flatten the chain to get a flat list of samples (taken from emcee documentation)
#flat_samples = reader.get_chain(discard=100, thin=15, flat=True)

#tau = reader.get_autocorr_time()
#burnin = int(2 * np.max(tau))
#thin = int(0.5 * np.min(tau))
#burnin=100
#samples = reader.get_chain(discard=burnin,thin=thin, flat=True) #add back thin=thin
#log_prob_samples = reader.get_log_prob(discard=burnin,thin=thin, flat=True) #same here

#plotting log probability vs samples
#fig = plt.figure()
#ax = fig.add_subplot(111)
#plt.plot(samples[:,0],log_prob_samples,linestyle='None')
#plt.xlabel("samples")
#plt.ylabel("log probability")

#fig.savefig('/home/enabbie/5126_fitresults/log_prob_plot.png',dpi=300)

#plotting chains
system_list, freeparams, fixed_params, true_values = read_fit_param_csv('/home/enabbie/koi134_prior.csv')

ndim = len(freeparams.unpack())
fig, axes = plt.subplots(ndim, figsize=(ndim, 10), sharex=True) #CHANGE NUMBER OF SUBPLOTS AND LABELS
#samples = reader.get_chain()

labels = [freeparams[i].name for i in range(len(freeparams.unpack()))]
for i in range(ndim):
    ax = axes[i]
    #ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.plot(range(len(chain_df[0])), chain_df[i])
    ax.set_xlim(0, len(chain_df[0]))
    ax.set_ylabel(labels[i],rotation='horizontal')
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");

fig.savefig('/home/enabbie/koi134_fitresults/chains_plot.png',dpi=300)

#############################################################
###########         Obtain results, then          ###########
########### re-pack and store parameters in their ###########
###########      proper planet dictionaries       ###########
#############################################################

#displaying best fit parameters
bestfit_df = pd.DataFrame(columns=['name','value','lower_e','upper_e','1sgima'])

bestfit_values = []
upper_err = []
lower_err = []
sig_arr = []
epochs = []
centers = []
t_err = []

for i in range(ndim):
    #mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    mcmc = np.percentile(chain_df[i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.7f}_{{-{1:.7f}}}^{{{2:.7f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])

    bestfit_values.append(mcmc[1])
    upper_err.append(q[1])
    lower_err.append(q[0])

    onesig_err = np.sqrt(q[0]*q[1])
    sig_arr.append(onesig_err)

    #display(Math(txt))
    print(txt)

    if labels[i][:2] == 't0':
         word = labels[i]
         var_name = word.split('_')[0]                   #omits everything after '_' -> ex. t01_3 becomes t01
         transit_number = eval(word.split('_')[1])       #saves transit number as an integer (so we still know, for example, that t01_3 corresponds to transit 3)
         var_subscript = var_name[-1]                    #do this to get the object number

         centers.append(mcmc[1])
         epochs.append(transit_number)
         t_err.append(onesig_err)
         

bestfit_df["name"] = labels
bestfit_df["value"] = bestfit_values
bestfit_df["lower_e"] = lower_err
bestfit_df["upper_e"] = upper_err
bestfit_df["1sigma"] = sig_arr       #1 sigma error

bestfit_df.to_csv('/home/enabbie/koi134_fitresults/bestfit_values.csv')

transittimes_df = pd.DataFrame(columns=['epoch','time','err'])
transittimes_df['epoch'] = epochs
transittimes_df['time'] = centers
transittimes_df['err'] = t_err

transittimes_df.to_csv('/home/enabbie/koi134_fitresults/fitted_transittimes.csv')

exit()


### Updating dictionary with best fit values ###

for i in range(ndim):
    if (freeparams[i].objno == 0 ):  #if these are stellar parameters, update star dictionary
        system_list[0][labels[i]].value = np.percentile(flat_samples[:,i],[50]).item()
    else:
        system_list[int(labels[i][-1])][labels[i][:-1]].value = np.percentile(flat_samples[:,i],[50]).item()  #if they're planet parameters, match them to correct planet
        system_list[int(labels[i][-1])][labels[i][:-1]].err = np.percentile(flat_samples[:,i],[84]).item()    #also add error **CHECK THIS**

theta_test = []

for i in range(len(freeparams.unpack())):                  #add free parameters to test list one by one
    initial_guess = f'{freeparams.unpack()[i]}_test'       #this will make a variable called p1_test, p2_test, etc.
    locals()[initial_guess] = freeparams.data[i].value     #p1 = number, p2 = number, ...

    theta_test.append(locals()[initial_guess])


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

    phase = phasefold(t, system_list[i+1]['t0'].value, system_list[i+1]['p'].value)
    binned_phase, center = bin_data(phase,phase,500,0,1)

    phase_mask = phase > .5
    phase[phase_mask] -= 1

    edited_phase, center = bin_data(phase,phase,500,-.5,.5)
    
    if i ==0:
        binned_flux, center = bin_data(phase,1+flux-batman_model(system_list[2],system_list[0],t),500,-.5,.5)
    elif i ==1: 
        binned_flux, center = bin_data(phase,1+flux-batman_model(system_list[1],system_list[0],t),500,-.5,.5)

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
