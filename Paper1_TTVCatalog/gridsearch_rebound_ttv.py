import rebound
import numpy as np

import pandas as pd
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import os
os.environ['OMP_NUM_THREADS']="1"

# host star property and constants
stellar_mass = 1.17
stellar_radius = 1.8
Mjup_Msun= 0.000954588

# inclination of the transiting planet (take from the transit fit)
inc1 = 88.73

# inclination of the non-transiting planet, fixed for the grid search
inc2 = 87.

# read in the transit times we are fitting for 
df1 = pd.read_csv("transittime1.csv", delim_whitespace=True)
time1 = np.array(df1.time)
epoch1 = np.array(df1.epoch)
err1 = np.array(df1.err)

# compute the linear period of the transiting planet to have a start point for the grid search
fit0 = np.polyfit(epoch1, time1, 1)
linear_P1 = fit0[0]
res1 = 24*60*(time1-fit0[1]-epoch1*linear_P1)



# maximum number of transits we are computing
N1 = 23 
# maximum number of iteration during bisector
maxiter = 10000


def log_likelihood(theta):
    # the likelihood function using n-body integration
    
    # unpack the parameters
    logm1, logm2, p1, p2, sesinw1, secosw1, sesinw2, secosw2 = theta
    m1 = 10**(logm1)
    m2 = 10**(logm2)
    a1 = (p1/365.25)**(2./3.)*(stellar_mass)**(1./3.) 
    a2 = (p2/365.25)**(2./3.)*(stellar_mass)**(1./3.) 
    e1 = sesinw1**2+secosw1**2
    e2 = sesinw2**2+secosw2**2
    w1 = np.arctan2(sesinw1, secosw1)
    w2 = np.arctan2(sesinw2, secosw2)
   
    # initialize the n-body simulation
    sim = rebound.Simulation()
    sim.integrator="whfast"
    sim.add(m=stellar_mass)
    #print(e1, e2)
    #print(w1, w2)
    # note we fixed Omega here, maybe this needs to be part of the leastsquare
    # M can be left as 0 for now
    sim.add(m=m1*Mjup_Msun, a=a1,e=e1, inc =(90-inc1)/180.*np.pi,Omega=82.329609/180.*np.pi,omega=w1,M=0.0)
    sim.add(m=m2*Mjup_Msun, a=a2, e=e2, inc=(90-inc2)/180.*np.pi, Omega=27.931018/180.*np.pi, omega=w2,M=0.0)
    #print("min exit distance",((m1+m2)/3.*Mjup_Msun)**(1./3.)*((a1+a2)/2), a2-a1)
    sim.exit_min_distance = 3*((m1+m2)/3.*Mjup_Msun)**(1./3.)*((a1+a2)/2) 
    sim.move_to_com()
   

    # integrate until N1 transits happened, and record the mid transit times
    transittimes1 = np.zeros(N1)
    dt = p1/365.25*2.*np.pi/10. #0.001
    p = sim.particles
    i = 0
    try:
        while i<N1:
            x_old = p[1].x - p[0].x  # (Thanks to David Martin for pointing out a bug in this line!)
            t_old = sim.t
            sim.integrate(sim.t+dt) # check for transits every dt time units. Note that 0.01 is shorter than one orbit
            t_new = sim.t
            niter = 0
            if x_old*(p[1].x-p[0].x)<0. and p[1].z-p[0].z>0.:   # sign changed (y_old*y<0), planet in front of star (x>0)
                while t_new-t_old>1e-5:   # bisect until prec of 1e-5 reached
                    if x_old*(p[1].x-p[0].x)<0.:
                        t_new = sim.t
                    else:
                        t_old = sim.t
                    sim.integrate( (t_new+t_old)/2.)
                    niter+=1
                    if niter>maxiter:
                        raise RuntimeError("this is too many iterations")
                    #print('#',niter,t_new-t_old)
                transittimes1[i] = sim.t
                i += 1
                sim.integrate(sim.t+dt/10.)       # integrate 0.001 to be past the transit
    except rebound.Encounter as error:
      return -np.inf
 
    # convert time from rebound units to dates and compute residual against observed linear time and epoch
    res_p1_reb = (transittimes1-np.nanmin(transittimes1))*(365.25/2./np.pi)-linear_P1*np.array(range(N1))
    
    # get rid of gapped transits
    res_p1_reb = res_p1_reb[np.in1d(range(N1),epoch1)]
    #plt.plot(epoch1, res_p1_reb, '.')
    #plt.show()
    # convert units to minutes (this is not nesessary, just keep the units to be the same as the err
    res_p1_reb*=60.*24.
    # make sure we offset any constants in the timing of the first transit uning a leastsq
    def func1(x):
        return np.sum((res_p1_reb-res1-x)**2./(err1*24*60)**2.)
    x0 = [0]
    res = minimize(func1, x0)
    baseline1 = res.x
    
    # the final loglikelihood
    logprob = func1(baseline1) 
    return -0.5*logprob 

if __name__ == '__main__':
    
    p1_array = np.linspace(linear_P1-0.2, linear_P1+0.2, 11)
    # initialize the period ratio grid
    # TBD: need to search for 1:2, 2:3, 3:4, 1:3, 4:3, 3:2, 2:1, 3:1
    #Pratio_array = np.linspace(0.475,0.525,55)
    Pratio_array = np.linspace(0.63,0.7,55)
    #Pratio_array = np.linspace(1.45,1.55,105)
    #p2_array = np.linspace(34.55159-0.1, 34.55159+0.1, 20)
    
    
    # initialize the log mass, e and w grids
    # TBD: maybe instead of using grid, do a leastsq instead? need to time the process
    m1_array = np.linspace(np.log10(0.3), np.log10(10), 4) 
    m2_array = np.linspace(np.log10(0.3), np.log10(10), 4) 
    w1_array = np.array([0.,60.,120.,180.,240.])
    w2_array = np.array([0.,60.,120.,180.,240.])
    e1_array = np.array([0.1,0.3,0.5])
    e2_array = np.array([0.1,0.3,0.5])
    
    
    
    # loop over the grid and record the results
    # TBD: need to make this parallel
    for m1 in m1_array:
      for m2 in m2_array:
        for w1 in w1_array:
          for w2 in w2_array:
            for e1 in e1_array:
              for e2 in e2_array:
                for p1 in p1_array:
                  for pratio in Pratio_array:
                    p2 = p1*pratio
                    sesinw1 = e1**0.5*np.sin(w1/180.*np.pi)
                    secosw1 = e1**0.5*np.cos(w1/180.*np.pi)
                    sesinw2 = e2**0.5*np.sin(w2/180.*np.pi)
                    secosw2 = e2**0.5*np.cos(w2/180.*np.pi)
                    theta = [m1, m2, p1, p2, sesinw1, secosw1, sesinw2, secosw2]
                    try:
                      logprob = log_likelihood(theta)
                    except RuntimeError:
                      logprob = -np.inf  
                    
                    # TBD: write to a file instead of print out to terminal
                    print(m1, m2, p1, p2, sesinw1, secosw1, sesinw2, secosw2, logprob) 
